"""MorphoNup graph (134,442 parameters with 9 shape channels in 4 groups).

sequence (147x4) + shape (143x9) -> GroupedJointMotifs at 4 scales -> DyadPool
k-mers (340) -> Dense(32)
concat -> BN -> Dropout -> Dense(64) -> Dropout -> logit
logit + physical adapter correction (bounded) + motif-statistics correction (bounded) -> output logit

Keep the layer construction order: saved weights are matched by layer class and graph order.
Analysis arms switch components off through cfg["network"] (NETWORK lists the switches; all on = MorphoNup).
An input an arm no longer uses is joined with weight 0 (ZeroConnect), so every arm takes the same three inputs.
"""
import numpy as np
import tensorflow as tf

from .layers import (BoundedCorrection, CenterCrop1D, ChannelSubset, ConnectedAnchor, DyadPool, GlobalPool,
                     GroupedJointMotifs,
                     MotifSpectrum, MotifStatistics, PhysicalMoments, StatisticsByWord, ZeroConnect)

L = tf.keras.layers
NETWORK = dict(sequence_motifs=True,     # PWM term of the motifs
               shape_attenuation=True,   # shape-prototype term of the motifs
               kmers=True,               # 1-4-mer composition branch
               motif_statistics=True,    # dimer/trimer statistics correction
               physical_adapter=True,    # physical adapter correction
               dyad_bins=True,           # seven soft dyad bins in every DyadPool (False: mean + max only)
               motif_context=None)       # candidate: None | "conv" | "gru" | "conv_small" | "gru_small" context block


def network_switches(cfg):
    switches = {**NETWORK, **cfg.get("network", {})}
    unknown = set(switches) - set(NETWORK)
    if unknown:
        raise KeyError(f"unknown network switches {unknown}")
    return switches


def adapter(cfg, channels, groups, dyad_bins=True):
    """Physical correction: grouped local convolutions, dyad pooling, moments and Gabor energies -> bounded delta."""
    inputs = {"embedding": L.Input((64,), name="embedding"), "anchor_logit": L.Input((1,), name="anchor_logit"),
              "physics": L.Input((143, channels), name="physics")}
    z = ConnectedAnchor(name="connected_anchor")([inputs["anchor_logit"], inputs["embedding"], inputs["physics"]])
    pieces = []
    for j, indices in enumerate(groups):
        x = ChannelSubset(indices, name=f"group_{j}")(inputs["physics"])
        pieces.append(L.Conv1D(4, 5, padding="same", activation="gelu", name=f"local_group_{j}")(x))
    local = L.Concatenate(name="grouped_local")(pieces)
    context = L.Concatenate(name="physical_context")([
        (DyadPool(name="physical_dyad_pool") if dyad_bins else GlobalPool(name="physical_global_pool"))(local),
        PhysicalMoments(name="physical_moments")(inputs["physics"]),
        MotifSpectrum(name="physical_gabor", trainable=False)(inputs["physics"])])
    context = L.Dense(16, activation="gelu", name="context_embedding")(context)
    context = L.Dropout(cfg["adapter_dropout"], name="context_dropout")(context)
    joint = L.Concatenate(name="adapter_readout_input")([context, inputs["embedding"]])
    delta = L.Dense(1, kernel_initializer="zeros", bias_initializer="zeros", name="zero_output")(joint)
    out = BoundedCorrection(cfg["max_logit_change"], cfg["correction_penalty"], name="bounded_correction")([z, delta])
    return tf.keras.Model(inputs, out, name="physical_adapter")


def build_core(cfg, channels, groups):
    net = network_switches(cfg)
    inputs = {"sequence": L.Input((147, 4), name="sequence"), "kmers": L.Input((cfg["kmer_dim"],), name="kmers"),
              "physics": L.Input((143, channels), name="physics")}
    pools, maps = [], []
    for k, count in cfg["motifs"]:
        matches = GroupedJointMotifs(k, count, groups, sequence=net["sequence_motifs"], physical=net["shape_attenuation"],
                                     name=f"motifs_{k}")([inputs["sequence"], inputs["physics"]])
        pools.append((DyadPool(name=f"dyad_pool_{k}") if net["dyad_bins"] else GlobalPool(name=f"global_pool_{k}"))(matches))
        maps.append(matches)
    if net["motif_context"]:
        # candidate: model how motif matches combine along the sequence (as stacked conv/recurrent models do)
        length = 148 - max(k for k, _ in cfg["motifs"])
        x = L.Concatenate(name="motif_maps")([CenterCrop1D(length, name=f"crop_{k}")(m)
                                              for (k, _), m in zip(cfg["motifs"], maps)])
        if net["motif_context"] in ("conv", "gru"):       # first attempt: wide and unregularised (overfits)
            x = L.Conv1D(64, 9, padding="same", activation="gelu", name="context_conv")(x)
            x = L.MaxPool1D(2, name="context_pool")(x)
            if net["motif_context"] == "gru":
                x = L.Bidirectional(L.GRU(32, return_sequences=True), name="context_gru")(x)
            else:
                x = L.Conv1D(64, 5, padding="same", dilation_rate=2, activation="gelu", name="context_conv_2")(x)
        elif net["motif_context"] in ("conv_small", "gru_small"):   # bottleneck to 32 channels + spatial dropout
            x = L.Conv1D(32, 1, activation="gelu", name="context_bottleneck")(x)
            x = L.Conv1D(32, 9, padding="same", activation="gelu", name="context_conv")(x)
            x = L.SpatialDropout1D(0.25, name="context_dropout_1")(x)
            x = L.MaxPool1D(2, name="context_pool")(x)
            if net["motif_context"] == "gru_small":
                x = L.Bidirectional(L.GRU(16, return_sequences=True), name="context_gru")(x)
            else:
                x = L.Conv1D(32, 5, padding="same", dilation_rate=2, activation="gelu", name="context_conv_2")(x)
            x = L.SpatialDropout1D(0.25, name="context_dropout_2")(x)
        else:
            raise ValueError(f"unknown motif_context {net['motif_context']!r}")
        pools.append((DyadPool(name="dyad_pool_context") if net["dyad_bins"] else GlobalPool(name="global_pool_context"))(x))
    if net["kmers"]:
        pools.append(L.Dense(32, activation="gelu", name="composition_embedding")(inputs["kmers"]))
    h = L.BatchNormalization(name="feature_normalization")(L.Concatenate(name="all_features")(pools))
    h = L.Dropout(cfg["head_dropout"])(h)
    h = L.Dense(64, activation="gelu", name="joint_embedding")(h)
    z = L.Dense(1, name="joint_logit", dtype="float32")(L.Dropout(cfg["head_dropout"])(h))
    out = z
    if net["physical_adapter"]:
        out = adapter(cfg["adapter_settings"], channels, groups, net["dyad_bins"])(
            dict(embedding=h, anchor_logit=z, physics=inputs["physics"]))
    if net["motif_statistics"]:
        stats = MotifStatistics(name="motif_statistics")(inputs["sequence"])
        stats = L.Normalization(name="statistics_normalization")(stats)
        stats = StatisticsByWord(name="statistics_by_word")(stats)
        stats = L.Dense(4, activation="gelu", name="shared_statistic_encoder")(stats)
        stats = L.Dropout(0.25, name="statistics_dropout")(stats)
        stats = L.Flatten(name="word_features")(stats)
        delta = L.Dense(1, kernel_initializer="zeros", bias_initializer="zeros", name="statistics_readout")(stats)
        out = BoundedCorrection(1.0, 0.02, name="statistics_correction")([out, delta])
    used = dict(sequence=net["sequence_motifs"] or net["motif_statistics"], kmers=net["kmers"],
                physics=net["shape_attenuation"] or net["physical_adapter"] or not net["sequence_motifs"])
    unused = [inputs[name] for name, on in used.items() if not on]
    if unused:
        out = ZeroConnect(name="unused_inputs")([out, *unused])
    return tf.keras.Model(inputs, out, name=f"MorphoNup_C{channels}")


def initialize(core, a, b, seed):
    """Prototype centres/PWMs sampled from training rows; statistics normalizer from training rows."""
    rng = np.random.default_rng(seed)
    provenance = []
    for layer in core.layers:
        if not isinstance(layer, GroupedJointMotifs):
            continue
        rows = rng.integers(0, len(a["sequence"]), layer.count)
        starts = rng.integers(0, 148 - layer.k, layer.count)
        strands = rng.integers(0, 2, layer.count)
        seqs, phys = [], []
        for row, start, strand in zip(rows, starts, strands):
            view = (a, b)[strand]
            lo = int(start + layer.k // 2 - 3)
            seqs.append(view["sequence"][row, start:start + layer.k])
            phys.append(view["physics"][row, lo:lo + 3])
            provenance.append(dict(layer=layer.name, row=int(row), start=int(start), strand=int(strand)))
        if layer.sequence:
            layer.motif_logits.assign(np.log(np.asarray(seqs, np.float32) * 0.8 + 0.05))
        if layer.physical:
            layer.physical_centers.assign(np.asarray(phys, np.float32))
    if "motif_statistics" not in [layer.name for layer in core.layers]:
        return dict(seed=seed, prototypes=provenance)
    extractor = core.get_layer("motif_statistics")
    parts = [np.asarray(extractor(v["sequence"][s:s + 128])) for v in (a, b) for s in range(0, len(v["sequence"]), 128)]
    x = np.concatenate(parts).astype(np.float64)
    norm = core.get_layer("statistics_normalization")
    norm.set_weights([x.mean(0).astype(np.float32), np.maximum(x.var(0), 1e-6).astype(np.float32),
                      np.array(len(x), np.int64)])
    norm.finalize_state()
    return dict(seed=seed, prototypes=provenance)
