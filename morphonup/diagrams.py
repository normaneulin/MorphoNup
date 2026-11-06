"""Schematic diagrams of MorphoNup (architecture, shape preprocessing, evaluation protocol) for the README and thesis.

Every dimension and parameter count is read from a freshly built graph (network.build_core with config.SETTINGS and
config.SHAPE), so the diagrams follow the code. Written to docs/figures/ by scripts/draw_diagrams.py.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from . import config
from .figures import INK, INK2, Writer, style

SEQ, SHAPE, JOINT, OUT = "#e3eefb", "#dff3ea", "#f1f0ec", "#fbefd9"      # light tints: sequence, shape, joint, output
EDGE, ARROW = "#56636b", "#50616b"


class Canvas:
    def __init__(self, w, h):
        self.fig, self.ax = plt.subplots(figsize=(w, h))
        self.ax.set(xlim=(0, w), ylim=(0, h))
        self.ax.axis("off")
        self.nodes = {}

    def box(self, key, x, y, w, h, text, color, size=8.0, bold_first=True):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=color,
                                         edgecolor=EDGE, linewidth=0.8))
        lines = text.split("\n")
        if bold_first:
            self.ax.text(x + w / 2, y + h - 0.13, lines[0], ha="center", va="top", fontsize=size + 0.6, weight="bold",
                         color=INK)
            self.ax.text(x + w / 2, y + h - 0.13 - 0.27, "\n".join(lines[1:]), ha="center", va="top", fontsize=size,
                         linespacing=1.38, color=INK)
        else:
            self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.38, color=INK)
        self.nodes[key] = (x, y, w, h)

    def anchor(self, key, side, frac=0.5):
        x, y, w, h = self.nodes[key]
        return dict(top=(x + w * frac, y + h), bottom=(x + w * frac, y), left=(x, y + h * frac), right=(x + w, y + h * frac))[side]

    def arrow(self, start, end, label=None, label_at=None):
        self.ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=ARROW, linewidth=0.9))
        if label:
            x, y = label_at or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            self.ax.text(x, y, label, fontsize=7, color=INK2, ha="center", va="center",
                         bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    def path(self, points, label=None, label_at=None):
        """Polyline with an arrow head on the last segment (routes around boxes)."""
        xs, ys = zip(*points)
        if len(points) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=ARROW, lw=0.9)
        self.arrow(points[-2], points[-1], label, label_at)


def _graph():
    from .network import build_core
    from .preprocess import PhysicalReducer
    reducer = PhysicalReducer.from_spec(config.SHAPE)
    cfg = config.settings()
    core = build_core(cfg, reducer.channels, reducer.groups())
    layer = {l.name: l for l in core.layers}
    adapter = {l.name: l for l in layer["physical_adapter"].layers}
    return cfg, reducer, core, layer, adapter


def architecture(w):
    cfg, reducer, core, layer, adapter = _graph()
    total = core.count_params()
    trainable = sum(int(v.numpy().size) for v in core.trainable_variables)
    scales = [(k, n, layer[f"motifs_{k}"].output.shape[1], layer[f"motifs_{k}"].count_params()) for k, n in cfg["motifs"]]
    pools = [layer[f"dyad_pool_{k}"].output.shape[1] for k, _ in cfg["motifs"]]
    feat = layer["all_features"].output.shape[1]
    ch = reducer.channels
    groups = reducer.groups()
    explicit = ", ".join(reducer.channel_names()[:len(config.SHAPE["explicit"])])
    pls_k = config.SHAPE["blocks"][0]["k"]
    adapter_params = layer["physical_adapter"].count_params()
    stats_params = sum(layer[n].count_params() for n in ("statistics_normalization", "shared_statistic_encoder",
                                                         "statistics_readout", "statistics_correction"))
    head_params = sum(layer[n].count_params() for n in ("feature_normalization", "joint_embedding", "joint_logit"))
    motif_params = sum(s[3] for s in scales)

    c = Canvas(13.6, 9.4)
    c.box("seq", 0.15, 8.05, 3.7, 1.15, "One-hot DNA sequence\nB × 147 × 4 (A, C, G, T)\nforward and reverse-complement view", SEQ)
    c.box("shape", 4.25, 8.05, 5.1, 1.15, "14 DNAshape tracks (DNAshapeR)\n7 base-pair tracks + 6 step tracks averaged onto "
          "base pairs + EP\nB × 143 × 14 per view", SHAPE)
    c.box("kmer", 9.75, 8.05, 3.7, 1.15, "1–4-mer frequencies\nB × 340 (4 + 16 + 64 + 256)\nz-scored on fit rows", SEQ)
    c.box("prep", 4.25, 6.3, 5.1, 1.35, f"Shape preprocessing (fit rows only)\nz-score; {explicit} kept as channels\n"
          f"7 RC-symmetric tracks → dyad-weighted PLS1 → {pls_k}\n"
          f"Buckle, Shear, Shift, Tilt dropped → B × 143 × {ch} ({len(groups)} groups)", SHAPE, size=7.6)
    motif_lines = "\n".join(f"k = {k} bp, {n} motifs → B × {L} × {n}" for k, n, L, _ in scales)
    c.box("motifs", 0.15, 3.95, 5.0, 1.95, f"Grouped joint motifs ({motif_params:,} parameters)\n{motif_lines}\n"
          "match = PWM score × RBF similarity of the central\n3-bp shape patch to the motif's shape prototype", JOINT, size=7.6)
    c.box("adapter", 5.55, 3.95, 3.8, 1.95, f"Physical adapter ({adapter_params:,})\n{len(groups)} grouped Conv1D(4, k = 5) "
          f"→ B × 143 × {adapter['grouped_local'].output.shape[-1]}\nDyadPool {adapter['physical_dyad_pool'].output.shape[1]} "
          f"+ moments {adapter['physical_moments'].output.shape[1]}\n+ fixed Gabor energies {adapter['physical_gabor'].output.shape[1]}"
          f"\n→ Dense {adapter['context_embedding'].output.shape[1]} (GELU)", SHAPE, size=7.6)
    c.box("stats", 9.75, 3.95, 3.7, 1.95, f"Motif statistics ({stats_params:,})\n80 dimers/trimers × 10 descriptors\n"
          "(7 dyad bins + 3 periodicities)\nnormalized on fit rows\nshared Dense(10 → 4) → B × 320", SEQ, size=7.6)
    pool_sum = " + ".join(str(p) for p in pools)
    c.box("head", 0.15, 1.6, 5.0, 1.95, f"DyadPool + head ({head_params:,})\nper scale: mean + max + 7 soft dyad bins\n"
          f"{pool_sum} + k-mer Dense(32) = B × {feat:,}\nBatchNorm → Dropout {cfg['head_dropout']} → Dense 64 (GELU)\n"
          f"→ Dropout {cfg['head_dropout']} → base logit z", JOINT, size=7.6)
    c.box("adout", 5.55, 1.6, 3.8, 1.95, "Adapter readout\n16 shape + 64 head features\nzero-initialized Dense(1) → δ₁\n"
          f"bounded: ±{cfg['adapter_settings']['max_logit_change']:.0f} · tanh(δ₁), penalty "
          f"{cfg['adapter_settings']['correction_penalty']}·mean(δ₁²)", SHAPE, size=7.6)
    c.box("stout", 9.75, 1.6, 3.7, 1.95, "Statistics readout\nDropout 0.25\nzero-initialized Dense(1) → δ₂\n"
          "bounded: ±1 · tanh(δ₂), penalty 0.02·mean(δ₂²)", SEQ, size=7.6)
    c.box("out", 2.4, 0.08, 8.8, 1.1, "Output\np = sigmoid(z + δ₁ + δ₂) for each view;  prediction = mean of forward and "
          f"reverse-complement p;  threshold 0.5\none network per fold: {total:,} parameters ({trainable:,} trainable)", OUT, size=7.8)

    c.arrow(c.anchor("shape", "bottom"), c.anchor("prep", "top"))
    c.path([c.anchor("seq", "bottom", 0.4), (1.63, 5.9)])
    c.path([c.anchor("prep", "bottom", 0.2), (5.27, 6.05), (3.6, 6.05), (3.6, 5.9)], "B × 143 × 9", (4.4, 6.05))
    c.arrow(c.anchor("prep", "bottom", 0.72), c.anchor("adapter", "top", 0.5))
    c.path([c.anchor("seq", "top", 0.5), (2.0, 9.32), (13.55, 9.32), (13.55, 6.15), (11.6, 6.15), (11.6, 5.9)],
           "same sequence", (12.9, 9.32))
    c.path([c.anchor("kmer", "bottom", 0.25), (10.67, 7.6), (9.55, 7.6), (9.55, 3.75), (4.2, 3.75), (4.2, 3.55)],
           f"k-mers → Dense 32 ({layer['composition_embedding'].count_params():,})", (6.3, 3.75))
    c.arrow(c.anchor("motifs", "bottom"), c.anchor("head", "top"))
    c.arrow(c.anchor("adapter", "bottom"), c.anchor("adout", "top"))
    c.arrow(c.anchor("stats", "bottom"), c.anchor("stout", "top"))
    c.arrow(c.anchor("head", "right", 0.45), c.anchor("adout", "left", 0.45), "64 + z", (5.35, 2.65))
    c.arrow(c.anchor("head", "bottom", 0.6), (4.5, 1.18))
    c.arrow(c.anchor("adout", "bottom"), (7.45, 1.18))
    c.arrow(c.anchor("stout", "bottom", 0.4), (10.4, 1.18))
    c.ax.legend(handles=[Patch(facecolor=SEQ, edgecolor=EDGE, label="sequence-derived"),
                         Patch(facecolor=SHAPE, edgecolor=EDGE, label="shape-derived"),
                         Patch(facecolor=JOINT, edgecolor=EDGE, label="sequence × shape"),
                         Patch(facecolor=OUT, edgecolor=EDGE, label="output")],
                loc="lower left", bbox_to_anchor=(0.0, -0.01), fontsize=7, ncol=1, frameon=False)
    w.save(c.fig, "architecture", f"MorphoNup computational graph (B = batch). Each sequence is presented as its forward and "
           f"reverse-complement view; the same network scores both and the probabilities are averaged. {total:,} "
           f"parameters, {trainable:,} trainable. All normalization statistics, PLS rotations and motif prototypes are "
           "fitted on the fit rows of each outer fold.",
           pd.DataFrame([dict(layer=l.name, type=type(l).__name__, output=str(getattr(l, "output").shape),
                                                parameters=l.count_params()) for l in core.layers]))


def shape_preprocessing(w):
    from .preprocess import ANTISYMMETRIC, NAMES14
    explicit = config.SHAPE["explicit"]
    block = config.SHAPE["blocks"][0]
    c = Canvas(11.0, 4.6)
    rows = [("Kept as channels (z-scored)", explicit, SHAPE),
            (f"Reduced: dyad-weighted PLS1 → {block['k']} components", block["channels"], JOINT),
            ("Not used (sign flips on the reverse strand)", ANTISYMMETRIC, "#f4e4e2")]
    assert sorted(explicit + block["channels"] + ANTISYMMETRIC) == sorted(NAMES14)
    y = 3.35
    for i, (title, names, color) in enumerate(rows):
        c.box(f"r{i}", 0.2, y - 1.3 * i, 4.6, 0.95, f"{title}\n{', '.join(names)}", color, size=7.8)
    c.box("z", 5.6, 3.35, 2.4, 0.95, f"{len(explicit)} channels\nz = (x − mean) / SD", SHAPE, size=7.8)
    c.box("pls", 5.6, 2.05, 2.4, 0.95, f"{block['k']} channels\nscores re-standardized", JOINT, size=7.8)
    c.box("drop", 5.6, 0.75, 2.4, 0.95, "0 channels\nzero PLS loadings by symmetry", "#f4e4e2", size=7.8)
    c.box("out", 8.6, 1.7, 2.2, 1.6, f"Shape input\nB × 143 × {len(explicit) + block['k']}\n{len(explicit) + 1} groups:\n"
          f"{', '.join(explicit)}, PLS block", SHAPE, size=7.8)
    for a, b in (("r0", "z"), ("r1", "pls"), ("r2", "drop")):
        c.arrow(c.anchor(a, "right"), c.anchor(b, "left"))
    c.arrow(c.anchor("z", "right"), c.anchor("out", "left", 0.75))
    c.arrow(c.anchor("pls", "right"), c.anchor("out", "left", 0.35))
    c.ax.text(0.2, 0.35, f"Fitted on the fit rows of each outer fold, forward and reverse-complement views pooled; PLS weights "
              f"each position by a Gaussian centred on the dyad (σ = {config.SHAPE['dyad_sigma']:.0f} bp).", fontsize=7.5,
              color=INK2)
    w.save(c.fig, "shape_preprocessing", "From 14 DNAshape tracks to the 9 shape channels MorphoNup uses. Buckle, Shear, "
           "Shift and Tilt change sign on the reverse strand, so with both strands pooled their pointwise covariance with the "
           "label is zero and PLS gives them zero weight.",
           pd.DataFrame([dict(track=n, role="explicit" if n in explicit else "PLS" if n in block["channels"]
                                                else "not used") for n in NAMES14]))


def protocol(w):
    s = config.settings()
    c = Canvas(13.6, 3.6)
    y = 2.05
    steps = [
        ("data", f"Dataset\n{len(config.DATASETS)} benchmark units\n147-bp sequences\n+ 14 shape tracks", SEQ),
        ("split", f"Outer k-fold (k = 5 or 10)\nstratified, seed {config.SPLIT_SEED}\nRC-identical sequences\nstay in one fold", JOINT),
        ("roles", "Per fold\ntest = 1/k (never trained on)\ntune = 10 % of the rest\nfit = the other 90 %", JOINT),
        ("prep", "Fit rows only\nk-mer z-scores, shape\nz-scores + PLS, motif\nprototypes, statistics norm", SHAPE),
        ("base", f"Base run\nSAM (ρ {s['sam_rho']}) + AdamW\n≤ {s['epochs']} epochs; best EMA\ntune log loss → epoch e*", JOINT),
        ("refit", "Refit (reported)\nfit + tune rows\ne* epochs, base\nlearning rates replayed", OUT),
    ]
    x = 0.15
    for key, text, color in steps:
        c.box(key, x, y, 2.1, 1.35, text, color, size=7.4)
        x += 2.25
    for a, b in zip([k for k, *_ in steps][:-1], [k for k, *_ in steps][1:]):
        c.arrow(c.anchor(a, "right"), c.anchor(b, "left"))
    c.box("pred", 9.15, 0.2, 2.1, 1.35, "Test predictions\nmean of forward and\nRC probability\nthreshold 0.5", OUT, size=7.4)
    c.box("pool", 6.9, 0.2, 2.1, 1.35, "Pooled over k folds\none metric set per seed\n(acc., MCC, AUC, …)", JOINT, size=7.4)
    c.box("seeds", 4.65, 0.2, 2.1, 1.35, f"Seeds {config.SEEDS[0]}–{config.SEEDS[-1]}\nper-fold seed = seed + fold\n"
          "reported: mean ± SD\nover seeds", JOINT, size=7.4)
    c.arrow(c.anchor("refit", "bottom", 0.3), c.anchor("pred", "top", 0.7))
    c.arrow(c.anchor("pred", "left"), c.anchor("pool", "right"))
    c.arrow(c.anchor("pool", "left"), c.anchor("seeds", "right"))
    c.box("figs", 0.15, 0.2, 4.3, 1.35, "Written automatically\nper fold: learning curves, confusion\nper seed: ROC/PR, reliability, "
          "gaps, epochs\nover seeds: mean ± SD table and figures\n(PNG 600 dpi + PDF + CSV)", "white", size=7.4)
    c.arrow(c.anchor("seeds", "left"), c.anchor("figs", "right"))
    w.save(c.fig, "protocol", "Evaluation protocol of every MorphoNup result: outer k-fold cross-validation with a "
           "tuning split inside each training fold, base run for checkpoint selection, refit on fit + tune rows, "
           "pooled out-of-fold metrics per seed and mean ± SD over three seeds.",
           pd.DataFrame([dict(step=k, text=t.replace("\n", " ")) for k, t, _ in steps]))


def draw_all(out=None):
    style()
    w = Writer(out or config.WORKSPACE / "docs" / "figures")
    architecture(w)
    shape_preprocessing(w)
    protocol(w)
    return w.finish()
