"""Analysis arms: MorphoNup with one named change, retrained with the same splits, seeds and recipe.

Each arm lists only what differs from MorphoNup (`full`):
  settings  -> overrides of config.SETTINGS (e.g. motif scales, head dropout, SAM, RC view)
  network   -> switches of network.NETWORK (components removed from the graph)
  shape     -> overrides of config.SHAPE (shape preprocessing)
Virtual arms need no training: `no_refit` is the base stage of `full`.
Comparisons pair every arm with `full` on the same dataset, k, seeds and outer test rows (morphonup.analysis).
"""
from copy import deepcopy

from . import config

ARMS = {
    "full": dict(group="model", part="Reference", label="MorphoNup", change="reference model"),
    # ---- architecture: one component removed or reduced
    "three_scales": dict(group="architecture", part="Motif module", label="w/o 21-bp motif scale",
                         change="motif scales 7/11/15 bp (no 21-bp scale)",
                         settings=dict(motifs=[[7, 64], [11, 48], [15, 32]])),
    "head_dropout_035": dict(group="architecture", part="Head", label="Head dropout 0.35",
                             change="head dropout 0.35 instead of 0.5", settings=dict(head_dropout=0.35)),
    "no_shape_attenuation": dict(group="architecture", part="Motif module", label="w/o shape attenuation",
                                 change="motifs are plain sequence PWMs (no shape-prototype term); the physical adapter stays",
                                 network=dict(shape_attenuation=False)),
    "no_physical_adapter": dict(group="architecture", part="Branches and corrections", label="w/o physical adapter",
                                change="no physical adapter correction; shape still enters the motifs",
                                network=dict(physical_adapter=False)),
    "no_motif_statistics": dict(group="architecture", part="Branches and corrections", label="w/o motif statistics",
                                change="no dimer/trimer statistics correction", network=dict(motif_statistics=False)),
    "no_kmer": dict(group="architecture", part="Branches and corrections", label="w/o k-mer branch", change="no 1-4-mer composition branch",
                    network=dict(kmers=False)),
    "global_pool": dict(group="architecture", part="Motif module", label="w/o dyad bins",
                        change="every DyadPool keeps mean + max only (no 7 dyad bins)", network=dict(dyad_bins=False)),
    "no_rc_view": dict(group="architecture", part="Training and inference", label="w/o RC view",
                       change="trained and evaluated on the forward strand only (no RC view, no RC-consistency term)",
                       settings=dict(rc_view=False)),
    "no_sam": dict(group="architecture", part="Training and inference", label="w/o SAM", change="plain AdamW steps (no sharpness-aware minimization)",
                   settings=dict(sam=False)),
    # ---- modality: which inputs the model sees
    "sequence_only": dict(group="modality", part="Input set", label="Sequence only",
                          change="one-hot motifs, k-mers and motif statistics; no shape anywhere",
                          network=dict(shape_attenuation=False, physical_adapter=False)),
    "shape_only": dict(group="modality", part="Input set", label="Shape only",
                       change="9 reduced shape channels only (shape-prototype motifs + physical adapter); no sequence routes",
                       network=dict(sequence_motifs=False, kmers=False, motif_statistics=False)),
    "shuffled_shape": dict(group="modality", part="Input set", label="Shuffled shape",
                           change="network and inputs unchanged, but every sequence receives the shape tracks of another, "
                                  "randomly chosen sequence (one fixed permutation): shape values without their information",
                           shape=dict(shuffle_seed=42)),
    "no_pls": dict(group="modality", part="Shape preprocessing", label="w/o PLS reduction",
                   change="the 7 RC-symmetric tracks enter z-scored as one group (7 channels) instead of 6 PLS components",
                   shape=dict(blocks=[dict(channels=config.SYM7, method="none", k=7)])),
    "raw_shape": dict(group="modality", part="Shape preprocessing", label="All 14 tracks, unreduced",
                      change="no shape preprocessing: all 14 DNAshape tracks z-scored (incl. the 4 RC-antisymmetric ones), "
                             "no PLS, 6 biophysical groups",
                      shape=dict(raw=True)),
    "no_ep": dict(group="modality", part="Shape preprocessing", label="w/o EP",
                  change="EP removed; MGW, Roll + PLS block kept", shape=dict(explicit=["MGW", "Roll"])),
}
VIRTUAL = {
    "no_refit": dict(group="architecture", part="Training and inference", label="w/o refit", change="base run reported (fit rows only, no refit)",
                     source=("full", "base")),
}
ARCHITECTURE = ["no_shape_attenuation", "three_scales", "global_pool", "no_kmer", "no_physical_adapter",
                "no_motif_statistics", "head_dropout_035", "no_rc_view", "no_sam", "no_refit"]     # in model order, by part
MODALITY = ["sequence_only", "shape_only", "shuffled_shape", "no_pls", "raw_shape", "no_ep"]
GROUPS = dict(architecture=ARCHITECTURE, modality=MODALITY)
PARTS = dict(architecture=["Motif module", "Branches and corrections", "Head", "Training and inference"],
             modality=["Input set", "Shape preprocessing"])          # figure and table sections, top to bottom


def describe(arm):
    return {**ARMS, **VIRTUAL}[arm]


def resolve(arm):
    """(training settings, shape spec) of a trainable arm."""
    if arm not in ARMS:
        raise KeyError(f"{arm!r} is not a trainable arm; choose one of: " + ", ".join(ARMS))
    spec = ARMS[arm]
    cfg = config.settings()
    cfg.update(deepcopy(spec.get("settings", {})))
    if spec.get("network"):
        cfg["network"] = dict(spec["network"])
    shape = deepcopy(config.SHAPE)
    shape.update(deepcopy(spec.get("shape", {})))
    return cfg, shape


def table():
    """All arms as rows (for the notebooks)."""
    import pandas as pd
    rows = [dict(arm=name, group=s["group"], part=s["part"], label=s["label"], change=s["change"],
                 needs_training=name not in VIRTUAL) for name, s in {**ARMS, **VIRTUAL}.items()]
    return pd.DataFrame(rows)
