"""MorphoNup configuration: paths, dataset catalog, frozen hyper-parameters.

These values define the final model (DatasetNup_1 Homo sapiens, outer 5-fold:
89.10 +/- 0.03 % over seeds 151000-153000). Do not change them for a run; an analysis arm
states its own change explicitly.
"""
import os
from copy import deepcopy
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
# Datasets: MORPHONUP_DATA if set, else ./data inside the repository (unzipped from dataset/MorphoNup_data.zip,
# README section 3), else ../data (the original lab workspace layout).
DATA_ROOT = Path(os.environ["MORPHONUP_DATA"]) if os.environ.get("MORPHONUP_DATA") else \
    WORKSPACE / "data" if (WORKSPACE / "data").is_dir() else WORKSPACE.parent / "data"
RESULTS_ROOT = Path(os.environ.get("MORPHONUP_RESULTS", WORKSPACE / "results")).resolve()   # override for tests only

# The 12 benchmark units: dataset ID (as returned by morphonup.data.dataset_identity) -> folder under DATA_ROOT
DATASETS = {
    "DatasetNup_1/Homo_sapiens": "DatasetNup_1/Homo_sapiens",
    "DatasetNup_1/Caenorhabditis_elegans": "DatasetNup_1/Caenorhabditis_elegans",
    "DatasetNup_1/Drosophila_melanogaster": "DatasetNup_1/Drosophila_melanogaster",
    "DatasetNup_3/Homo_sapiens_hg38": "DatasetNup_3",
    "DatasetNup_2/Drosophila_melanogaster/5U": "DatasetNup_2/Drosophila_melanogaster/5U",
    "DatasetNup_2/Homo_sapiens/5U": "DatasetNup_2/Homo_sapiens/5U",
    "DatasetNup_2/Saccharomyces_cerevisiae/PM": "DatasetNup_2/Saccharomyces_cerevisiae/PM",
    "DatasetNup_2/Saccharomyces_cerevisiae/WG": "DatasetNup_2/Saccharomyces_cerevisiae/WG",
    "DatasetNup_2/Drosophila_melanogaster/LC": "DatasetNup_2/Drosophila_melanogaster/LC",
    "DatasetNup_2/Drosophila_melanogaster/PM": "DatasetNup_2/Drosophila_melanogaster/PM",
    "DatasetNup_2/Homo_sapiens/PM": "DatasetNup_2/Homo_sapiens/PM",
    "DatasetNup_2/Homo_sapiens/LC": "DatasetNup_2/Homo_sapiens/LC",
}

SEEDS = [151000, 152000, 153000]                   # the reported seeds; per-fold seed = seed + fold
# The first runs (HS and Ce benchmark, architecture analysis, part of the modality analysis) also trained seeds
# 154000 and 155000. Since 2026-11-06 only SEEDS are reported: those runs stay on disk, untouched, but every
# table, figure and analysis over seeds (figures.summary_figures, analysis.seeds_of) reads SEEDS only.
UNREPORTED_SEEDS = [154000, 155000]
SPLIT_SEED = 42                                    # make_splits(ds, k, seed=42): the outer folds of every reported run

# Shape preprocessing, fitted on the fit rows of each fold (forward + RC pooled).
# Buckle, Shear, Shift, Tilt are RC-antisymmetric: zero pointwise PLS loadings, so they are not used.
SYM7 = ["Opening", "ProT", "Stagger", "Stretch", "HelT", "Rise", "Slide"]
SHAPE = dict(explicit=["MGW", "Roll", "EP"], blocks=[dict(channels=SYM7, method="pls", k=6)], dyad_sigma=35.0)

SETTINGS = dict(
    # network
    motifs=[[7, 64], [11, 48], [15, 32], [21, 24]], kmer_dim=340, head_dropout=0.5,
    adapter_settings=dict(adapter_dropout=0.25, max_logit_change=1.0, correction_penalty=0.02),
    # objective: BCE over 4 dropout/strand passes + RC Jensen-Shannon + R-Drop KL + correction penalties
    label_smoothing=0.02, rc_consistency_weight=0.1, rdrop_weight=2.0,
    # optimizer: SAM over all weights + AdamW (clipnorm 1)
    sam_rho=0.05, learning_rate=1e-3, weight_decay=1e-3, min_lr=1e-5, batch_size=64, prediction_batch_size=64,
    wd_exclude=["bias", "beta", "gamma", "logit", "physical_centers", "residual_scale", "pair_gate"],
    # schedule and checkpoint selection on the tune rows (EMA of tune NLL)
    epochs=120, warmup_epochs=3, select_start_epoch=8, min_delta=1e-4, ema_decay=0.8,
    patience=24, lr_patience=8, train_eval_every=5,
)


def settings():
    return deepcopy(SETTINGS)


def dataset_folder(dataset_id):
    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset_id!r}; choose one of: " + ", ".join(DATASETS))
    return DATA_ROOT / DATASETS[dataset_id]


def gpu_memory_mb(n_sequences):
    """Per-process GPU cap: 2,560 MB was enough for <= 7k sequences; the 114k-sequence run used 10,000 MB."""
    return 2560 if n_sequences <= 20000 else 10000
