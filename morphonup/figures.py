"""Figures of every run, drawn from the saved run files only (so a lost figure is redrawn without retraining).

Each figure is written as <name>.png (600 dpi) + <name>.pdf + <name>.csv (the plotted values), and listed with a
suggested caption in figure_catalog.csv of its folder. Panels are title-free for the manuscript.

  fold_<ff>/figures/          learning curves (base + refit), test-fold confusion matrices
  seed_<s>/pooled/<stage>/figures/   ROC + PR, confusion matrix, reliability, per-fold metrics, generalization gap
  seed_<s>/pooled/figures/    base vs refit (ROC/PR, per-fold accuracy), learning curves of all folds, best epochs
  <arm>/summary/              mean +/- SD over seeds (csv, md) + seed metrics, ROC/PR of all seeds
  <results>/benchmark/        all datasets x k: table, heatmap, k=5 vs k=10

Colors: categorical slots 1-3 of the reference palette (validated all-pairs, light surface): refit = blue,
base = orange, tune rows = aqua (aqua is < 3:1 contrast, so it is always paired with a legend and line style);
neutral inks for context series; single-hue blue ramp for counts.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from . import config  # noqa: E402
from .metrics import reliability, scores  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9993", "#e6e5e1"
RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
STAGE_COLOR = dict(base=ORANGE, refit=BLUE)
STAGE_LABEL = dict(base="Base (fit rows)", refit="Refit (fit + tune rows; reported)")
NAMES = dict(accuracy="Accuracy", balanced_accuracy="Balanced accuracy", sensitivity="Sensitivity",
             specificity="Specificity", precision="Precision", f1="F1", mcc="MCC", kappa="Cohen's kappa",
             roc_auc="ROC AUC", pr_auc="PR AUC", nll="Log loss", brier="Brier score", ece="ECE",
             correct="Correct test sequences", gap_train_minus_test="Train − test accuracy gap")
FOLD_METRICS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "roc_auc", "pr_auc"]
TABLE_METRICS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "kappa",
                 "roc_auc", "pr_auc", "nll", "brier", "ece"]
CLASS_NAMES = ["Linker", "Nucleosome"]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.7, "lines.linewidth": 1.5, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})


class Writer:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records = []

    def save(self, fig, name, caption, data):
        fig.savefig(self.root / f"{name}.png", dpi=600, bbox_inches="tight")
        fig.savefig(self.root / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        data.to_csv(self.root / f"{name}.csv", index=False)
        self.records.append(dict(figure=name, png=f"{name}.png", pdf=f"{name}.pdf", data=f"{name}.csv",
                                 suggested_caption=caption))

    def finish(self):
        if self.records:
            pd.DataFrame(self.records).to_csv(self.root / "figure_catalog.csv", index=False)
        return [self.root / r["png"] for r in self.records]


def _json(path):
    return json.loads(Path(path).read_text())


def _thin(x, y, limit=4000):
    """At most `limit` curve points (endpoints kept) so curves of 160k-sequence sets stay light."""
    if len(x) <= limit:
        return x, y
    keep = np.unique(np.linspace(0, len(x) - 1, limit).round().astype(int))
    return x[keep], y[keep]


def _stage_folds(seed_dir, stage):
    return sorted(seed_dir.glob(f"fold_*/{stage}/fold_summary.json"))


# ---------------------------------------------------------------- per fold

def fold_figures(fold_dir, epoch_cap=120):
    """Learning curves (base, and refit when present) and test-fold confusion matrices of one outer fold."""
    style()
    fold_dir = Path(fold_dir)
    base = fold_dir / "base"
    if not (base / "fold_summary.json").exists():
        return []
    w = Writer(fold_dir / "figures")
    fold = int(fold_dir.name.split("_")[1])
    h = pd.read_csv(base / "history.csv")
    h["epoch_number"] = h["epoch"] + 1
    best = _json(base / "fold_summary.json")["best_epoch"]
    refit = fold_dir / "refit"
    r = pd.read_csv(refit / "history.csv") if (refit / "fold_summary.json").exists() else None
    if r is not None:
        r["epoch_number"] = r["epoch"] + 1

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), layout="constrained")
    ax = axes[0]
    ax.plot(h.epoch_number, h.loss, color=ORANGE, label="Base training objective")
    ax.plot(h.epoch_number, h.val_bce, color=AQUA, lw=0.8, alpha=0.45)
    ax.plot(h.epoch_number, h.val_bce_ema, color=AQUA, ls="--", label="Tune log loss (EMA; thin: raw)")
    if r is not None:
        ax.plot(r.epoch_number, r.loss, color=BLUE, label="Refit training objective")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax = axes[1]
    ax.plot(h.epoch_number, h.accuracy, color=ORANGE, lw=0.9, label="Base training (dropout on)")
    ev = h.dropna(subset=["train_eval_accuracy"])
    ax.plot(ev.epoch_number, ev.train_eval_accuracy, "o", ms=3.2, mfc="white", mec=ORANGE, mew=0.9,
            label="Base training (eval mode)")
    ax.plot(h.epoch_number, h.val_accuracy, color=AQUA, ls="--", label="Tune")
    if r is not None:
        ax.plot(r.epoch_number, r.accuracy, color=BLUE, lw=0.9, label="Refit training (dropout on)")
    ax.set(xlabel="Epoch", ylabel="Accuracy")
    ax = axes[2]
    ax.plot(h.epoch_number, h.learning_rate_used, color=ORANGE, label="Base")
    if r is not None:
        ax.plot(r.epoch_number, h.learning_rate_used.to_numpy()[:len(r)], color=BLUE, ls=(0, (1.5, 1.5)), label="Refit (replayed)")
    ax.set(xlabel="Epoch", ylabel="Learning rate", yscale="log")
    for ax in axes:
        ax.axvline(best, color=INK2, ls=":", lw=0.9)
        ax.grid(axis="y")
    axes[0].annotate(f"selected epoch {best}", (best, 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                     textcoords="offset points", fontsize=7, color=INK2, va="top")
    handles = [hd for ax in axes[:2] for hd in ax.get_legend_handles_labels()[0]]
    labels = [lb for ax in axes[:2] for lb in ax.get_legend_handles_labels()[1]]
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
    axes[2].legend(fontsize=7)
    data = h[["epoch_number", "loss", "accuracy", "train_eval_accuracy", "val_bce", "val_bce_ema", "val_accuracy",
              "learning_rate_used"]].assign(stage="base")
    if r is not None:
        data = pd.concat([data, r[["epoch_number", "loss", "accuracy"]].assign(
            stage="refit", learning_rate_used=h.learning_rate_used.to_numpy()[:len(r)])], ignore_index=True)
    cap = " The best epoch equals the epoch cap: the base run was still improving." if best >= epoch_cap else ""
    w.save(fig, "learning_curves",
           f"Fold {fold}: base run (fit rows, checkpoint chosen on tune rows) and refit (fit + tune rows, base epochs and "
           f"learning rates replayed). Dotted line: selected base epoch ({best}).{cap} The training objective includes "
           "label smoothing, RC-consistency and R-Drop terms, so it is not on the scale of the tune log loss; training "
           "accuracy with dropout on is below the eval-mode points.", data)

    stages = [s for s in ("base", "refit") if (fold_dir / s / "test_predictions.csv").exists()]
    fig, axes = plt.subplots(1, len(stages), figsize=(2.6 * len(stages), 2.4), layout="constrained", squeeze=False)
    rows = []
    for ax, stage in zip(axes[0], stages):
        p = pd.read_csv(fold_dir / stage / "test_predictions.csv")
        rows += _confusion(ax, p.label, p.probability, f"Predicted ({stage})", stage)
    w.save(fig, "confusion_matrix",
           f"Fold {fold}, outer test rows: confusion matrices at threshold 0.5 (counts; percentages within the observed class).",
           pd.DataFrame(rows))
    return w.finish()


def _confusion(ax, y, p, xlabel, stage):
    cm = confusion_matrix(np.asarray(y, int), np.asarray(p) >= 0.5, labels=[0, 1])
    rates = cm / cm.sum(1, keepdims=True)
    ax.imshow(rates, cmap=RAMP, vmin=0, vmax=1)
    rows = []
    for i in (0, 1):
        for j in (0, 1):
            ax.text(j, i, f"{cm[i, j]:,}\n{rates[i, j]:.1%}", ha="center", va="center", fontsize=8.5,
                    color="white" if rates[i, j] > 0.55 else INK)
            rows.append(dict(stage=stage, observed=CLASS_NAMES[i], predicted=CLASS_NAMES[j], count=int(cm[i, j]),
                             rate_within_observed=float(rates[i, j])))
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel=xlabel, ylabel="Observed")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return rows


# ---------------------------------------------------------------- per seed (pooled over the k test folds)

def stage_figures(seed_dir, stage):
    """Figures of one stage of one seed, pooled over its k outer test folds."""
    style()
    seed_dir = Path(seed_dir)
    target = seed_dir / "pooled" / stage
    if not (target / "metrics.json").exists():
        return []
    w = Writer(target / "figures")
    m = _json(target / "metrics.json")
    df = pd.read_csv(target / "test_predictions.csv")
    y, p = df.label.to_numpy(), df.probability.to_numpy()
    color, k = STAGE_COLOR[stage], m["k"]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), layout="constrained")
    curves = _roc_pr(axes, y, p, color, f"AUC {m['roc_auc']:.3f}", f"AP {m['pr_auc']:.3f}", stage)
    _chance(axes, y.mean())
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
    w.save(fig, "roc_pr", f"{STAGE_LABEL[stage]}: ROC and precision-recall curves of the pooled out-of-fold predictions "
           f"(k={k}, n={m['n']:,}). Dashed: chance (ROC) and nucleosome prevalence {y.mean():.3f} (PR).", curves)

    fig, ax = plt.subplots(figsize=(2.8, 2.5), layout="constrained")
    rows = _confusion(ax, y, p, "Predicted", stage)
    w.save(fig, "confusion_matrix", f"{STAGE_LABEL[stage]}: pooled confusion matrix over the {k} outer test folds at "
           f"threshold 0.5; accuracy {m['accuracy']:.4f} ({m['correct']:,}/{m['n']:,}), MCC {m['mcc']:.4f}.", pd.DataFrame(rows))

    rel = reliability(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), layout="constrained")
    r = rel[rel["count"] > 0]
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.8)
    axes[0].plot(r.predicted, r.observed, "o-", color=color, ms=4, label=f"ECE {m['ece']:.3f}")
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed nucleosome fraction", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(loc="upper left")
    edges = np.linspace(0, 1, 21)
    hist = []
    for cls, c, ls in ((1, BLUE, "-"), (0, ORANGE, "--")):
        counts = np.histogram(p[y == cls], bins=edges)[0]
        axes[1].stairs(counts, edges, color=c, ls=ls, lw=1.4, label=CLASS_NAMES[cls])
        hist += [dict(panel="histogram", observed=CLASS_NAMES[cls], lower=a, upper=b, count=int(n))
                 for a, b, n in zip(edges[:-1], edges[1:], counts)]
    axes[1].axvline(0.5, color=INK2, ls=":", lw=0.9)
    axes[1].set(xlabel="Predicted nucleosome probability", ylabel="Sequences", xlim=(0, 1))
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, title="Observed class", title_fontsize=8)
    data = pd.concat([rel.assign(panel="reliability"), pd.DataFrame(hist)], ignore_index=True)
    w.save(fig, "reliability", f"{STAGE_LABEL[stage]}: reliability (10 equal-width bins, empty bins omitted) and "
           "predicted-probability distributions by observed class; dotted line: decision threshold 0.5.", data)

    folds = _fold_metrics(seed_dir, stage)
    folds.to_csv(target / "fold_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 2.6), layout="constrained")
    x = np.arange(len(FOLD_METRICS))
    jitter = np.linspace(-0.18, 0.18, len(folds)) if len(folds) > 1 else np.zeros(1)
    for i, metric in enumerate(FOLD_METRICS):
        ax.scatter(x[i] + jitter, folds[metric], s=16, color=color, alpha=0.8, linewidths=0, zorder=3,
                   label="Outer fold" if i == 0 else None)
        ax.plot([x[i] - 0.28, x[i] + 0.28], [m[metric]] * 2, color=INK, lw=1.6, zorder=4,
                label="Pooled out-of-fold" if i == 0 else None)
    ax.set_xticks(x, [NAMES[n] for n in FOLD_METRICS], rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.grid(axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    w.save(fig, "fold_metrics", f"{STAGE_LABEL[stage]}: metric of each outer test fold (dots) and of the pooled "
           f"out-of-fold predictions (bar), k={k}. Folds share training rows, so their spread is descriptive, "
           "not an independent-sample interval.", folds.assign(pooled=False).pipe(
               lambda d: pd.concat([d, pd.DataFrame([{**{n: m[n] for n in FOLD_METRICS}, "pooled": True}])], ignore_index=True)))

    summaries = pd.DataFrame([_json(f) for f in _stage_folds(seed_dir, stage)])
    fig, ax = plt.subplots(figsize=(max(3.4, 0.42 * k + 1.6), 2.6), layout="constrained")
    for col, label, mk, c, fill in (("train_accuracy", "Training rows", "s", INK2, "white"),
                                    ("tune_accuracy", "Tune rows" + (" (in training)" if stage == "refit" else ""), "^", MUTED, MUTED),
                                    ("test_accuracy", "Outer test rows", "o", color, color)):
        ax.plot(summaries.fold, summaries[col], mk, ms=5.5, mfc=fill, mec=c, mew=1.1, ls="none", label=label)
    ax.set(xlabel="Outer fold", ylabel="Accuracy (eval mode)", xticks=summaries.fold)
    ax.grid(axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    gap = summaries.gap_train_minus_test.mean() * 100
    w.save(fig, "generalization_gap", f"{STAGE_LABEL[stage]}: accuracy of the selected network on its training rows, "
           f"the tune rows and the outer test rows of each fold (dropout off). Mean training-minus-test gap {gap:.1f} pp.",
           summaries[["fold", "train_accuracy", "tune_accuracy", "test_accuracy", "gap_train_minus_test"]])
    return w.finish()


def _fold_metrics(seed_dir, stage):
    rows = []
    for f in _stage_folds(seed_dir, stage):
        d = pd.read_csv(f.parent / "test_predictions.csv")
        s = scores(d.label, d.probability)
        rows.append(dict(fold=int(d.fold.iloc[0]), n=s["n"], correct=s["tp"] + s["tn"], **{n: s[n] for n in TABLE_METRICS}))
    return pd.DataFrame(rows)


def _roc_pr(axes, y, p, color, roc_label, pr_label, stage, **line):
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    fpr, tpr = _thin(fpr, tpr)
    recall, precision = _thin(recall[::-1], precision[::-1])
    axes[0].plot(fpr, tpr, color=color, label=roc_label, **line)
    axes[1].plot(recall, precision, color=color, label=pr_label, **line)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    return pd.concat([pd.DataFrame(dict(stage=stage, curve="ROC", x=fpr, y=tpr)),
                      pd.DataFrame(dict(stage=stage, curve="PR", x=recall, y=precision))], ignore_index=True)


def _chance(axes, prevalence):
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.8)
    axes[1].axhline(prevalence, color=MUTED, ls="--", lw=0.8)


def seed_figures(seed_dir, epoch_cap=120):
    """Per-stage pooled figures plus base-vs-refit and all-fold training figures of one seed."""
    style()
    seed_dir = Path(seed_dir)
    made = [png for stage in ("base", "refit") for png in stage_figures(seed_dir, stage)]
    stages = [s for s in ("base", "refit") if (seed_dir / "pooled" / s / "metrics.json").exists()]
    if not stages:
        return made
    w = Writer(seed_dir / "pooled" / "figures")
    metrics = {s: _json(seed_dir / "pooled" / s / "metrics.json") for s in stages}
    k = metrics[stages[0]]["k"]

    if len(stages) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), layout="constrained")
        parts = []
        for s in stages:
            d = pd.read_csv(seed_dir / "pooled" / s / "test_predictions.csv")
            m = metrics[s]
            parts.append(_roc_pr(axes, d.label.to_numpy(), d.probability.to_numpy(), STAGE_COLOR[s],
                                 f"{s.capitalize()}: AUC {m['roc_auc']:.3f}", f"{s.capitalize()}: AP {m['pr_auc']:.3f}", s,
                                 ls="-" if s == "refit" else "--"))
        _chance(axes, d.label.mean())
        axes[0].legend(loc="lower right")
        axes[1].legend(loc="lower left")
        w.save(fig, "roc_pr_base_vs_refit", f"Pooled out-of-fold ROC and precision-recall curves of the base run and the "
               f"refit (reported) of each fold, k={k}, same test rows.", pd.concat(parts, ignore_index=True))

        fb, fr = (_fold_metrics(seed_dir, s).set_index("fold") for s in ("base", "refit"))
        fig, ax = plt.subplots(figsize=(3.0, 2.8), layout="constrained")
        for fold in fb.index:
            ax.plot([0, 1], [fb.accuracy[fold], fr.accuracy[fold]], color=MUTED, lw=0.8, zorder=1)
        ax.scatter(np.zeros(len(fb)), fb.accuracy, s=22, color=ORANGE, zorder=3, label="Outer fold")
        ax.scatter(np.ones(len(fr)), fr.accuracy, s=22, color=BLUE, zorder=3)
        for x, s in ((0, "base"), (1, "refit")):
            ax.plot([x - 0.18, x + 0.18], [metrics[s]["accuracy"]] * 2, color=INK, lw=1.6, zorder=4,
                    label="Pooled" if s == "refit" else None)
            ax.annotate(f"{metrics[s]['accuracy']:.4f}", (x + 0.2, metrics[s]["accuracy"]), fontsize=7, color=INK2, va="center")
        ax.set(xticks=[0, 1], xticklabels=["Base", "Refit"], xlim=(-0.45, 1.6), ylabel="Test accuracy")
        ax.grid(axis="y")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
        diff = int(metrics["refit"]["correct"] - metrics["base"]["correct"])
        w.save(fig, "fold_accuracy_base_vs_refit", f"Test accuracy of each outer fold (lines pair the same fold) for the "
               f"base run and its refit, k={k}; bars: pooled. Refit minus base: {diff:+d} correct sequences.",
               fb[["accuracy"]].rename(columns={"accuracy": "base"}).join(fr[["accuracy"]].rename(columns={"accuracy": "refit"})).reset_index())

    histories, best = [], {}
    for f in _stage_folds(seed_dir, "base"):
        h = pd.read_csv(f.parent / "history.csv")
        fold = int(f.parent.parent.name.split("_")[1])
        best[fold] = _json(f)["best_epoch"]
        histories.append(h.assign(fold=fold, epoch_number=h.epoch + 1))
    if histories:
        hist = pd.concat(histories, ignore_index=True)
        cols = min(5, k)
        rows_n = int(np.ceil(k / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(1.55 * cols + 0.6, 1.5 * rows_n + 0.6), sharex=True, sharey=True,
                                 layout="constrained", squeeze=False)
        for ax, fold in zip(axes.ravel(), sorted(best)):
            h = hist[hist.fold == fold]
            ax.plot(h.epoch_number, h.loss, color=ORANGE, lw=1.0)
            ax.plot(h.epoch_number, h.val_bce_ema, color=AQUA, lw=1.0, ls="--")
            sel = h[h.epoch_number == best[fold]]
            ax.plot(sel.epoch_number, sel.val_bce_ema, "o", color=INK, ms=3.5)
            ax.text(0.97, 0.95, f"fold {fold}\nepoch {best[fold]}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
            ax.grid(axis="y")
        for ax in axes.ravel()[len(best):]:
            ax.axis("off")
        fig.supxlabel("Epoch", fontsize=9)
        fig.supylabel("Loss", fontsize=9)
        fig.legend([plt.Line2D([], [], color=ORANGE), plt.Line2D([], [], color=AQUA, ls="--"),
                    plt.Line2D([], [], color=INK, marker="o", ls="none", ms=3.5)],
                   ["Base training objective", "Tune log loss (EMA)", "Selected checkpoint"], loc="outside upper center", ncol=3, fontsize=7)
        w.save(fig, "learning_curves_all_folds", f"Base run of every outer fold (k={k}): training objective, EMA of the "
               "tune log loss and the selected checkpoint (lowest EMA tune loss after epoch 8). The refit replays each "
               "fold's selected epoch count.", hist[["fold", "epoch_number", "loss", "val_bce_ema", "val_accuracy"]])

        fig, ax = plt.subplots(figsize=(max(3.0, 0.42 * k + 1.4), 2.4), layout="constrained")
        folds = sorted(best)
        ax.bar(folds, [best[f] for f in folds], width=0.6, color=ORANGE, edgecolor="white", linewidth=1)
        ax.axhline(epoch_cap, color=INK2, ls="--", lw=0.9)
        ax.annotate(f"epoch cap {epoch_cap}", (1, epoch_cap), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=7, color=INK2)
        ax.set(xlabel="Outer fold", ylabel="Selected epoch", xticks=folds, ylim=(0, epoch_cap * 1.12))
        ax.grid(axis="y")
        hits = sum(best[f] >= epoch_cap for f in folds)
        w.save(fig, "best_epochs", f"Selected base epoch per outer fold (k={k}); {hits} of {k} folds reached the "
               f"{epoch_cap}-epoch cap (a fold at the cap was still improving when training stopped).",
               pd.DataFrame(dict(fold=folds, best_epoch=[best[f] for f in folds], epoch_cap=epoch_cap)))
    return made + w.finish()


# ---------------------------------------------------------------- per dataset x k x arm, over seeds

def summary_figures(arm_dir):
    """Mean +/- SD over the reported seeds (config.SEEDS) of the pooled metrics of each stage; seed strip plot;
    ROC/PR of those seeds. Other seed folders (config.UNREPORTED_SEEDS) are left out."""
    style()
    arm_dir = Path(arm_dir)
    rows = []
    for f in sorted(arm_dir.glob("seed_*/pooled/*/metrics.json")):
        if int(f.parents[2].name.split("_")[1]) not in config.SEEDS:
            continue
        m = _json(f)
        rows.append(dict(seed=int(f.parents[2].name.split("_")[1]), stage=f.parent.name, k=m["k"], n=m["n"],
                         correct=m["correct"], epoch_cap_hits=m.get("epoch_cap_hits", np.nan),
                         gap_train_minus_test=m["mean_gap_train_minus_test"], **{n: m[n] for n in TABLE_METRICS}))
    if not rows:
        return []
    out = arm_dir / "summary"
    w = Writer(out / "figures")
    seeds = pd.DataFrame(rows).sort_values(["stage", "seed"])
    seeds.to_csv(out / "seed_metrics.csv", index=False)
    agg = []
    for stage, d in seeds.groupby("stage"):
        for n in ["correct"] + TABLE_METRICS + ["gap_train_minus_test"]:
            v = d[n].astype(float)
            agg.append(dict(stage=stage, metric=n, n_seeds=len(d), mean=v.mean(), sd=v.std(ddof=1) if len(d) > 1 else np.nan,
                            min=v.min(), max=v.max(), best_seed=int(d.seed[v.idxmax()]) if n not in ("nll", "brier", "ece", "gap_train_minus_test") else int(d.seed[v.idxmin()])))
    agg = pd.DataFrame(agg)
    agg.to_csv(out / "summary.csv", index=False)
    run_config = arm_dir / "run_config.json"
    dataset = _json(run_config)["dataset_id"] if run_config.exists() else arm_dir.parent.parent.name
    lines = [f"# {dataset}, {arm_dir.parent.name}, arm `{arm_dir.name}`: pooled out-of-fold metrics over seeds", ""]
    for stage, d in agg.groupby("stage"):
        n_seeds = int(d.n_seeds.iloc[0])
        lines += [f"## {STAGE_LABEL[stage]} — {n_seeds} seed(s)", "", "| Metric | Mean ± SD | Min | Max | Best seed |", "|---|---|---|---|---|"]
        for _, r in d.iterrows():
            f = ".1f" if r.metric == "correct" else ".4f"
            sd = "" if np.isnan(r["sd"]) else f" ± {r['sd']:{f}}"
            lines.append(f"| {NAMES.get(r.metric, r.metric)} | {r['mean']:{f}}{sd} | {r['min']:{f}} | {r['max']:{f}} | {r.best_seed} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))

    fig, ax = plt.subplots(figsize=(6.4, 2.7), layout="constrained")
    x = np.arange(len(FOLD_METRICS))
    stages = [s for s in ("base", "refit") if s in set(seeds.stage)]
    offset = dict(base=-0.17, refit=0.17) if len(stages) == 2 else {stages[0]: 0.0}
    for stage in stages:
        d = seeds[seeds.stage == stage]
        for i, metric in enumerate(FOLD_METRICS):
            jit = np.linspace(-0.06, 0.06, len(d)) if len(d) > 1 else np.zeros(1)
            ax.scatter(x[i] + offset[stage] + jit, d[metric], s=14, color=STAGE_COLOR[stage], linewidths=0, zorder=3,
                       label=f"{stage.capitalize()} (one dot per seed)" if i == 0 else None)
            ax.plot([x[i] + offset[stage] - 0.12, x[i] + offset[stage] + 0.12], [d[metric].mean()] * 2, color=INK, lw=1.4, zorder=4,
                    label="Mean over seeds" if i == 0 and stage == stages[-1] else None)
    ax.set_xticks(x, [NAMES[n] for n in FOLD_METRICS], rotation=25, ha="right")
    ax.set_ylabel("Pooled out-of-fold score")
    ax.grid(axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    k = int(seeds.k.iloc[0])
    w.save(fig, "seed_metrics", f"Pooled out-of-fold metrics of each seed (dots) and their mean (bar), k={k}; "
           f"seeds {', '.join(map(str, sorted(seeds.seed.unique())))}.", seeds)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), layout="constrained")
    parts = []
    for stage in stages:
        d = seeds[seeds.stage == stage]
        for i, seed in enumerate(sorted(d.seed)):
            p = pd.read_csv(arm_dir / f"seed_{seed}" / "pooled" / stage / "test_predictions.csv")
            auc, ap = d[d.seed == seed].roc_auc.iloc[0], d[d.seed == seed].pr_auc.iloc[0]
            parts.append(_roc_pr(axes, p.label.to_numpy(), p.probability.to_numpy(), STAGE_COLOR[stage],
                                 None, None, stage, lw=0.9, alpha=0.75, ls="-" if stage == "refit" else "--").assign(seed=seed))
        lo, hi = d.roc_auc.min(), d.roc_auc.max()
        axes[0].plot([], [], color=STAGE_COLOR[stage], ls="-" if stage == "refit" else "--",
                     label=f"{stage.capitalize()}, {len(d)} seed(s): AUC {lo:.3f}–{hi:.3f}" if len(d) > 1 else f"{stage.capitalize()}: AUC {lo:.3f}")
        lo, hi = d.pr_auc.min(), d.pr_auc.max()
        axes[1].plot([], [], color=STAGE_COLOR[stage], ls="-" if stage == "refit" else "--",
                     label=f"{stage.capitalize()}: AP {lo:.3f}–{hi:.3f}" if len(d) > 1 else f"{stage.capitalize()}: AP {lo:.3f}")
    _chance(axes, p.label.mean())
    axes[0].legend(loc="lower right", fontsize=7)
    axes[1].legend(loc="lower left", fontsize=7)
    w.save(fig, "roc_pr_seeds", f"Pooled out-of-fold ROC and precision-recall curves, one line per seed, k={k}.",
           pd.concat(parts, ignore_index=True))
    return w.finish()


# ---------------------------------------------------------------- all datasets

def benchmark_figures(results_root, arm="full", stage="refit"):
    """Cross-dataset table (mean +/- SD over seeds) and figures for every finished dataset x k."""
    style()
    results_root = Path(results_root)
    rows = []
    for f in sorted(results_root.glob(f"**/k*/{arm}/summary/summary.csv")):
        arm_dir = f.parent.parent
        dataset = arm_dir.parent.parent.relative_to(results_root).as_posix()
        k = int(arm_dir.parent.name[1:])
        s = pd.read_csv(f)
        s = s[s.stage == stage].set_index("metric")
        if s.empty:
            continue
        cfg = _json(arm_dir / "run_config.json")
        rows.append(dict(dataset=dataset, k=k, n=cfg["data_manifest"]["n"],
                         nucleosome_fraction=cfg["data_manifest"]["positive"] / cfg["data_manifest"]["n"],
                         n_seeds=int(s.n_seeds.iloc[0]),
                         **{f"{n}_mean": s.loc[n, "mean"] for n in TABLE_METRICS},
                         **{f"{n}_sd": s.loc[n, "sd"] for n in TABLE_METRICS}))
    if not rows:
        return []
    out = results_root / "benchmark"
    w = Writer(out / "figures")
    table = pd.DataFrame(rows).sort_values(["n", "dataset", "k"])
    table.to_csv(out / "benchmark_table.csv", index=False)
    lines = [f"# MorphoNup benchmark ({stage}, arm `{arm}`): mean ± SD over seeds", "",
             "| Dataset | k | N | Nuc. fraction | Seeds | Accuracy | Balanced acc. | MCC | ROC AUC | PR AUC |", "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda r, n: f"{r[n + '_mean']:.4f}" + ("" if np.isnan(r[n + "_sd"]) else f" ± {r[n + '_sd']:.4f}")
    for _, r in table.iterrows():
        lines.append(f"| {r.dataset} | {r.k} | {r.n:,} | {r.nucleosome_fraction:.2f} | {r.n_seeds} | {fmt(r, 'accuracy')} | "
                     f"{fmt(r, 'balanced_accuracy')} | {fmt(r, 'mcc')} | {fmt(r, 'roc_auc')} | {fmt(r, 'pr_auc')} |")
    (out / "benchmark_table.md").write_text("\n".join(lines) + "\n")

    shown = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "mcc", "roc_auc", "pr_auc"]
    for k in sorted(table.k.unique()):
        d = table[table.k == k]
        vals = d[[f"{n}_mean" for n in shown]].to_numpy()
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(d) + 1.3), layout="constrained")
        ax.imshow(vals, cmap=RAMP, vmin=max(0.0, np.floor(np.nanmin(vals) * 10) / 10), vmax=1, aspect="auto")
        lo = max(0.0, np.floor(np.nanmin(vals) * 10) / 10)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if (vals[i, j] - lo) / (1 - lo) > 0.6 else INK)
        ax.set_xticks(range(len(shown)), [NAMES[n] for n in shown], rotation=25, ha="right")
        ax.set_yticks(range(len(d)), [f"{r.dataset} (n={r.n:,})" for _, r in d.iterrows()])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        w.save(fig, f"heatmap_k{k}", f"MorphoNup ({stage}) on every dataset, outer {k}-fold CV: mean over seeds of the "
               "pooled out-of-fold metrics (threshold 0.5). Rows sorted by dataset size.", d)

    both = table.groupby("dataset").k.nunique()
    both = both[both >= 2].index
    if len(both):
        d = table[table.dataset.isin(both)]
        order = d.drop_duplicates("dataset").sort_values("n").dataset.tolist()
        ks = sorted(d.k.unique())
        colors = [BLUE, ORANGE, AQUA]
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 0.3 * len(order) + 1.4), sharey=True, layout="constrained")
        for ax, metric in zip(axes, ("accuracy", "mcc")):
            for i, ds in enumerate(order):
                v = d[d.dataset == ds].set_index("k")[f"{metric}_mean"]
                ax.plot(v.values, [i] * len(v), color=MUTED, lw=0.8, zorder=1)
            for k, c, mk in zip(ks, colors, ("o", "s", "^")):
                v = d[d.k == k].set_index("dataset").reindex(order)
                ax.errorbar(v[f"{metric}_mean"], range(len(order)), xerr=v[f"{metric}_sd"], fmt=mk, color=c, ms=5,
                            elinewidth=0.8, capsize=0, label=f"k = {k}", zorder=3)
            ax.set(xlabel=NAMES[metric])
            ax.grid(axis="x")
        axes[0].set_yticks(range(len(order)), order)
        axes[0].legend(loc="lower left", fontsize=7)
        w.save(fig, "k_comparison", f"MorphoNup ({stage}): pooled out-of-fold accuracy and MCC at each k (mean ± SD over "
               "seeds; no bar = one seed). Lines join the same dataset.", d)
    return w.finish()


def ensure_figures(arm_dir, epoch_cap=120):
    """Draw whatever figures of a dataset/k/arm folder are missing (fold, seed) and refresh the seed summary."""
    arm_dir = Path(arm_dir)
    made = []
    for seed_dir in sorted(arm_dir.glob("seed_*")):
        for fold_dir in sorted(seed_dir.glob("fold_*")):
            if not (fold_dir / "figures" / "figure_catalog.csv").exists():
                made += fold_figures(fold_dir, epoch_cap)
        if (seed_dir / "pooled" / "refit" / "metrics.json").exists() and \
                not (seed_dir / "pooled" / "figures" / "figure_catalog.csv").exists():
            made += seed_figures(seed_dir, epoch_cap)
    return made + summary_figures(arm_dir)


def catalog(folder):
    """All figure catalogs under a folder, with file paths, as one table."""
    folder = Path(folder)
    parts = [pd.read_csv(f).assign(folder=str(f.parent.relative_to(folder))) for f in sorted(folder.rglob("figure_catalog.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
