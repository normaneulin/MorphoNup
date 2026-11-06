"""Paired comparison of analysis arms with MorphoNup (same dataset, k, seeds and outer test rows).

For every arm and every seed it has in common with `full` (refit predictions; `no_refit` = full's base stage):
  delta = MorphoNup minus arm (positive: the removed component helps), per seed and per outer fold;
  McNemar exact test on the discordant test sequences of each seed; paired t-test over seeds (accuracy, MCC, ROC AUC)
  with a 95 % t interval, Holm-adjusted over the arms of one analysis.
Effects are conditional on the rest of the model: arms are not an additive decomposition.

Output: <results>/<dataset>/k<K>/analysis/<architecture|modality>/
  tables   manuscript_table (main), metrics_table (every metric), comparison (every statistic) as .md/.csv/.tex;
           seed_metrics.csv, fold_deltas.csv, fold_metrics.csv, training_curves.csv
  figures  deltas (forest plot), boxplot_accuracy, accuracy, metrics_panel, delta_heatmap, discordance, fold_deltas,
           roc_pr_parts, training_curves, training_diagnostics, parameters (+ roc_pr_arms for modality)
"""
import fcntl
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd
from scipy import stats

from . import arms, config
from .figures import BLUE, INK, INK2, MUTED, GRID, Writer, _chance, _roc_pr, style

REFERENCE = "full"
# per analysis: what a positive Δ = MorphoNup − variant means, and what every variant changes (captions, table notes)
HELPS = dict(architecture="the removed or changed component helps",
             modality="what the variant lacks, scrambles or preprocesses differently helps")
CHANGES = dict(architecture="removes or changes one component", modality="changes the inputs or the shape preprocessing")
HIGHER = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "roc_auc", "pr_auc"]
TABLE = HIGHER + ["nll", "brier"]                                          # nll, brier: lower is better
DELTAS = [("accuracy", "Δ accuracy (pp)", 100), ("mcc", "Δ MCC", 1), ("roc_auc", "Δ ROC AUC", 1)]
SHORT = dict(accuracy="Acc.", balanced_accuracy="Bal. acc.", sensitivity="Sn", specificity="Sp", precision="Prec.",
             f1="F1", mcc="MCC", roc_auc="ROC AUC", pr_auc="PR AUC")
PANEL = [("accuracy", "Accuracy (%)", 100), ("sensitivity", "Sensitivity (%)", 100), ("specificity", "Specificity (%)", 100),
         ("mcc", "MCC", 1), ("roc_auc", "ROC AUC", 1), ("pr_auc", "PR AUC", 1)]
LINES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]        # categorical slots 1-5 (validated, adjacent)
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
RED = "#e34948"                                                         # diverging pole opposite BLUE
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#a82e2d", RED, "#f4b4b3", "#f0efec", "#9cc3ef", BLUE, "#1c5cab"])   # gray midpoint = no change
REF_FILL = "#cde2fb"


def pooled_dir(k_dir, arm, seed):
    source, stage = arms.VIRTUAL[arm]["source"] if arm in arms.VIRTUAL else (arm, "refit")
    return Path(k_dir) / source / f"seed_{seed}" / "pooled" / stage


def seeds_of(k_dir, arm):
    source = arms.VIRTUAL[arm]["source"][0] if arm in arms.VIRTUAL else arm
    seeds = [int(p.name.split("_")[1]) for p in (Path(k_dir) / source).glob("seed_*")]
    return sorted(s for s in seeds if s in config.SEEDS and (pooled_dir(k_dir, arm, s) / "metrics.json").exists())


def mcnemar(b, c):
    """Exact two-sided McNemar test on b, c discordant pairs."""
    return 1.0 if b + c == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))


def _load(k_dir, arm, seed):
    d = pooled_dir(k_dir, arm, seed)
    return json.loads((d / "metrics.json").read_text()), pd.read_csv(d / "test_predictions.csv").set_index("index").sort_index()


def holm(p):
    """Holm step-down adjusted p-values over one family of tests (NaN entries are not tests and stay NaN)."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if ok.size:
        order = ok[np.argsort(p[ok], kind="stable")]
        out[order] = np.minimum(1.0, np.maximum.accumulate((ok.size - np.arange(ok.size)) * p[order]))
    return out


def t_interval(v, level=0.95):
    """Mean and t confidence interval over seeds (NaN bounds with fewer than two seeds)."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), np.nan, np.nan
    h = stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def _paired_p(a, b):
    """Two-sided paired t-test p over seeds; NaN with < 2 seeds or identical differences."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(stats.ttest_rel(a, b).pvalue) if len(a) >= 2 and np.ptp(a - b) > 0 else np.nan


def _sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def compare(k_dir, arm_list, name):
    """Tables and figures comparing `arm_list` with MorphoNup on their common seeds."""
    k_dir = Path(k_dir)
    ref_seeds = seeds_of(k_dir, REFERENCE)
    seed_rows, fold_rows = [], []
    for arm in [REFERENCE] + list(arm_list):
        for seed in sorted(set(seeds_of(k_dir, arm)) & set(ref_seeds)):
            m, d = _load(k_dir, arm, seed)
            row = dict(arm=arm, label=arms.describe(arm)["label"], seed=seed, parameters=m["parameters"], n=m["n"],
                       correct=m["correct"], **{t: m[t] for t in TABLE})
            if arm != REFERENCE:
                rm, r = _load(k_dir, REFERENCE, seed)
                if not r.index.equals(d.index):
                    raise ValueError(f"{arm} seed {seed}: test rows differ from {REFERENCE}")
                ref_ok = (r.probability >= 0.5) == (r.label == 1)
                arm_ok = (d.probability >= 0.5) == (d.label == 1)
                b, c = int((ref_ok & ~arm_ok).sum()), int((~ref_ok & arm_ok).sum())
                row.update(delta_correct=rm["correct"] - m["correct"], ref_right_arm_wrong=b, ref_wrong_arm_right=c,
                           mcnemar_p=mcnemar(b, c), **{f"delta_{t}": rm[t] - m[t] for t in TABLE})
                for fold, idx in r.groupby("fold").groups.items():
                    fold_rows.append(dict(arm=arm, seed=seed, fold=int(fold),
                                          delta_accuracy=float(ref_ok[idx].mean() - arm_ok[idx].mean())))
            seed_rows.append(row)
    seeds = pd.DataFrame(seed_rows)
    if seeds.empty or set(seeds.arm) == {REFERENCE}:
        return []
    folds = pd.DataFrame(fold_rows)
    present = [a for a in arm_list if a in set(seeds.arm)]
    summary = []
    for arm in present:
        d = seeds[seeds.arm == arm]
        ref = seeds[(seeds.arm == REFERENCE) & seeds.seed.isin(d.seed)].set_index("seed").loc[d.seed]
        row = dict(arm=arm, part=arms.describe(arm)["part"], label=arms.describe(arm)["label"],
                   change=arms.describe(arm)["change"], n_seeds=len(d), parameters=int(d.parameters.iloc[0]),
                   parameters_full=int(ref.parameters.iloc[0]),
                   **{f"{t}_{s}": f(d[t]) for t in ("accuracy", "mcc", "roc_auc") for s, f in (("mean", np.mean), ("sd", _sd))},
                   delta_correct_mean=d.delta_correct.mean(), delta_accuracy_sd=_sd(d.delta_accuracy))
        for metric, _, _ in DELTAS:
            mean, low, high = t_interval(d[f"delta_{metric}"])
            row.update({f"delta_{metric}_mean": mean, f"delta_{metric}_ci_low": low, f"delta_{metric}_ci_high": high,
                        f"paired_t_p_{metric}": _paired_p(ref[metric], d[metric])})
        row.update(seeds_full_better=int((d.delta_correct > 0).sum()), seeds_equal=int((d.delta_correct == 0).sum()),
                   seeds_arm_better=int((d.delta_correct < 0).sum()), mcnemar_p_min=d.mcnemar_p.min(),
                   mcnemar_p_max=d.mcnemar_p.max(), discordant_full_right=int(d.ref_right_arm_wrong.sum()),
                   discordant_arm_right=int(d.ref_wrong_arm_right.sum()))
        summary.append(row)
    summary = pd.DataFrame(summary)
    for metric, _, _ in DELTAS:            # one family per metric: every arm of this analysis vs MorphoNup
        summary[f"holm_p_{metric}"] = holm(summary[f"paired_t_p_{metric}"])
    fold_info, curves = _fold_data(k_dir, [REFERENCE] + present, seeds)
    out = k_dir / "analysis" / name
    out.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(out / "seed_metrics.csv", index=False)
    folds.to_csv(out / "fold_deltas.csv", index=False)
    fold_info.to_csv(out / "fold_metrics.csv", index=False)
    curves.to_csv(out / "training_curves.csv", index=False)
    summary.to_csv(out / "comparison.csv", index=False)
    _tables(out, k_dir, name, seeds, summary, present)
    return _figures(out, k_dir, name, seeds, folds, summary, present, fold_info, curves)


def _fold_data(k_dir, arm_list, seeds):
    """Per outer fold of every arm and seed: test accuracy, selected epoch and train - test gap of the reported stage;
    and the base-stage tune log loss per epoch averaged over folds (training curves; not for virtual arms)."""
    info, histories = [], []
    for arm in arm_list:
        source, stage = arms.VIRTUAL[arm]["source"] if arm in arms.VIRTUAL else (arm, "refit")
        for seed in sorted(seeds.seed[seeds.arm == arm]):
            _, d = _load(k_dir, arm, seed)
            correct = ((d.probability >= 0.5) == (d.label == 1)).groupby(d.fold).mean()
            for fold, acc in correct.items():
                folder = k_dir / source / f"seed_{seed}" / f"fold_{int(fold):02d}"
                s = json.loads((folder / stage / "fold_summary.json").read_text()) \
                    if (folder / stage / "fold_summary.json").exists() else {}
                info.append(dict(arm=arm, seed=seed, fold=int(fold), accuracy=float(acc), best_epoch=s.get("best_epoch"),
                                 gap_train_minus_test=s.get("gap_train_minus_test")))
                if arm not in arms.VIRTUAL and (folder / "base" / "history.csv").exists():
                    h = pd.read_csv(folder / "base" / "history.csv", usecols=["epoch", "val_bce"])
                    histories.append(h.assign(arm=arm))
    curves = pd.DataFrame(columns=["arm", "epoch", "tune_log_loss", "folds"])
    if histories:
        h = pd.concat(histories, ignore_index=True)
        curves = (h.groupby(["arm", "epoch"]).val_bce.agg(["mean", "count"]).reset_index()
                  .rename(columns={"mean": "tune_log_loss", "count": "folds"}))
        curves["epoch"] += 1
    return pd.DataFrame(info), curves


# ----------------------------------------------------------------------------------------------------- tables

def _p(p):
    return "—" if np.isnan(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def _signed(v, digits):
    return f"{v:+.{digits}f}".replace("-", "−")


def _ci(r, metric, scale, digits):
    mean = _signed(scale * r[f"delta_{metric}_mean"], digits)
    if np.isnan(r[f"delta_{metric}_ci_low"]):
        return mean
    return (f"{mean} [{_signed(scale * r[f'delta_{metric}_ci_low'], digits)}, "
            f"{_signed(scale * r[f'delta_{metric}_ci_high'], digits)}]")


def _fmt(m, s, scale=1, digits=4):
    return f"{scale * m:.{digits}f}" + ("" if np.isnan(s) else f" ± {scale * s:.{digits}f}")


def _write_table(out, stem, table, note, caption):
    """table as .csv, .md (with the note below) and .tex (booktabs; caption + note)."""
    table.to_csv(out / f"{stem}.csv", index=False)
    head = "| " + " | ".join(table.columns) + " |"
    body = ["| " + " | ".join(map(str, v)) + " |" for v in table.to_numpy()]
    (out / f"{stem}.md").write_text("\n".join([head, "|" + "---|" * len(table.columns), *body, "", note]) + "\n")

    def tex(v):
        v = str(v)
        for a, b in (("\\", r"\textbackslash{}"), ("%", r"\%"), ("_", r"\_"), ("&", r"\&"), ("#", r"\#")):
            v = v.replace(a, b)
        for a, b in (("±", r"$\pm$"), ("Δ", r"$\Delta$"), ("−", r"$-$"), ("—", "--"), ("<", r"$<$"), ("×", r"$\times$")):
            v = v.replace(a, b)
        return v
    lines = [r"% requires \usepackage{booktabs}", r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
             rf"\caption{{{tex(caption)} {tex(note)}}}", rf"\label{{tab:{out.name}_{stem}}}",
             rf"\begin{{tabular}}{{ll{'r' * (len(table.columns) - 2)}}}", r"\toprule",
             " & ".join(tex(c) for c in table.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(tex(v) for v in row) + r" \\" for row in table.to_numpy()]
    (out / f"{stem}.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n")


def _tables(out, k_dir, name, seeds, summary, present):
    ref = seeds[seeds.arm == REFERENCE]
    dataset = json.loads((k_dir / REFERENCE / "run_config.json").read_text())["dataset_id"]
    n = len(ref)
    lines = [f"# {name.capitalize()} analysis: {dataset}, {k_dir.name}", "",
             f"MorphoNup (refit): accuracy {_fmt(ref.accuracy.mean(), _sd(ref.accuracy))}, {n} seed(s), "
             f"{int(ref.parameters.iloc[0]):,} parameters.", "",
             f"Δ = MorphoNup − arm on the same seeds and test rows (positive: {HELPS[name]}). "
             "95 % CI: t interval over seeds. p (t): paired t-test over seeds on accuracy; p (Holm): adjusted over the "
             f"{len(summary)} arm(s) of this analysis. McNemar: exact test per seed on discordant sequences.", "",
             "| Part | Arm | Change | Seeds | Accuracy | Δ accuracy (pp) [95 % CI] | Δ correct | Δ MCC | Δ ROC AUC | "
             "Seeds MorphoNup better / equal / worse | p (t) | p (Holm) | p MCC (Holm) | p ROC AUC (Holm) | "
             "McNemar p (range) | Parameters |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.part} | {r.label} | {r.change} | {r.n_seeds} | {_fmt(r.accuracy_mean, r.accuracy_sd)} | "
                     f"{_ci(r, 'accuracy', 100, 2)} | {_signed(r.delta_correct_mean, 1)} | {_signed(r.delta_mcc_mean, 4)} | "
                     f"{_signed(r.delta_roc_auc_mean, 4)} | {r.seeds_full_better} / {r.seeds_equal} / {r.seeds_arm_better} | "
                     f"{_p(r.paired_t_p_accuracy)} | {_p(r.holm_p_accuracy)} | {_p(r.holm_p_mcc)} | {_p(r.holm_p_roc_auc)} | "
                     f"{r.mcnemar_p_min:.3g}–{r.mcnemar_p_max:.3g} | {r.parameters:,} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    summary.to_csv(out / "comparison.csv", index=False)

    protocol = (f"Pooled out-of-fold metrics on {dataset} ({k_dir.name}), mean ± SD over seeds. Each variant "
                f"{CHANGES[name]} and is retrained with the same splits, seeds and recipe"
                + ("; w/o refit is the base stage of the MorphoNup runs (no extra training)." if "no_refit" in present else "."))
    rows = [dict(Part="Reference", Model="MorphoNup (all components)", Seeds=n, Parameters=f"{int(ref.parameters.iloc[0]):,}",
                 **{"Accuracy (%)": _fmt(ref.accuracy.mean(), _sd(ref.accuracy), 100, 2),
                    "MCC": _fmt(ref.mcc.mean(), _sd(ref.mcc)), "ROC AUC": _fmt(ref.roc_auc.mean(), _sd(ref.roc_auc)),
                    "Δ accuracy (pp) [95 % CI]": "—", "Δ correct": "—", "Δ MCC [95 % CI]": "—", "p (Holm)": "—"})]
    for _, r in summary.iterrows():
        rows.append(dict(Part=r.part, Model=r.label, Seeds=r.n_seeds, Parameters=f"{r.parameters:,}",
                         **{"Accuracy (%)": _fmt(r.accuracy_mean, r.accuracy_sd, 100, 2),
                            "MCC": _fmt(r.mcc_mean, r.mcc_sd), "ROC AUC": _fmt(r.roc_auc_mean, r.roc_auc_sd),
                            "Δ accuracy (pp) [95 % CI]": _ci(r, "accuracy", 100, 2),
                            "Δ correct": _signed(r.delta_correct_mean, 1), "Δ MCC [95 % CI]": _ci(r, "mcc", 1, 4),
                            "p (Holm)": _p(r.holm_p_accuracy)}))
    _write_table(out, "manuscript_table", pd.DataFrame(rows),
                 protocol + f" Δ = MorphoNup − variant on the same test sequences (positive: {HELPS[name]}); "
                 "95 % CI: t interval over seeds; p: paired t-test over seeds on accuracy, Holm-adjusted over the "
                 f"{len(summary)} variant(s).",
                 f"{name.capitalize()} analysis of MorphoNup.")

    rows = []
    for arm in [REFERENCE] + present:
        d = seeds[seeds.arm == arm]
        rows.append(dict(Part=arms.describe(arm)["part"], Model="MorphoNup (all components)" if arm == REFERENCE
                         else arms.describe(arm)["label"], Seeds=len(d),
                         **{f"{SHORT[t]} (%)": _fmt(d[t].mean(), _sd(d[t]), 100, 2) for t in HIGHER[:6]},
                         **{SHORT[t]: _fmt(d[t].mean(), _sd(d[t])) for t in HIGHER[6:]},
                         **{"Log loss": _fmt(d.nll.mean(), _sd(d.nll)), "Brier": _fmt(d.brier.mean(), _sd(d.brier))}))
    _write_table(out, "metrics_table", pd.DataFrame(rows),
                 protocol + " Acc.: accuracy; Bal. acc.: balanced accuracy; Sn: sensitivity (nucleosome recall); "
                 "Sp: specificity (linker recall); Prec.: precision; threshold 0.5.",
                 f"Every metric of each variant in the {name} analysis.")


# ---------------------------------------------------------------------------------------------------- figures

def _layout(arm_list, group, reference=False):
    """Rows from top to bottom: a header per part, then its arms -> [(y, arm or None, text)]."""
    sections = [("Reference", [REFERENCE])] if reference else []
    sections += [(p, [a for a in arm_list if arms.describe(a)["part"] == p]) for p in arms.PARTS[group]]
    rows, y = [], 0.0
    for part, members in sections:
        if not members:
            continue
        rows.append((y, None, part))
        y -= 0.85
        for arm in members:
            rows.append((y, arm, arms.describe(arm)["label"]))
            y -= 1.0
        y -= 0.35
    return rows


def _label_rows(ax, rows):
    ax.set_yticks([y for y, _, _ in rows], [t for _, _, t in rows])
    for tick, (_, arm, _) in zip(ax.get_yticklabels(), rows):
        if arm is None:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(rows[-1][0] - 0.7, rows[0][0] + 0.6)


def _figures(out, k_dir, name, seeds, folds, summary, present, fold_info, curves):
    style()
    w = Writer(out / "figures")
    ctx = dict(w=w, k_dir=k_dir, name=name, seeds=seeds, folds=folds, summary=summary, present=present,
               fold_info=fold_info, curves=curves, n_seeds=int((seeds.arm == REFERENCE).sum()))
    for draw in (_fig_deltas, _fig_boxplot, _fig_accuracy, _fig_metrics_panel, _fig_heatmap, _fig_discordance,
                 _fig_fold_deltas, _fig_roc_pr_parts, _fig_training_curves, _fig_training_diagnostics, _fig_parameters,
                 _fig_roc_pr_arms):
        try:                                   # one broken figure must not block the others (or the training job)
            draw(**ctx)
        except Exception as error:             # noqa: BLE001
            plt.close("all")
            print(f"FIGURE ERROR analysis/{name}/{draw.__name__[5:]}: {error!r}", flush=True)
    return w.finish()


def _fig_deltas(w, name, seeds, summary, present, n_seeds, **_):
    """Forest plot of the paired differences: the main figure."""
    rows = _layout(present, name)
    arm_rows = [(y, a) for y, a, _ in rows if a]
    s = summary.set_index("arm")
    fig, axes = plt.subplots(1, len(DELTAS) + 1, figsize=(7.4, 0.27 * len(rows) + 1.25), sharey=True, layout="constrained",
                             gridspec_kw=dict(width_ratios=[1] * len(DELTAS) + [0.95]))
    for ax, (metric, xlabel, scale) in zip(axes, DELTAS):
        for yi, arm in arm_rows:
            v = scale * seeds.loc[seeds.arm == arm, f"delta_{metric}"].to_numpy()
            jit = np.linspace(-0.2, 0.2, len(v)) if len(v) > 1 else np.zeros(1)
            ax.scatter(v, yi + jit, s=11, color=BLUE, alpha=0.8, linewidths=0, zorder=3)
            low, high = scale * s.at[arm, f"delta_{metric}_ci_low"], scale * s.at[arm, f"delta_{metric}_ci_high"]
            if np.isfinite(low):
                ax.plot([low, high], [yi, yi], color=INK, lw=1.3, solid_capstyle="butt", zorder=4)
            filled = s.at[arm, f"holm_p_{metric}"] < 0.05
            ax.scatter(scale * s.at[arm, f"delta_{metric}_mean"], yi, marker="D", s=24, facecolor=INK if filled else "white",
                       edgecolor=INK, linewidths=1.0, zorder=5)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="x")
    _label_rows(axes[0], rows)
    text = axes[-1]
    text.axis("off")
    tr = text.get_yaxis_transform()
    text.text(0.02, rows[0][0], "Δ correct", transform=tr, ha="left", va="center", fontsize=7, fontweight="bold", color=INK)
    text.text(0.98, rows[0][0], "p (Holm)", transform=tr, ha="right", va="center", fontsize=7, fontweight="bold", color=INK)
    for yi, arm in arm_rows:
        text.text(0.02, yi, _signed(s.at[arm, "delta_correct_mean"], 1), transform=tr, ha="left", va="center", fontsize=7,
                  color=INK2)
        text.text(0.98, yi, _p(s.at[arm, "holm_p_accuracy"]), transform=tr, ha="right", va="center", fontsize=7, color=INK2)
    handles = [Line2D([], [], marker="o", ls="", ms=3.5, color=BLUE, alpha=0.8, label="Seed"),
               Line2D([], [], marker="D", ms=4.5, color=INK, mfc=INK, lw=1.3, label="Mean, 95 % CI (Holm p < 0.05)"),
               Line2D([], [], marker="D", ms=4.5, color=INK, mfc="white", lw=1.3, label="Mean, 95 % CI (not significant)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=7)
    w.save(fig, "deltas", f"{name.capitalize()} analysis: MorphoNup minus each variant on the same seeds and outer test rows "
           f"(pooled out-of-fold, {n_seeds} seed(s)); right of 0 = {HELPS[name]}. Dots: seeds; "
           "diamond and bar: mean and 95 % t interval over seeds; filled: paired t-test over seeds, Holm-adjusted over the "
           "variants of this panel, p < 0.05. Right column: mean difference in correctly classified test sequences and "
           "the Holm-adjusted p for accuracy. Effects are conditional on the rest of the model, not an additive "
           "decomposition.",
           summary[["arm", "part", "label", "n_seeds", "delta_correct_mean"]
                   + [f"{p}{m}{q}" for m, _, _ in DELTAS for p, q in (("delta_", "_mean"), ("delta_", "_ci_low"),
                                                                     ("delta_", "_ci_high"), ("paired_t_p_", ""),
                                                                     ("holm_p_", ""))]])


def _fig_boxplot(w, name, present, fold_info, n_seeds, **_):
    """Classic ablation box plot: one box per model (full model first) over every outer test fold of every seed."""
    order = [REFERENCE] + [a for p in arms.PARTS[name] for a in present if arms.describe(a)["part"] == p]
    xs, x, prev = [], 0.0, None
    for arm in order:
        part = arms.describe(arm)["part"]
        x += 0.0 if prev is None else (1.0 if part == prev else 1.7)
        xs.append(x)
        prev = part
    fig, ax = plt.subplots(figsize=(0.46 * len(order) + 1.6, 3.7), layout="constrained")
    rng = np.random.default_rng(0)
    ref_mean = 100 * fold_info.loc[fold_info.arm == REFERENCE, "accuracy"].mean()
    ax.axhline(ref_mean, color=BLUE, lw=0.9, ls="--", zorder=1)
    for xi, arm in zip(xs, order):
        v = 100 * fold_info.loc[fold_info.arm == arm, "accuracy"].to_numpy()
        is_ref = arm == REFERENCE
        ax.boxplot(v, positions=[xi], widths=0.62, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=REF_FILL if is_ref else "white", edgecolor=BLUE if is_ref else INK2, linewidth=0.9),
                   medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=INK2, linewidth=0.9),
                   capprops=dict(color=INK2, linewidth=0.9), zorder=2)
        ax.scatter(xi + rng.uniform(-0.2, 0.2, len(v)), v, s=7, color=BLUE if is_ref else MUTED, alpha=0.75,
                   linewidths=0, zorder=3)
        ax.scatter(xi, v.mean(), marker="D", s=16, facecolor="white", edgecolor=INK, linewidths=0.9, zorder=4)
    labels = ["MorphoNup (full)" if a == REFERENCE else arms.describe(a)["label"] for a in order]
    ax.set_xticks(xs, labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for part in dict.fromkeys(arms.describe(a)["part"] for a in order):
        members = [xi for xi, a in zip(xs, order) if arms.describe(a)["part"] == part]
        ax.text(np.mean(members), 1.01, textwrap.fill(part, 14), transform=tr, ha="center", va="bottom", fontsize=7,
                fontweight="bold", color=INK2)
    for left, right in zip(xs[:-1], xs[1:]):
        if right - left > 1.0:
            ax.axvline((left + right) / 2, color=GRID, lw=0.8, ls=":", zorder=0)
    ax.set_ylabel("Accuracy per outer test fold (%)")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)
    w.save(fig, "boxplot_accuracy", f"{name.capitalize()} analysis: test accuracy of MorphoNup and of each variant on every "
           f"outer test fold ({n_seeds} seeds × folds; dots). Box: median and quartiles, whiskers 1.5 × IQR; diamond: mean; "
           "dashed line: MorphoNup mean. Fold-to-fold spread is larger than most differences, and folds of one seed share "
           "training rows, so the paired forest plot (deltas) carries the inference.",
           fold_info[fold_info.arm.isin(order)][["arm", "seed", "fold", "accuracy"]])


def _dot_rows(ax, order, values_of, ref_values, color_ref=BLUE):
    """Seed dots + mean ± SD per row of `order`, with the reference mean ± SD as a band."""
    ref_sd = _sd(ref_values) if len(ref_values) > 1 else 0.0
    ax.axvspan(np.mean(ref_values) - ref_sd, np.mean(ref_values) + ref_sd, color=BLUE, alpha=0.08, lw=0)
    ax.axvline(np.mean(ref_values), color=BLUE, lw=0.9, ls="--")
    for yi, arm, _ in order:
        if arm is None:
            continue
        v = values_of(arm)
        jit = np.linspace(-0.2, 0.2, len(v)) if len(v) > 1 else np.zeros(1)
        ax.scatter(v, yi + jit, s=11, color=color_ref if arm == REFERENCE else MUTED, linewidths=0, zorder=3)
        if len(v) > 1:
            ax.plot([v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1)], [yi, yi], color=INK, lw=1.0, zorder=4)
        ax.plot([v.mean()] * 2, [yi - 0.3, yi + 0.3], color=INK, lw=1.6, zorder=4)
    ax.grid(axis="x")


def _fig_accuracy(w, name, seeds, present, **_):
    order = _layout(present, name, reference=True)
    fig, ax = plt.subplots(figsize=(4.8, 0.27 * len(order) + 1.0), layout="constrained")
    _dot_rows(ax, order, lambda a: seeds.loc[seeds.arm == a, "accuracy"].to_numpy(),
              seeds.loc[seeds.arm == REFERENCE, "accuracy"].to_numpy())
    _label_rows(ax, order)
    ax.set_xlabel("Pooled out-of-fold accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    w.save(fig, "accuracy", f"{name.capitalize()} analysis: pooled out-of-fold accuracy of each variant (dots: seeds; tick: "
           "mean; bar: ± SD); dashed line and band: MorphoNup mean ± SD.", seeds[["arm", "seed", "accuracy", "correct", "n"]])


def _fig_metrics_panel(w, name, seeds, present, n_seeds, **_):
    """The usual multi-metric comparison (as a dot plot: bars on a truncated axis would exaggerate the differences)."""
    order = _layout(present, name, reference=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 2 * (0.19 * len(order) + 0.75)), sharey=True, layout="constrained")
    for ax, (metric, xlabel, scale) in zip(axes.ravel(), PANEL):
        _dot_rows(ax, order, lambda a: scale * seeds.loc[seeds.arm == a, metric].to_numpy(),
                  scale * seeds.loc[seeds.arm == REFERENCE, metric].to_numpy())
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    for row in axes:
        _label_rows(row[0], order)
    w.save(fig, "metrics_panel", f"{name.capitalize()} analysis: six pooled out-of-fold metrics of every variant ({n_seeds} "
           "seeds; dots: seeds; tick: mean; bar: ± SD); dashed line and band: MorphoNup mean ± SD. Sensitivity = nucleosome "
           "recall, specificity = linker recall (threshold 0.5).",
           seeds[["arm", "seed"] + [m for m, _, _ in PANEL]])


def _fig_heatmap(w, name, seeds, present, n_seeds, **_):
    """Δ of every metric for every variant (diverging: blue = MorphoNup better, red = variant better)."""
    values, bold = np.full((len(present), len(HIGHER)), np.nan), np.zeros((len(present), len(HIGHER)), bool)
    for i, arm in enumerate(present):
        for j, metric in enumerate(HIGHER):
            mean, low, high = t_interval(100 * seeds.loc[seeds.arm == arm, f"delta_{metric}"].to_numpy())
            values[i, j] = mean
            bold[i, j] = np.isfinite(low) and (low > 0 or high < 0)
    vmax = max(0.5, np.ceil(np.nanmax(np.abs(values)) * 2) / 2)
    fig, ax = plt.subplots(figsize=(0.62 * len(HIGHER) + 3.0, 0.33 * len(present) + 1.7), layout="constrained")
    im = ax.imshow(values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(present)):
        for j in range(len(HIGHER)):
            ax.text(j, i, _signed(values[i, j], 2), ha="center", va="center", fontsize=6.5,
                    fontweight="bold" if bold[i, j] else "normal",
                    color="white" if abs(values[i, j]) > 0.6 * vmax else INK)
    ax.set_xticks(range(len(HIGHER)), [SHORT[m] for m in HIGHER], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(present)), [arms.describe(a)["label"] for a in present])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    parts = [arms.describe(a)["part"] for a in present]
    for i in range(1, len(present)):
        if parts[i] != parts[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=3)
    for part in dict.fromkeys(parts):
        idx = [i for i, p in enumerate(parts) if p == part]
        ax.annotate("", xy=(len(HIGHER) - 0.35, idx[0] - 0.35), xytext=(len(HIGHER) - 0.35, idx[-1] + 0.35),
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), annotation_clip=False)
        ax.text(len(HIGHER) - 0.2, np.mean(idx), textwrap.fill(part, 14), ha="left", va="center", fontsize=7,
                fontweight="bold", color=INK2)
    bar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, aspect=40)
    bar.set_label("Δ = MorphoNup − variant (percentage points; MCC and AUC × 100)", fontsize=7)
    bar.ax.tick_params(labelsize=7)
    data = pd.DataFrame(values, index=present, columns=HIGHER).reset_index(names="arm")
    w.save(fig, "delta_heatmap", f"{name.capitalize()} analysis: mean difference MorphoNup − variant for nine metrics "
           f"({n_seeds} seeds; × 100). Blue: the variant is worse ({HELPS[name]}); red: the variant is better; "
           "bold: the 95 % t interval over seeds excludes 0.", data)


def _fig_discordance(w, name, seeds, summary, present, n_seeds, **_):
    """Paired view behind McNemar: test sequences only one of the two models classifies correctly."""
    rows = _layout(present, name)
    fig, ax = plt.subplots(figsize=(5.4, 0.27 * len(rows) + 1.2), layout="constrained")
    top = 1.0
    for yi, arm, _ in rows:
        if arm is None:
            continue
        d = seeds[seeds.arm == arm]
        b, c = d.ref_right_arm_wrong.mean(), d.ref_wrong_arm_right.mean()
        ax.barh(yi, b, height=0.6, color=BLUE, edgecolor="white", linewidth=1)
        ax.barh(yi, -c, height=0.6, color=RED, edgecolor="white", linewidth=1)
        ax.annotate(f"net {_signed(b - c, 1)}", (b, yi), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK2)
        top = max(top, b, c)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-top * 1.15, top * 1.45)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.set_xlabel(f"Test sequences per seed (mean of {n_seeds})")
    ax.grid(axis="x")
    _label_rows(ax, rows)
    fig.legend(handles=[Patch(color=BLUE, label="MorphoNup right, variant wrong"),
                        Patch(color=RED, label="Variant right, MorphoNup wrong")],
               loc="outside upper center", ncol=2, fontsize=7)
    w.save(fig, "discordance", f"{name.capitalize()} analysis: the discordant test sequences behind the McNemar test, mean "
           f"per seed ({n_seeds} seeds). Blue: classified correctly by MorphoNup only; red: by the variant only; net = "
           "blue − red = difference in correct sequences.",
           summary[["arm", "label", "discordant_full_right", "discordant_arm_right", "n_seeds"]])


def _fig_fold_deltas(w, name, folds, present, **_):
    if folds.empty:
        return
    rows = _layout(present, name)
    fig, ax = plt.subplots(figsize=(4.8, 0.27 * len(rows) + 1.0), layout="constrained")
    rng = np.random.default_rng(0)
    for yi, arm, _ in rows:
        if arm is None:
            continue
        v = 100 * folds.loc[folds.arm == arm, "delta_accuracy"].to_numpy()
        ax.boxplot(v, positions=[yi], orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="white", edgecolor=INK2, linewidth=0.9),
                   medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=INK2, linewidth=0.9),
                   capprops=dict(color=INK2, linewidth=0.9), zorder=2)
        ax.scatter(v, yi + rng.uniform(-0.22, 0.22, len(v)), s=7, color=BLUE, alpha=0.7, linewidths=0, zorder=3)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    _label_rows(ax, rows)
    ax.set_xlabel("Δ accuracy per outer test fold (pp)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="x")
    w.save(fig, "fold_deltas", f"{name.capitalize()} analysis: MorphoNup minus variant accuracy on each outer test fold "
           "of each seed (dots; box: median and quartiles, whiskers 1.5 × IQR). Folds of one seed share training rows "
           "and fold-to-fold noise is large, so this panel is descriptive; inference uses the seed-level test.", folds)


def _fig_roc_pr_parts(w, k_dir, name, seeds, present, **_):
    """ROC and precision-recall curves, one column per part: MorphoNup and the variants of that part (one seed)."""
    parts = [p for p in arms.PARTS[name] if any(arms.describe(a)["part"] == p for a in present)]
    groups = [[a for a in present if arms.describe(a)["part"] == p] for p in parts]
    common = sorted(set.intersection(*[set(seeds.seed[seeds.arm == a]) for a in [REFERENCE] + present]))
    if not common or max(len(g) for g in groups) + 1 > len(LINES):
        return
    seed = common[0]
    fig, axes = plt.subplots(2, len(parts), figsize=(min(7.4, 1.95 * len(parts) + 0.7), 5.0), layout="constrained",
                             squeeze=False)
    data = []
    for col, (part, members) in enumerate(zip(parts, groups)):
        pair = axes[:, col]
        for arm, color, ls in zip([REFERENCE] + members, LINES, STYLES):
            _, d = _load(k_dir, arm, seed)
            label = "MorphoNup" if arm == REFERENCE else arms.describe(arm)["label"]
            auc = seeds.loc[seeds.arm == arm, "roc_auc"].mean()
            ap = seeds.loc[seeds.arm == arm, "pr_auc"].mean()
            data.append(_roc_pr(pair, d.label.to_numpy(), d.probability.to_numpy(), color, f"{label} ({auc:.4f} / {ap:.4f})",
                                "_nolegend_", arm, ls=ls, lw=1.1).assign(arm=arm, part=part))
        _chance(pair, d.label.mean())
        pair[0].set(xlim=(0, 0.5), ylim=(0.5, 1.0))          # zoom: at full scale the curves coincide
        pair[1].set(xlim=(0.5, 1.0), ylim=(0.75, 1.0))
        pair[0].set_title(part, fontsize=8, loc="left", color=INK)
        pair[1].legend(*pair[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.3),
                       fontsize=5.5, handlelength=2.2)
        if col:
            pair[0].set_ylabel("")
            pair[1].set_ylabel("")
    w.save(fig, "roc_pr_parts", f"{name.capitalize()} analysis: pooled out-of-fold ROC (top) and precision–recall (bottom) "
           f"curves of MorphoNup and the variants of each part, seed {seed}, axes zoomed to the upper-left (ROC) and "
           "upper-right (PR) region where the curves separate; legend: ROC AUC / average precision, mean over seeds. "
           "Differences of a few thousandths of AUC are hard to see as curves; the forest plot and tables give them.",
           pd.concat(data, ignore_index=True))


def _fig_training_curves(w, name, curves, present, **_):
    """Tune log loss per epoch (base runs), mean over folds and seeds, MorphoNup vs the variants of each part."""
    trained = [a for a in present if a not in arms.VIRTUAL and a in set(curves.arm)]
    parts = [p for p in arms.PARTS[name] if any(arms.describe(a)["part"] == p for a in trained)]
    if REFERENCE not in set(curves.arm) or not parts:
        return
    start = config.SETTINGS["select_start_epoch"]
    fig, axes = plt.subplots(1, len(parts), figsize=(min(7.4, 1.95 * len(parts) + 0.7), 2.5), sharey=True,
                             layout="constrained", squeeze=False)
    for ax, part in zip(axes[0], parts):
        members = [a for a in trained if arms.describe(a)["part"] == part]
        for arm, color, ls in zip([REFERENCE] + members, LINES, STYLES):
            c = curves[curves.arm == arm]
            c = c[(c.folds >= 0.5 * c.folds.max()) & (c.epoch >= start)]   # until half the folds have stopped
            ax.plot(c.epoch, c.tune_log_loss, color=color, ls=ls, lw=1.1,
                    label="MorphoNup" if arm == REFERENCE else arms.describe(arm)["label"])
        ax.set_title(part, fontsize=8, loc="left", color=INK)
        ax.set_xlabel("Epoch")
        ax.legend(loc="upper right", fontsize=5.5, handlelength=2.2)
        ax.grid(axis="y")
    axes[0, 0].set_ylabel("Tune log loss")
    w.save(fig, "training_curves", f"{name.capitalize()} analysis: log loss on the tune rows per epoch of the base runs, "
           "mean over every outer fold and seed, for MorphoNup and the variants of each part, from epoch "
           f"{start} (where checkpoint selection starts); a curve stops when fewer than half of its folds are still "
           "training (early stopping), so its last epochs average only the longer-running folds.",
           curves[curves.arm.isin([REFERENCE] + trained)])


def _fig_training_diagnostics(w, name, fold_info, present, **_):
    """Selected epoch and train - test accuracy gap per fold: how each change affects fitting and overfitting."""
    order = _layout(present, name, reference=True)
    if fold_info.best_epoch.isna().all():
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.27 * len(order) + 1.0), sharey=True, layout="constrained")
    rng = np.random.default_rng(0)
    for ax, (col, xlabel, scale) in zip(axes, (("best_epoch", "Selected epoch per outer fold", 1),
                                              ("gap_train_minus_test", "Train − test accuracy per fold (pp)", 100))):
        for yi, arm, _ in order:
            if arm is None:
                continue
            v = scale * fold_info.loc[fold_info.arm == arm, col].dropna().to_numpy(float)
            if not len(v):
                continue
            is_ref = arm == REFERENCE
            ax.boxplot(v, positions=[yi], orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=REF_FILL if is_ref else "white", edgecolor=BLUE if is_ref else INK2,
                                     linewidth=0.9),
                       medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=INK2, linewidth=0.9),
                       capprops=dict(color=INK2, linewidth=0.9), zorder=2)
            ax.scatter(v, yi + rng.uniform(-0.22, 0.22, len(v)), s=6, color=BLUE if is_ref else MUTED, alpha=0.75,
                       linewidths=0, zorder=3)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="x")
    _label_rows(axes[0], order)
    w.save(fig, "training_diagnostics", f"{name.capitalize()} analysis: per outer fold and seed, the epoch chosen on the "
           "tune rows (left; cap 120) and the gap between training and test accuracy (right) of the reported model. "
           "Box: median and quartiles, whiskers 1.5 × IQR; MorphoNup in blue.",
           fold_info[["arm", "seed", "fold", "best_epoch", "gap_train_minus_test"]])


def _fig_parameters(w, name, seeds, present, **_):
    order = _layout(present, name, reference=True)
    fig, ax = plt.subplots(figsize=(4.8, 0.27 * len(order) + 1.0), layout="constrained")
    params = seeds.drop_duplicates("arm").set_index("arm")["parameters"]
    for yi, arm, _ in order:
        if arm is None:
            continue
        ax.barh(yi, params[arm], height=0.62, color=BLUE if arm == REFERENCE else MUTED, edgecolor="white", linewidth=1)
        ax.annotate(f"{params[arm]:,}", (params[arm], yi), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK2)
    _label_rows(ax, order)
    ax.set(xlabel="Parameters (trainable + non-trainable)", xlim=(0, params.max() * 1.22))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
    ax.grid(axis="x")
    w.save(fig, "parameters", f"{name.capitalize()} analysis: parameter count of each variant (MorphoNup in blue).",
           params.reindex([a for _, a, _ in order if a]).reset_index())


def _fig_roc_pr_arms(w, k_dir, name, seeds, present, **_):
    """Modality: MorphoNup and every input set on full axes (preprocessing variants differ by ~0.001 AUC: roc_pr_parts)."""
    present = [a for a in present if arms.describe(a)["part"] == "Input set"]
    if name != "modality" or not present:
        return
    common = sorted(set.intersection(*[set(seeds.seed[seeds.arm == a]) for a in [REFERENCE] + present]))
    if not common or len(present) + 1 > len(LINES):
        return
    seed = common[0]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), layout="constrained")
    parts = []
    for arm, color, ls in zip([REFERENCE] + present, LINES, STYLES):
        m, d = _load(k_dir, arm, seed)
        parts.append(_roc_pr(axes, d.label.to_numpy(), d.probability.to_numpy(), color,
                             f"{arms.describe(arm)['label']} ({m['roc_auc']:.3f})", f"{arms.describe(arm)['label']} ({m['pr_auc']:.3f})",
                             arm, ls=ls, lw=1.3).assign(arm=arm))
    _chance(axes, d.label.mean())
    axes[0].legend(loc="lower right", fontsize=6.5, title="ROC AUC", title_fontsize=7)
    axes[1].legend(loc="lower left", fontsize=6.5, title="PR AUC", title_fontsize=7)
    w.save(fig, "roc_pr_arms", f"Modality analysis, seed {seed}: pooled out-of-fold ROC and precision-recall curves "
           "of MorphoNup and each input-set variant (legend: ROC AUC / average precision of this seed).", pd.concat(parts, ignore_index=True))


def refresh(k_dir):
    """Recompute every arm group that has at least one arm with a finished seed in common with MorphoNup."""
    k_dir = Path(k_dir)
    (k_dir / "analysis").mkdir(parents=True, exist_ok=True)
    with open(k_dir / "analysis" / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)          # parallel arm jobs of one dataset/k may finish a seed together
        ref = set(seeds_of(k_dir, REFERENCE))
        made = []
        for name, group in arms.GROUPS.items():
            present = [a for a in group if set(seeds_of(k_dir, a)) & ref]
            if present:
                made += compare(k_dir, present, name)
        return made
