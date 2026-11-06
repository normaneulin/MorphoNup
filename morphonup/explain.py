"""Explaining the saved MorphoNup models on their own outer test rows (no training).

Feature attribution only: how much each input of the model (one-hot sequence, k-mer composition, shape channels)
contributes to its predictions. Motif and sequence-pattern analyses are a separate study and are not done here.

For each outer fold the saved model (refit by default) is rebuilt, checked against its saved test predictions, and
explained on a class-balanced sample of that fold's test rows; the background is a class-balanced sample of the same
fold's training rows.

1. Group Shapley values (interventional, exact) at three resolutions, all from one set of model evaluations.
   The finest players are the one-hot sequence (which also feeds the motif statistics), the 1-4-mer frequencies and
   every shape channel (MGW, Roll, EP, PLS 1-6): 11 players, all 2^11 coalitions evaluated, so there is no sampling
   approximation. A coalition keeps the target's values for its players and takes every other input from a
   background sequence; its value is the mean over the background of the reported probability (mean of the forward
   and RC views). The coarser levels are the same game with players merged (a merged player is in or out as a whole):
     modality   Sequence, k-mers, Shape (all 9 shape channels)               3 players
     tracks     Sequence, k-mers, MGW, Roll, EP, PLS block                   6 players
     channels   Sequence, k-mers, MGW, Roll, EP, PLS 1 ... PLS 6            11 players
   At every level the values of a sequence sum to f(x) - mean_b f(b) (checked). Levels answer different questions and
   do not add up across levels: Shape at the modality level is not the sum of its channels at the channel level.
   Pairwise Shapley interaction indices are computed at the tracks level.
   Caveat: mixing inputs from different sequences creates combinations that never occur together (k-mers and shape
   are computed from the sequence), as in any interventional SHAP.
2. Integrated gradients on the logit of each view (midpoint rule; baselines: uniform 0.25 one-hot, z-score 0 for
   k-mers and shape): where along the sequence the one-hot sequence and each shape channel act. RC-view attributions
   are mapped back to forward coordinates and averaged with the forward view. Completeness errors are recorded.
3. PLS loadings of each fold, to read the PLS players in terms of the 7 original tracks.

Output: <results>/<dataset>/k<K>/<arm>/explain/seed_<s>/ : fold_<ff>/ arrays, figures/, shap_table, audit.json.
Statements are associations learned by the model, not causal claims.
"""
import json
import time
from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from . import arms
from .analysis import _write_table
from .cv import fold_inputs, outer_parts
from .figures import BLUE, INK, INK2, MUTED, ORANGE, Writer, style
from .network import build_core
from .training import build_model, predict_views

DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b8302f"])
CLASS = {1: ("Nucleosome", BLUE), 0: ("Linker", ORANGE)}
LEVELS = ("modality", "tracks", "channels")
LEVEL_TITLE = dict(modality="Input modality", tracks="Shape tracks (PLS block as one)", channels="Shape channels")
OUTCOMES = [("TP", 1, True, BLUE), ("FN", 1, False, "#86b6ef"), ("TN", 0, False, ORANGE), ("FP", 0, True, "#f4b393")]
META = ["index", "fold", "label", "probability", "base_value"]


# ---------------------------------------------------------------- model of one fold

def load_fold(ds, X, arm_dir, part, seed, stage="refit"):
    """Rebuild preprocessing (from the stage's own training rows) and load the saved weights of one fold."""
    arm_dir = Path(arm_dir)
    cfg, shape = arms.resolve(arm_dir.name)
    fit = part["fit"] if stage == "base" else np.sort(np.concatenate([part["fit"], part["tune"]]))
    reducer, views = fold_inputs(ds, X, fit, shape)
    tf.keras.backend.clear_session()
    core = build_core(cfg, reducer.channels, reducer.groups())
    model = build_model(cfg, core)
    model.load_weights(arm_dir / f"seed_{seed}" / f"fold_{part['fold']:02d}" / stage / "best.weights.h5")
    return core, model, cfg, reducer, views, fit


def check_predictions(model, cfg, views, part, stage_dir):
    """Rebuilt model vs the test probabilities saved by the training run: max |difference| and class flips at 0.5.
    GPU (TF32) and CPU float32 differ by ~1e-4, hence the tolerance 1e-3 of tests/check_inference_parity.py."""
    saved = pd.read_csv(stage_dir / "test_predictions.csv").set_index("index").probability.loc[part["test"]].to_numpy()
    a, b, _ = views(part["test"])
    p = predict_views(model, a, b, cfg)[2]
    return float(np.max(np.abs(p - saved))), int(((p >= 0.5) != (saved >= 0.5)).sum())


# ---------------------------------------------------------------- players and levels

def channel_label(name):
    """PLS0_3 -> PLS 4 (1-based component number); other channel names unchanged."""
    head, _, tail = name.partition("_")
    if tail.isdigit() and head.rstrip("0123456789") in ("PLS", "PCA"):
        return f"{head.rstrip('0123456789')} {int(tail) + 1}"
    return name


def fine_players(channel_names):
    """Finest players: name -> (input key, channel indices or None); one player per shape channel."""
    out = {"Sequence": ("sequence", None), "k-mers": ("kmers", None)}
    for i, name in enumerate(channel_names):
        out[channel_label(name)] = ("physics", [i])
    covered = sorted(i for key, idx in out.values() if key == "physics" for i in idx)
    if covered != list(range(len(channel_names))):
        raise ValueError("the shape players must cover every shape channel exactly once")
    return out


def levels(channel_names):
    """level -> {player name: [finest-player indices]} (finest players: Sequence, k-mers, then one per channel)."""
    tracks = {"Sequence": [0], "k-mers": [1]}
    for i, name in enumerate(channel_names):
        head = name.partition("_")[0]
        block = head.rstrip("0123456789") in ("PLS", "PCA") and "_" in name
        tracks.setdefault(f"{head.rstrip('0123456789')} block" if block else name, []).append(i + 2)
    fine = list(fine_players(channel_names))
    return dict(modality={"Sequence": [0], "k-mers": [1], "Shape": list(range(2, len(fine)))},
                tracks=tracks, channels={n: [i] for i, n in enumerate(fine)})


# ---------------------------------------------------------------- exact group Shapley values

def coalition_values(core, cfg, targets, background, players, batch=4096):
    """v(S) for every coalition S of the players: (2^n, T) mean reported probability over the background."""
    names = list(players)
    n, T, B = len(names), len(targets[0]["sequence"]), len(background[0]["sequence"])
    if n > 12:
        raise ValueError(f"{n} players need 2^{n} coalitions")
    call = tf.function(lambda x: tf.sigmoid(core(x, training=False)), reduce_retracing=True)
    values = np.zeros((2 ** n, T))
    channels = targets[0]["physics"].shape[-1]
    for mask in range(2 ** n):
        keep = {"sequence": False, "kmers": False}
        keep_phys = np.zeros(channels, bool)
        for j, name in enumerate(names):
            if mask >> j & 1:
                key, idx = players[name]
                if key == "physics":
                    keep_phys[idx] = True
                else:
                    keep[key] = True
        probs = []
        for t_view, b_view in zip(targets, background):
            mixed = {}
            for key in ("sequence", "kmers"):
                src = t_view[key][:, None] if keep[key] else b_view[key][None]
                mixed[key] = np.broadcast_to(src, (T, B) + t_view[key].shape[1:])
            mixed["physics"] = np.where(keep_phys, t_view["physics"][:, None], b_view["physics"][None])
            flat = {k: np.ascontiguousarray(v.reshape((T * B,) + v.shape[2:]), dtype=np.float32) for k, v in mixed.items()}
            p = np.concatenate([np.asarray(call({k: v[s:s + batch] for k, v in flat.items()})).ravel()
                                for s in range(0, T * B, batch)])
            probs.append(p.reshape(T, B))
        p = probs[0] if not cfg.get("rc_view", True) else 0.5 * (probs[0] + probs[1])
        values[mask] = p.mean(1)
    return values


def merged_game(values, members):
    """Values (2^m, T) of the game whose player j is the union of the finest players `members[j]`."""
    bits = [sum(1 << i for i in g) for g in members]
    masks = [sum(bits[j] for j in range(len(bits)) if c >> j & 1) for c in range(2 ** len(bits))]
    return values[masks]


def shapley(values, members):
    """Exact Shapley values (T, m) of the merged game, its empty-coalition value and its full-coalition value."""
    v = merged_game(values, members)
    m = len(members)
    phi = np.zeros((v.shape[1], m))
    for j in range(m):
        for c in range(2 ** m):
            if not c >> j & 1:
                s = bin(c).count("1")
                phi[:, j] += factorial(s) * factorial(m - s - 1) / factorial(m) * (v[c | 1 << j] - v[c])
    return phi, v[0], v[-1]


def interactions(values, members):
    """Pairwise Shapley interaction indices (T, m, m) of the merged game (diagonal 0)."""
    v = merged_game(values, members)
    m = len(members)
    out = np.zeros((v.shape[1], m, m))
    for i in range(m):
        for j in range(i + 1, m):
            for c in range(2 ** m):
                if c >> i & 1 or c >> j & 1:
                    continue
                s = bin(c).count("1")
                w = factorial(s) * factorial(m - s - 2) / factorial(m - 1)
                out[:, i, j] += w * (v[c | 1 << i | 1 << j] - v[c | 1 << i] - v[c | 1 << j] + v[c])
            out[:, j, i] = out[:, i, j]
    return out


# ---------------------------------------------------------------- integrated gradients

def integrated_gradients(core, targets, steps=32, chunk=16):
    """IG of the logit for each view; RC attributions mapped to forward coordinates; mean of the two views.

    Returns sequence (T,147,4), physics (T,143,C), k-mer total (T,), completeness error (T,) of the forward view.
    """
    @tf.function(reduce_retracing=True)
    def grads(x):
        with tf.GradientTape() as tape:
            tape.watch(list(x.values()))
            z = core(x, training=False)
        return tape.gradient(z, x), z

    alphas = (np.arange(steps, dtype=np.float32) + 0.5) / steps
    out = []
    for view in targets:
        base = {"sequence": np.full_like(view["sequence"][:1], 0.25), "kmers": np.zeros_like(view["kmers"][:1]),
                "physics": np.zeros_like(view["physics"][:1])}
        attr = {k: np.zeros_like(v) for k, v in view.items()}
        error = np.zeros(len(view["sequence"]))
        for s in range(0, len(view["sequence"]), chunk):
            x = {k: v[s:s + chunk] for k, v in view.items()}
            n = len(x["sequence"])
            path = {k: (base[k][:, None] + alphas.reshape((1, steps) + (1,) * (v.ndim - 1)) * (v[:, None] - base[k][:, None]))
                    .reshape((n * steps,) + v.shape[1:]).astype(np.float32) for k, v in x.items()}
            g, _ = grads(path)
            for k, v in x.items():
                mean_grad = np.asarray(g[k]).reshape((n, steps) + v.shape[1:]).mean(1)
                attr[k][s:s + n] = (v - base[k]) * mean_grad
            z_x = np.asarray(core(x, training=False)).ravel()
            z_b = np.asarray(core({k: np.repeat(base[k], n, 0) for k in x}, training=False)).ravel()
            total = sum(attr[k][s:s + n].reshape(n, -1).sum(1) for k in x)
            error[s:s + n] = total - (z_x - z_b)
        out.append((attr, error))
    (fa, ferr), (ra, _) = out
    sequence = 0.5 * (fa["sequence"] + ra["sequence"][:, ::-1, ::-1])
    physics = 0.5 * (fa["physics"] + ra["physics"][:, ::-1])
    kmers = 0.5 * (fa["kmers"].sum(1) + ra["kmers"].sum(1))
    return sequence, physics, kmers, ferr


# ---------------------------------------------------------------- driver

def balanced_sample(rng, rows, y, n):
    """n rows (half per class, as far as available), sorted."""
    return np.sort(np.concatenate([rng.choice(rows[y[rows] == c], min(n // 2, int((y[rows] == c).sum())), replace=False)
                                   for c in (0, 1)]))


def explain(ds, arm_dir, k, seed, folds=None, stage="refit", n_targets=200, n_background=50, steps=32, out=None,
            log=print):
    """Explain the saved models of one seed; writes arrays, tables, audit.json and figures."""
    from .preprocess import shape_tracks
    arm_dir = Path(arm_dir)
    out = Path(out) if out else arm_dir / "explain" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    X = shape_tracks(ds, arms.resolve(arm_dir.name)[1])
    audit = dict(arm=arm_dir.name, stage=stage, k=k, seed=seed, n_targets_per_fold=n_targets,
                 n_background=n_background, ig_steps=steps, folds={})
    for part in outer_parts(ds, k):
        if folds and part["fold"] not in folds:
            continue
        fdir = out / f"fold_{part['fold']:02d}"
        if (fdir / "done.json").exists():
            log(f"skip explain fold {part['fold']}")
            audit["folds"][part["fold"]] = json.loads((fdir / "done.json").read_text())
            continue
        fdir.mkdir(exist_ok=True)
        core, model, cfg, reducer, views, fit = load_fold(ds, X, arm_dir, part, seed, stage)
        stage_dir = arm_dir / f"seed_{seed}" / f"fold_{part['fold']:02d}" / stage
        reproduced, flips = check_predictions(model, cfg, views, part, stage_dir)
        if reproduced > 1e-3 or flips:
            raise RuntimeError(f"fold {part['fold']}: rebuilt model differs from the saved test predictions "
                               f"(max |Δp| = {reproduced:.2e}, {flips} class flips); not explaining a different model")
        rng = np.random.default_rng(seed + part["fold"])
        idx = balanced_sample(rng, part["test"], ds.y, n_targets)
        bg = balanced_sample(rng, fit, ds.y, n_background)
        ta, tb, ty = views(idx)
        ba, bb, _ = views(bg)
        names = reducer.channel_names()
        players = fine_players(names)
        t0 = time.time()
        values = coalition_values(core, cfg, (ta, tb), (ba, bb), players)
        shap_seconds = time.time() - t0
        np.savez_compressed(fdir / "coalition_values.npz", values=values, players=np.array(list(players)), index=idx,
                            label=ty.astype(int), background_index=bg)
        additivity = {}
        for level, groups in levels(names).items():
            phi, base, fx = shapley(values, list(groups.values()))
            additivity[level] = float(np.max(np.abs(phi.sum(1) + base - fx)))
            pd.DataFrame(phi, columns=list(groups)).assign(index=idx, fold=part["fold"], label=ty.astype(int),
                                                          probability=fx, base_value=base) \
                .to_csv(fdir / f"shap_{level}.csv", index=False)
            if level == "tracks":
                np.savez_compressed(fdir / "interactions_tracks.npz", interactions=interactions(values, list(groups.values())),
                                    players=np.array(list(groups)), label=ty.astype(int))
        t0 = time.time()
        seq_attr, phys_attr, kmer_attr, ig_err = integrated_gradients(core, (ta, tb), steps)
        ig_seconds = time.time() - t0
        np.savez_compressed(fdir / "integrated_gradients.npz", index=idx, label=ty.astype(int), sequence=seq_attr,
                            physics=phys_attr, kmers=kmer_attr, completeness_error=ig_err,
                            channel_names=np.array([channel_label(c) for c in names]))
        summary = pd.DataFrame({"index": idx, "gc_fraction": ta["sequence"][:, :, 1:3].sum(2).mean(1)})
        for c, name in enumerate(names):                       # mean (z-scored / PLS score) value of each channel
            summary[channel_label(name)] = ta["physics"][:, :, c].mean(1)
        summary.to_csv(fdir / "input_summary.csv", index=False)
        state = np.load(stage_dir / "preprocessor.npz")
        if "block0_rotation" in state.files:
            np.savez_compressed(fdir / "pls_loadings.npz", rotation=state["block0_rotation"], idx=state["block0_idx"])
        done = dict(n_targets=int(len(idx)), n_background=int(len(bg)),
                    background_nucleosome_fraction=float(ds.y[bg].mean()), players=list(players),
                    levels={lv: list(g) for lv, g in levels(names).items()},
                    rebuilt_vs_saved_max_abs_probability_difference=reproduced, rebuilt_vs_saved_class_flips=flips,
                    shapley_additivity_max_error=additivity, base_value_mean=float(values[0].mean()),
                    ig_completeness_max_abs_error=float(np.max(np.abs(ig_err))),
                    ig_completeness_median_abs_error=float(np.median(np.abs(ig_err))),
                    seconds_shapley=round(shap_seconds, 1), seconds_integrated_gradients=round(ig_seconds, 1))
        (fdir / "done.json").write_text(json.dumps(done, indent=2))
        audit["folds"][part["fold"]] = done
        log(f"explain fold {part['fold']}: {len(idx)} targets, {len(bg)} background, {2 ** len(players)} coalitions in "
            f"{shap_seconds:.0f}s, IG {ig_seconds:.0f}s; rebuilt model |Δp| {reproduced:.1e}; additivity "
            f"{max(additivity.values()):.1e}; IG completeness median {done['ig_completeness_median_abs_error']:.1e}")
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    return explain_figures(out)


# ---------------------------------------------------------------- figures and tables

def _load(fold_dirs, name):
    return pd.concat([pd.read_csv(d / name) for d in fold_dirs], ignore_index=True)


def _players(df):
    return [c for c in df.columns if c not in META]


def _importance(ax, df, order, show_folds=True):
    """Horizontal bars of mean |SHAP| (all sequences) with one dot per outer fold; returns the y positions."""
    y = np.arange(len(order))
    ax.barh(y, df[order].abs().mean().to_numpy(), height=0.6, color=BLUE, edgecolor="white", linewidth=1)
    if show_folds:
        for fold, d in df.groupby("fold"):
            ax.scatter(d[order].abs().mean().to_numpy(), y, s=9, color=INK, zorder=3,
                       label="Outer fold" if fold == df.fold.min() else None)
    ax.set(yticks=y, yticklabels=order)
    ax.grid(axis="x")
    return y


def _fig_level(w, level, df, n_folds):
    order = list(df[_players(df)].abs().mean().sort_values().index)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.32 * len(order) + 1.4), sharey=True, layout="constrained")
    y = _importance(axes[0], df, order)
    axes[0].set_xlabel("Mean |SHAP| (probability)")
    axes[0].legend(loc="lower right", fontsize=7)
    for (label, (name, color)), off in zip(CLASS.items(), (0.17, -0.17)):
        axes[1].barh(y + off, df[df.label == label][order].mean().to_numpy(), height=0.32, color=color, label=name)
    axes[1].axvline(0, color=INK2, lw=0.8)
    axes[1].set_xlabel("Mean SHAP by observed class")
    axes[1].legend(loc="lower right", fontsize=7)
    axes[1].grid(axis="x")
    rows = [dict(player=p, mean_abs_shap=df[p].abs().mean(), share_of_total=df[p].abs().mean() / df[order].abs().mean().sum(),
                 mean_shap_nucleosome=df.loc[df.label == 1, p].mean(), mean_shap_linker=df.loc[df.label == 0, p].mean(),
                 **{f"mean_abs_shap_fold_{f}": d[p].abs().mean() for f, d in df.groupby("fold")}) for p in order[::-1]]
    w.save(fig, f"shap_{level}", f"Exact group Shapley values of the reported nucleosome probability, {LEVEL_TITLE[level].lower()} "
           f"level ({len(df)} held-out test sequences, {n_folds} outer fold(s); background: class-balanced training rows "
           "of each fold). Left: mean absolute contribution (bars: all sequences; dots: per fold). Right: mean signed "
           "contribution by observed class (positive pushes towards nucleosome).", pd.DataFrame(rows))


def _fig_overview(w, shap, n_folds):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1, 1]))
    rows = []
    for ax, level, letter in zip(axes, LEVELS, "abc"):
        df = shap[level]
        order = list(df[_players(df)].abs().mean().sort_values().index)
        y = _importance(ax, df, order, show_folds=False)
        total = df[order].abs().mean().sum()
        for yi, p in zip(y, order):
            v = df[p].abs().mean()
            ax.text(v, yi, f" {100 * v / total:.0f} %", va="center", fontsize=6.5, color=INK2)
            rows.append(dict(level=level, player=p, mean_abs_shap=v, share_of_total=v / total))
        ax.set_title(f"{letter}  {LEVEL_TITLE[level]}", fontsize=8, loc="left", color=INK)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_xlim(0, df[order].abs().mean().max() * 1.35)
    w.save(fig, "shap_overview", f"Group Shapley importance at three resolutions of the same game ({len(shap['modality'])} "
           f"held-out test sequences, {n_folds} outer fold(s)): (a) input modality, (b) shape tracks with the PLS block as "
           "one player, (c) every shape channel. Labels: share of the total mean |SHAP| at that level. Values of "
           "different levels are not additive across levels.", pd.DataFrame(rows))


def _summary_value(player, summary):
    """Scalar input value used to colour a player's points: GC fraction for sequence and k-mers, mean channel value
    for a shape channel, mean of the block's first component for a block."""
    if player in ("Sequence", "k-mers"):
        return summary["gc_fraction"].to_numpy(), "GC fraction"
    if player in summary.columns:
        return summary[player].to_numpy(), f"mean {player}"
    if player.endswith("block"):
        first = f"{player.split()[0]} 1"
        return summary[first].to_numpy(), f"mean {first}"
    return None, None


def _aligned(summary, df):
    """Input summaries in the row order of a SHAP table."""
    return summary.set_index("index").loc[df["index"].to_numpy()].reset_index(drop=True)


def _fig_beeswarm(w, level, df, summary, n_folds):
    order = list(df[_players(df)].abs().mean().sort_values().index)
    inputs = _aligned(summary, df)
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(order) + 1.3), layout="constrained")
    rng = np.random.default_rng(0)
    rows = []
    for yi, p in enumerate(order):
        v = df[p].to_numpy()
        raw, what = _summary_value(p, inputs)
        hist, edges = np.histogram(v, bins=40)
        dens = hist[np.clip(np.digitize(v, edges[1:-1]), 0, len(hist) - 1)] / max(hist.max(), 1)
        jitter = rng.uniform(-1, 1, len(v)) * 0.38 * dens
        color = stats.rankdata(raw) / len(raw) if raw is not None else np.full(len(v), 0.5)
        ax.scatter(v, yi + jitter, c=color, cmap=DIVERGING, vmin=0, vmax=1, s=3.5, linewidths=0, alpha=0.8, rasterized=True)
        rows += [dict(player=p, index=int(i), shap=float(a), input_value=float(r) if raw is not None else np.nan,
                      input_definition=what or "") for i, a, r in zip(df["index"], v, raw if raw is not None else v)]
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set(yticks=range(len(order)), yticklabels=order, xlabel="SHAP value (change in nucleosome probability)")
    ax.grid(axis="x")
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=DIVERGING), ax=ax, shrink=0.6, aspect=25, pad=0.01)
    bar.set_ticks([0, 1], labels=["low", "high"])
    bar.set_label("Input value (rank)", fontsize=7)
    w.save(fig, f"shap_beeswarm_{level}", f"SHAP summary plot, {LEVEL_TITLE[level].lower()} level: one dot per explained "
           f"test sequence ({len(df)}, {n_folds} fold(s)), coloured by the rank of a scalar summary of that player's input "
           "(sequence and k-mers: GC fraction; a shape channel: its mean over the 143 positions; the PLS block: the mean "
           "of PLS 1). The summary is a reading aid; the player itself is the whole input group.", pd.DataFrame(rows))


def _fig_dependence(w, df, summary):
    order = list(df[_players(df)].abs().mean().sort_values(ascending=False).index)
    inputs = _aligned(summary, df)
    cols = 3
    rows_n = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.2, 2.1 * rows_n + 0.3), layout="constrained", squeeze=False)
    rows = []
    for ax, p in zip(axes.ravel(), order):
        raw, what = _summary_value(p, inputs)
        for label, (name, color) in CLASS.items():
            sel = df.label.to_numpy() == label
            ax.scatter(raw[sel], df[p].to_numpy()[sel], s=3, color=color, alpha=0.45, linewidths=0, label=name,
                       rasterized=True)
        rho = stats.spearmanr(raw, df[p])[0]
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{p}  (Spearman ρ = {rho:+.2f})", fontsize=7.5, loc="left", color=INK)
        ax.set_xlabel(what, fontsize=7.5)
        ax.set_ylabel("SHAP", fontsize=7.5)
        ax.grid(alpha=0.6)
        rows.append(dict(player=p, input_definition=what, spearman_rho=rho))
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=6.5, markerscale=3, loc="best")
    w.save(fig, "shap_dependence_tracks", "SHAP dependence, tracks level: each player's Shapley value against a scalar "
           "summary of its input (as in the summary plot), coloured by observed class; ρ: Spearman correlation over all "
           "explained sequences.", pd.DataFrame(rows))


def _fig_distribution(w, df):
    order = list(df[_players(df)].abs().mean().sort_values().index)
    fig, ax = plt.subplots(figsize=(6.4, 0.36 * len(order) + 1.2), layout="constrained")
    rng = np.random.default_rng(0)
    for yi, p in enumerate(order):
        for label, (cls, color) in CLASS.items():
            v = df.loc[df.label == label, p].to_numpy()
            ax.scatter(v, yi + (0.15 if label else -0.15) + rng.uniform(-0.1, 0.1, len(v)), s=4, color=color, alpha=0.5,
                       linewidths=0, label=cls if yi == len(order) - 1 else None, rasterized=True)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set(yticks=range(len(order)), yticklabels=order, xlabel="SHAP value (change in nucleosome probability)")
    ax.legend(loc="lower right", fontsize=7, markerscale=3, title="Observed", title_fontsize=7)
    ax.grid(axis="x")
    w.save(fig, "shap_distribution_tracks", "Group Shapley value of every explained test sequence, tracks level, by "
           "observed class.", df)


def _fig_outcome(w, df):
    order = list(df[_players(df)].abs().mean().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(7.0, 2.9), layout="constrained")
    x = np.arange(len(order))
    rows = []
    for k, (name, label, positive, color) in enumerate(OUTCOMES):
        sel = (df.label == label) & ((df.probability >= 0.5) == positive)
        v = df.loc[sel, order].mean().to_numpy() if sel.any() else np.full(len(order), np.nan)
        ax.bar(x + (k - 1.5) * 0.2, v, width=0.2, color=color, label=f"{name} (n = {int(sel.sum())})")
        rows += [dict(outcome=name, n=int(sel.sum()), player=p, mean_shap=float(a)) for p, a in zip(order, v)]
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(xticks=x, xticklabels=order, ylabel="Mean SHAP (probability)")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.grid(axis="y")
    w.save(fig, "shap_by_outcome_tracks", "Mean group Shapley value per player by prediction outcome (threshold 0.5): true "
           "positives (nucleosome, predicted nucleosome), false negatives, true negatives (linker, predicted linker) and "
           "false positives. Shows which inputs push the misclassified sequences the wrong way.", pd.DataFrame(rows))


def _fig_interactions(w, fold_dirs):
    data = [np.load(d / "interactions_tracks.npz") for d in fold_dirs]
    names = [str(p) for p in data[0]["players"]]
    inter = np.concatenate([d["interactions"] for d in data])
    mean_abs, mean = np.abs(inter).mean(0), inter.mean(0)
    fig, ax = plt.subplots(figsize=(4.8, 3.9), layout="constrained")
    shown = np.where(np.eye(len(names), dtype=bool), np.nan, mean_abs)
    im = ax.imshow(shown, cmap=plt.cm.Blues, vmin=0)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                ax.text(j, i, f"{mean_abs[i, j]:.3f}\n({mean[i, j]:+.3f})", ha="center", va="center", fontsize=5.5,
                        color="white" if mean_abs[i, j] > 0.6 * np.nanmax(shown) else INK)
    ax.set(xticks=range(len(names)), xticklabels=names, yticks=range(len(names)), yticklabels=names)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Mean |interaction|")
    rows = [dict(player_a=names[i], player_b=names[j], mean_abs_interaction=mean_abs[i, j], mean_interaction=mean[i, j])
            for i in range(len(names)) for j in range(i + 1, len(names))]
    w.save(fig, "shap_interactions_tracks", f"Pairwise Shapley interaction index between players, tracks level ({len(inter)} "
           "test sequences): mean absolute value (in brackets: mean signed value). Large values mean that the two inputs "
           "act together rather than separately.", pd.DataFrame(rows))


def _fig_agreement(w, shap_modality, ig):
    tot = dict(Sequence=np.concatenate([g["sequence"].reshape(len(g["sequence"]), -1).sum(1) for g in ig]),
               **{"k-mers": np.concatenate([g["kmers"] for g in ig])},
               Shape=np.concatenate([g["physics"].reshape(len(g["physics"]), -1).sum(1) for g in ig]))
    names = ["Sequence", "k-mers", "Shape"]
    s_share = np.array([shap_modality[n].abs().mean() for n in names])
    i_share = np.array([np.abs(tot[n]).mean() for n in names])
    s_share, i_share = s_share / s_share.sum(), i_share / i_share.sum()
    fig, ax = plt.subplots(figsize=(4.6, 2.6), layout="constrained")
    x = np.arange(len(names))
    ax.bar(x - 0.18, 100 * s_share, width=0.36, color=BLUE, label="Group Shapley (probability)")
    ax.bar(x + 0.18, 100 * i_share, width=0.36, color=MUTED, label="Integrated gradients (logit)")
    for xi, a, b in zip(x, s_share, i_share):
        ax.text(xi - 0.18, 100 * a, f"{100 * a:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.text(xi + 0.18, 100 * b, f"{100 * b:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set(xticks=x, xticklabels=names, ylabel="Share of total |attribution| (%)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y")
    w.save(fig, "attribution_agreement", "Share of each input modality in the total absolute attribution by two methods "
           "on the same test sequences: exact group Shapley values (reported probability) and integrated gradients "
           "(summed over positions; logit). Agreement in ranking supports the attribution; magnitudes are not comparable "
           "between methods.", pd.DataFrame(dict(modality=names, shapley_share=s_share, ig_share=i_share)))


def _fig_ig(w, ig):
    seq = np.concatenate([g["sequence"] for g in ig])
    phys = np.concatenate([g["physics"] for g in ig])
    label = np.concatenate([g["label"] for g in ig])
    channels = [str(c) for c in ig[0]["channel_names"]]
    pos_seq, pos_phys = np.arange(147) - 73, np.arange(143) - 71

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True, layout="constrained")
    rows = []
    for lab, (name, color) in CLASS.items():
        s, a = seq[label == lab].sum(2).mean(0), np.abs(seq[label == lab]).sum(2).mean(0)
        axes[0].plot(pos_seq, s, color=color, lw=1.2, label=name, ls="-" if lab else "--")
        axes[1].plot(pos_seq, a, color=color, lw=1.2, ls="-" if lab else "--")
        rows += [dict(observed=name, position=int(p), mean_attribution=float(u), mean_abs_attribution=float(b))
                 for p, u, b in zip(pos_seq, s, a)]
    axes[0].axhline(0, color=INK2, lw=0.7)
    axes[0].set_ylabel("Mean attribution\n(logit)")
    axes[1].set(ylabel="Mean |attribution|\n(logit)", xlabel="Position relative to the dyad (bp)")
    axes[0].legend(fontsize=7, loc="upper right")
    for ax in axes:
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
        ax.grid(axis="y")
    w.save(fig, "ig_sequence_profile", "Integrated-gradients attribution of the one-hot sequence per position (summed over "
           "the four bases), averaged over the explained test sequences of each class; forward and mapped "
           "reverse-complement views averaged; baseline: uniform base composition.", pd.DataFrame(rows))

    fig, axes = plt.subplots(len(channels), 1, figsize=(6.6, 0.62 * len(channels) + 1.0), sharex=True, layout="constrained")
    rows = []
    for ax, (c, name) in zip(np.atleast_1d(axes), enumerate(channels)):
        for lab, (cls, color) in CLASS.items():
            v = phys[label == lab][:, :, c].mean(0)
            ax.plot(pos_phys, v, color=color, lw=1.0, ls="-" if lab else "--", label=cls)
            rows += [dict(channel=name, observed=cls, position=int(p), mean_attribution=float(a)) for p, a in zip(pos_phys, v)]
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvline(0, color=MUTED, lw=0.6, ls=":")
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    np.atleast_1d(axes)[-1].set_xlabel("Position relative to the dyad (bp)")
    w.save(fig, "ig_shape_profile", "Mean integrated-gradients attribution of each shape channel per position, by observed "
           "class (baseline: the training-row mean of each channel).", pd.DataFrame(rows))

    profiles = [np.abs(seq).sum(2).mean(0)] + [np.abs(phys[:, :, c]).mean(0) for c in range(len(channels))]
    names = ["Sequence"] + channels
    fig, ax = plt.subplots(figsize=(6.8, 0.27 * len(names) + 1.3), layout="constrained")
    for r, (name, prof) in enumerate(zip(names, profiles)):
        pos = pos_seq if name == "Sequence" else pos_phys
        ax.imshow((prof / prof.max())[None], aspect="auto", cmap=plt.cm.Blues, vmin=0, vmax=1,
                  extent=(pos[0] - 0.5, pos[-1] + 0.5, r + 0.45, r - 0.45))
    ax.set(ylim=(len(names) - 0.5, -0.5), yticks=range(len(names)), yticklabels=names,
           xlabel="Position relative to the dyad (bp)", xlim=(pos_seq[0] - 0.5, pos_seq[-1] + 0.5))
    ax.axvline(0, color=INK2, lw=0.6, ls=":")
    ax.tick_params(axis="y", length=0)
    fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), ax=ax, shrink=0.8, label="Mean |attribution| (row max = 1)")
    rows = [dict(input=n, position=int(p), mean_abs_attribution=float(v))
            for n, prof in zip(names, profiles) for p, v in zip(pos_seq if n == "Sequence" else pos_phys, prof)]
    w.save(fig, "ig_position_heatmap", "Where along the 147 bp each input acts: mean absolute integrated-gradients "
           "attribution per position for the one-hot sequence and each shape channel, each row scaled to its own maximum "
           "(absolute values in the data file).", pd.DataFrame(rows))


def _fig_loadings(w, fold_dirs):
    loads = [np.load(d / "pls_loadings.npz") for d in fold_dirs if (d / "pls_loadings.npz").exists()]
    if not loads:
        return
    from .preprocess import NAMES14
    tracks = [NAMES14[i] for i in loads[0]["idx"]]
    rot = np.stack([l["rotation"] for l in loads])
    sign = np.sign(np.take_along_axis(rot, np.abs(rot[0]).argmax(0)[None, None, :].repeat(len(rot), 0), 1))
    mean = (rot * sign).mean(0)
    cols = [f"PLS {j + 1}" for j in range(mean.shape[1])]
    fig, ax = plt.subplots(figsize=(4.4, 2.8), layout="constrained")
    lim = np.abs(mean).max()
    im = ax.imshow(mean, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(mean[i, j]) > 0.6 * lim else INK)
    ax.set(xticks=range(len(cols)), xticklabels=cols, yticks=range(len(tracks)), yticklabels=tracks)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Rotation weight")
    w.save(fig, "pls_loadings", f"PLS rotation (z-scored track → component), mean over {len(rot)} fold(s) after aligning "
           "each component's sign; read the PLS players of the SHAP figures through these weights.",
           pd.DataFrame(mean, index=tracks, columns=cols).reset_index().rename(columns={"index": "track"}))


def _tables(out, shap, audit_rows):
    rows = []
    for level in LEVELS:
        df = shap[level]
        order = list(df[_players(df)].abs().mean().sort_values(ascending=False).index)
        total = df[order].abs().mean().sum()
        per_fold = df.groupby("fold")[order].apply(lambda d: d.abs().mean())
        for p in order:
            rows.append({"Level": LEVEL_TITLE[level], "Player": p,
                         "Mean |SHAP|": f"{df[p].abs().mean():.4f} ± {per_fold[p].std(ddof=1):.4f}"
                         if len(per_fold) > 1 else f"{df[p].abs().mean():.4f}",
                         "Share (%)": f"{100 * df[p].abs().mean() / total:.1f}",
                         "Mean SHAP, nucleosome": f"{df.loc[df.label == 1, p].mean():+.4f}",
                         "Mean SHAP, linker": f"{df.loc[df.label == 0, p].mean():+.4f}"})
    a = pd.DataFrame(audit_rows)
    note = (f"Exact group Shapley values of the reported nucleosome probability on {len(shap['modality'])} held-out test "
            f"sequences ({shap['modality'].fold.nunique()} outer fold(s), class-balanced; background: class-balanced "
            "training rows of each fold). Mean |SHAP| ± SD over folds. Levels are separate merges of the same game and "
            "are not additive across levels. Checks: rebuilt models reproduce the saved test probabilities to "
            f"{a.reproduced.max():.1e} (no class changes); Shapley values add up to f(x) − E f(background) to {a.additivity.max():.1e}.")
    _write_table(out, "shap_table", pd.DataFrame(rows), note, "Input attribution of MorphoNup.")


def explain_figures(out):
    """Figures and the summary table from the saved explanation files of every finished fold."""
    style()
    out = Path(out)
    fold_dirs = sorted(d for d in out.glob("fold_*") if (d / "done.json").exists())
    if not fold_dirs:
        return []
    w = Writer(out / "figures")
    shap = {level: _load(fold_dirs, f"shap_{level}.csv") for level in LEVELS}
    summary = _load(fold_dirs, "input_summary.csv")
    ig = [np.load(d / "integrated_gradients.npz") for d in fold_dirs]
    n_folds = len(fold_dirs)
    _fig_overview(w, shap, n_folds)
    for level in LEVELS:
        _fig_level(w, level, shap[level], n_folds)
    for level in ("tracks", "channels"):
        _fig_beeswarm(w, level, shap[level], summary, n_folds)
    _fig_dependence(w, shap["tracks"], summary)
    _fig_distribution(w, shap["tracks"])
    _fig_outcome(w, shap["tracks"])
    _fig_interactions(w, fold_dirs)
    _fig_agreement(w, shap["modality"], ig)
    _fig_ig(w, ig)
    _fig_loadings(w, fold_dirs)
    audit = [json.loads((d / "done.json").read_text()) for d in fold_dirs]
    _tables(out, shap, [dict(reproduced=a["rebuilt_vs_saved_max_abs_probability_difference"],
                             additivity=max(a["shapley_additivity_max_error"].values())) for a in audit])
    return w.finish()
