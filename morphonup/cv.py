"""Outer k-fold cross-validation of the final model: per fold base run -> refit, then pooling.

Per outer fold (make_splits(ds, k, seed=42)):
  fit  = train + calibration rows, tune = checkpoint-selection rows, test = outer test rows (never used in training).
  base : train on fit, select the checkpoint on EMA tune NLL.
  refit: retrain on fit + tune for the base run's best-epoch count, replaying its logged LR schedule
         (chosen on tune rows only). Its test predictions are the reported result. Still one network.
Preprocessing (shape reducer, k-mer z-scores, prototype initialization) is refitted on each stage's own training rows.

Arms (morphonup.arms): `full` is MorphoNup; analysis arms change one named component and are trained the same way.
Layout: <results>/<dataset id>/k<K>/<arm>/seed_<s>/fold_<ff>/{base,refit}/ and seed_<s>/pooled/{base,refit}/.
A stage with fold_summary.json is complete and skipped, so a killed job resumes where it stopped.
Figures (morphonup.figures) are drawn after every fold, every pooled seed and every seed summary; the arm
comparisons (morphonup.analysis) are refreshed whenever a seed of any arm finishes.
"""
import fcntl
import hashlib
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from . import analysis, arms, config, figures
from .data import make_splits
from .metrics import scores
from .network import build_core, initialize
from .preprocess import PhysicalReducer, kmer_views, shape_tracks
from .training import Selection, build_model, make_dataset, predict_views, replay_schedule

STAGES = ("base", "refit")


def outer_parts(ds, k):
    for p in make_splits(ds, k=k, seed=config.SPLIT_SEED):
        yield dict(fold=p["fold"], fit=np.sort(np.array(p["train"] + p["calibration"])),
                   tune=np.array(p["tune"]), test=np.array(p["test"]))


def split_digest(parts):
    text = json.dumps([{r: list(map(int, p[r])) for r in ("fit", "tune", "test")} for p in parts])
    return hashlib.sha256(text.encode()).hexdigest()


def run_dir(dataset_id, k, arm="full", root=None):
    return (root or config.RESULTS_ROOT) / dataset_id / f"k{k}" / arm


def fold_inputs(ds, X, fit, shape_spec, strand_flag=False):
    """Preprocessing fitted on `fit` rows only; returns the reducer and both strands' input views."""
    kmers = kmer_views(ds, fit, strand_flag)
    reducer = PhysicalReducer.from_spec(shape_spec)
    reducer.fit(X[0][fit], ds.y[fit], X[1][fit])
    phys = [reducer.transform(X[s]) for s in (0, 1)]

    def views(idx):
        return tuple({"sequence": ds.arrays["sequence"][s][idx], "kmers": kmers[s][idx], "physics": phys[s][idx]}
                     for s in (0, 1)) + (ds.y[idx].astype(np.float32),)
    return reducer, views


def train_stage(ds, X, part, cfg, shape_spec, seed, folder, stage):
    """One training run. stage='base': fit rows, tune selection. stage='refit': fit+tune rows, replayed schedule."""
    folder.mkdir(parents=True, exist_ok=True)
    fold, fit, tune, test = part["fold"], part["fit"], part["tune"], part["test"]
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed + fold)
    replay = None
    if stage == "refit":
        base = folder.parent / "base"
        epochs = json.loads((base / "fold_summary.json").read_text())["best_epoch"]
        replay = pd.read_csv(base / "history.csv")["learning_rate_used"].to_numpy()[:epochs]
        fit = np.sort(np.concatenate([fit, tune]))
        part = dict(part, fit=fit)
    reducer, views = fold_inputs(ds, X, fit, shape_spec, cfg.get("strand_flag", False))
    np.savez_compressed(folder / "preprocessor.npz", **reducer.state())
    (a, b, y), (va, vb, vy), (ta, tb, ty) = views(fit), views(tune), views(test)
    assert a["kmers"].shape[1] == cfg["kmer_dim"]
    core = build_core(cfg, reducer.channels, reducer.groups())
    model = build_model(cfg, core)
    initialize(core, a, b, seed + fold)
    logger = tf.keras.callbacks.CSVLogger(str(folder / "history.csv"))
    t0 = time.time()
    if replay is None:
        selection = Selection((a, b, y), (va, vb, vy), folder / "best.weights.h5", cfg)
        model.fit(make_dataset(a, b, y, cfg, seed + fold), epochs=cfg["epochs"], verbose=0, shuffle=False,
                  callbacks=[selection, logger, tf.keras.callbacks.TerminateOnNaN()])
        if selection.best_epoch is None:
            raise RuntimeError("No checkpoint selected.")
        model.load_weights(folder / "best.weights.h5")
        best_epoch = selection.best_epoch
    else:
        model.fit(make_dataset(a, b, y, cfg, seed + fold), epochs=len(replay), verbose=0, shuffle=False,
                  callbacks=[replay_schedule(model, replay), logger, tf.keras.callbacks.TerminateOnNaN()])
        model.save_weights(folder / "best.weights.h5")
        best_epoch = len(replay)
    predict = lambda u, v: predict_views(model, u, v, cfg)
    return finish(folder, part, stage, predict, (a, b, y), (va, vb, vy), (ta, tb, ty), best_epoch,
                  core.count_params(), time.time() - t0, reducer.channel_names())


def finish(folder, part, stage, predict, train, tune, test, best_epoch, params, seconds, channel_names):
    pa, pb, p = predict(*test[:2])
    pd.DataFrame(dict(index=part["test"], fold=part["fold"], label=test[2].astype(int),
                      probability_fwd=pa, probability_rc=pb, probability=p)).to_csv(folder / "test_predictions.csv", index=False)
    tr = scores(train[2], predict(*train[:2])[2])
    tu = scores(tune[2], predict(*tune[:2])[2])
    te = scores(test[2], p)
    summary = dict(fold=part["fold"], stage=stage, best_epoch=int(best_epoch), parameters=int(params),
                   seconds=round(seconds, 1), channels=channel_names, n_fit=len(part["fit"]), n_tune=len(part["tune"]),
                   n_test=len(part["test"]), tune_in_training=stage == "refit",
                   train_accuracy=tr["accuracy"], tune_accuracy=tu["accuracy"], test_accuracy=te["accuracy"],
                   train_nll=tr["nll"], tune_nll=tu["nll"], test_nll=te["nll"], test_auc=te["roc_auc"], test_mcc=te["mcc"],
                   test_correct=int(te["tp"] + te["tn"]), gap_train_minus_test=tr["accuracy"] - te["accuracy"])
    (folder / "fold_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def pool(seed_dir, stage, k):
    """Pooled metrics over the k outer test folds of one seed and stage -> seed_dir/pooled/<stage>/."""
    folds = sorted(seed_dir.glob(f"fold_*/{stage}/fold_summary.json"))
    if len(folds) != k:
        return None
    df = pd.concat([pd.read_csv(f.parent / "test_predictions.csv") for f in folds], ignore_index=True)
    summaries = [json.loads(f.read_text()) for f in folds]
    sc = scores(df.label.values, df.probability.values)
    out = dict(stage=stage, k=k, correct=int(sc["tp"] + sc["tn"]), **sc,
               fold_accuracy=[s["test_accuracy"] for s in summaries], best_epochs=[s["best_epoch"] for s in summaries],
               epoch_cap_hits=sum(s["best_epoch"] >= config.SETTINGS["epochs"] for s in summaries),
               mean_gap_train_minus_test=float(np.mean([s["gap_train_minus_test"] for s in summaries])),
               parameters=summaries[0]["parameters"])
    target = seed_dir / "pooled" / stage
    target.mkdir(parents=True, exist_ok=True)
    df.to_csv(target / "test_predictions.csv", index=False)
    (target / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


def draw(fn, *args, log=print):
    """Figures never stop a training job: a failure is logged and scripts/rebuild_figures.py redraws them later."""
    try:
        return fn(*args)
    except Exception as error:  # noqa: BLE001
        log(f"FIGURE ERROR {fn.__name__}({args[0]}): {type(error).__name__}: {error}")


def write_run_config(base_dir, ds, dataset_id, k, arm, cfg, shape_spec, parts):
    """run_config.json of a dataset/k/arm folder; refuses to mix runs with other splits or settings."""
    record = dict(dataset_id=dataset_id, dataset_path=str(ds.path), k=k, arm=arm,
                  arm_change=arms.describe(arm)["change"], split_seed=config.SPLIT_SEED,
                  split_sha256=split_digest(parts), settings=cfg, shape=shape_spec, tensorflow=tf.__version__,
                  data_manifest=ds.manifest)
    record = json.loads(json.dumps(record, default=str))
    path = base_dir / "run_config.json"
    if path.exists():
        old = json.loads(path.read_text())
        for key in ("split_sha256", "settings", "shape"):
            if old[key] != record[key]:
                raise RuntimeError(f"{path} has a different {key}; use a new results root instead of mixing runs.")
        record = {**old, **record}
    base_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return record


SMOKE = dict(epochs=2, warmup_epochs=1, select_start_epoch=1)   # pipeline check only, never research numbers


def run(ds, dataset_id, k, seeds, folds=None, arm="full", root=None, smoke=False, log=print, stages=STAGES):
    cfg, shape_spec = arms.resolve(arm)
    if smoke:
        if root is None or root.resolve() == config.RESULTS_ROOT.resolve():
            raise ValueError("smoke runs must write to a separate results root")
        cfg.update(SMOKE)
    parts = list(outer_parts(ds, k))
    X = shape_tracks(ds, shape_spec)
    base_dir = run_dir(dataset_id, k, arm, root)
    write_run_config(base_dir, ds, dataset_id, k, arm, cfg, shape_spec, parts)
    results = {}
    for seed in seeds:
        seed_dir = base_dir / f"seed_{seed}"
        for part in parts:
            if folds and part["fold"] not in folds:
                continue
            for stage in stages:                         # "refit" alone needs the finished base stage
                folder = seed_dir / f"fold_{part['fold']:02d}" / stage
                if (folder / "fold_summary.json").exists():
                    log(f"skip {dataset_id} k{k} {arm} seed {seed} fold {part['fold']} {stage}")
                    continue
                s = train_stage(ds, X, part, cfg, shape_spec, seed, folder, stage)
                log(f"{dataset_id} k{k} {arm} seed {seed} fold {s['fold']} {stage}: test {s['test_accuracy']*100:.2f}% "
                    f"({s['test_correct']}/{s['n_test']}) tune {s['tune_accuracy']*100:.2f}% "
                    f"train {s['train_accuracy']*100:.2f}% epoch {s['best_epoch']} {s['seconds']:.0f}s")
                draw(figures.fold_figures, folder.parent, cfg["epochs"], log=log)
        for stage in STAGES:
            r = pool(seed_dir, stage, k)
            if r is not None:
                results[(seed, stage)] = r
                log(f"POOLED {dataset_id} k{k} {arm} seed {seed} {stage}: {r['accuracy']*100:.3f}% "
                    f"({r['correct']}/{r['n']}) AUC {r['roc_auc']:.4f} MCC {r['mcc']:.4f} "
                    f"gap {r['mean_gap_train_minus_test']*100:.1f}pp cap hits {r['epoch_cap_hits']}/{k}")
        if any(key[0] == seed for key in results):
            draw(figures.seed_figures, seed_dir, cfg["epochs"], log=log)
            draw(figures.summary_figures, base_dir, log=log)
            draw(analysis.refresh, base_dir.parent, log=log)
            if arm == "full":
                bench = (root or config.RESULTS_ROOT) / "benchmark"
                bench.mkdir(parents=True, exist_ok=True)
                with open(bench / ".lock", "w") as lock:      # parallel dataset jobs may finish a seed together
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    draw(figures.benchmark_figures, root or config.RESULTS_ROOT, log=log)
    return results
