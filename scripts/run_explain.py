"""Explain the saved MorphoNup models of one seed on their own outer test rows (inference only, no training).

  python scripts/run_explain.py --dataset DatasetNup_1/Homo_sapiens --k 5 --seed 151000
  python scripts/run_explain.py --dataset DatasetNup_1/Homo_sapiens --k 5 --seed 151000 --figures-only

Writes results/<dataset>/k<K>/full/explain/seed_<s>/ (see morphonup/explain.py). Uses the GPU when available
(shared account: check `nvidia-smi` first); finished folds are skipped.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morphonup import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=config.SEEDS[0])
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--stage", choices=["refit", "base"], default="refit")
    ap.add_argument("--targets", type=int, default=200, help="test sequences explained per fold (class-balanced)")
    ap.add_argument("--background", type=int, default=50, help="background sequences from the fold's training rows")
    ap.add_argument("--steps", type=int, default=32, help="integrated-gradients steps")
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from saved explanation files")
    ap.add_argument("--gpu-mb", type=int, default=4096)
    args = ap.parse_args()

    import tensorflow as tf
    from morphonup.cv import run_dir
    from morphonup.data import Dataset
    from morphonup.explain import explain, explain_figures

    arm_dir = run_dir(args.dataset, args.k)
    if args.figures_only:
        print(len(explain_figures(arm_dir / "explain" / f"seed_{args.seed}")), "figures written")
        return
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_mb)])
    ds = Dataset(config.dataset_folder(args.dataset))
    pngs = explain(ds, arm_dir, args.k, args.seed, folds=args.folds, stage=args.stage, n_targets=args.targets,
                   n_background=args.background, steps=args.steps,
                   log=lambda m: print(m, flush=True))
    print(f"{len(pngs)} figures written to {arm_dir / 'explain' / f'seed_{args.seed}' / 'figures'}", flush=True)


if __name__ == "__main__":
    main()
