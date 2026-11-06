"""Train and evaluate the final MorphoNup model with outer k-fold CV on one dataset.

  python scripts/run_cv.py --dataset DatasetNup_1/Homo_sapiens --k 5 --seeds 151000
  python scripts/run_cv.py --dataset DatasetNup_2/Homo_sapiens/LC --k 10 --seeds 151000 --folds 1 2
  python scripts/run_cv.py --dataset DatasetNup_1/Homo_sapiens --k 5 --arm no_kmer      # an analysis arm

Long runs: check `nvidia-smi` first (shared account), then launch detached, e.g.
  setsid nohup python scripts/run_cv.py ... > results/logs/<name>.log 2>&1 &
Completed stages are skipped, so rerunning the same command resumes a killed job.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morphonup import arms, config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    ap.add_argument("--k", type=int, default=5, help="outer folds (5 or 10)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[config.SEEDS[0]])
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="subset of folds (default: all)")
    ap.add_argument("--arm", default="full", choices=list(arms.ARMS), help="full = MorphoNup; others: analysis arms")
    ap.add_argument("--smoke", action="store_true", help="2-epoch pipeline check into a separate --results root")
    ap.add_argument("--gpu-mb", type=int, default=None, help="per-process GPU memory cap (default by dataset size)")
    ap.add_argument("--results", type=Path, default=config.RESULTS_ROOT)
    args = ap.parse_args()

    import tensorflow as tf
    from morphonup.cv import run
    from morphonup.data import Dataset

    ds = Dataset(config.dataset_folder(args.dataset))
    # Fixed per-process cap so parallel workers cannot starve each other (memory growth never releases).
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.set_logical_device_configuration(
            gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_mb or config.gpu_memory_mb(len(ds.y)))])
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    run(ds, args.dataset, args.k, args.seeds, folds=args.folds, arm=args.arm, root=args.results, smoke=args.smoke,
        log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
