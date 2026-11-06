"""Redraw figures and summary tables from saved run files (no training, CPU only).

  python scripts/rebuild_figures.py                                  # every run + the benchmark
  python scripts/rebuild_figures.py --dataset DatasetNup_1/Homo_sapiens --k 5
  python scripts/rebuild_figures.py --levels seed summary            # skip per-fold figures

Pooled metrics are recomputed from the fold predictions first when every fold of a seed is present.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morphonup import config, figures  # noqa: E402

LEVELS = ("fold", "seed", "summary", "benchmark")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=config.RESULTS_ROOT)
    ap.add_argument("--dataset", default=None, help="dataset ID (default: all)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    args = ap.parse_args()
    from morphonup.cv import STAGES, pool

    cap = config.SETTINGS["epochs"]
    count = 0
    for run_config in sorted(args.results.glob("**/run_config.json")):
        arm_dir = run_config.parent
        record = json.loads(run_config.read_text())
        if (args.dataset and record["dataset_id"] != args.dataset) or (args.k and record["k"] != args.k) \
                or (args.arm and record["arm"] != args.arm):
            continue
        for seed_dir in sorted(arm_dir.glob("seed_*")):
            if "fold" in args.levels:
                for fold_dir in sorted(seed_dir.glob("fold_*")):
                    count += len(figures.fold_figures(fold_dir, cap))
            if "seed" in args.levels:
                for stage in STAGES:
                    pool(seed_dir, stage, record["k"])
                count += len(figures.seed_figures(seed_dir, cap))
        if "summary" in args.levels:
            count += len(figures.summary_figures(arm_dir))
        print(f"{record['dataset_id']} k{record['k']} {record['arm']}: figures redrawn", flush=True)
    if "benchmark" in args.levels and not (args.dataset or args.k or args.arm):
        count += len(figures.benchmark_figures(args.results))
    print(f"{count} figures written", flush=True)


if __name__ == "__main__":
    main()
