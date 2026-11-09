"""Run a queue of MorphoNup trainings with a fixed number of parallel jobs (GPU sharing within one card).

The queue file has one task per line: dataset_id <TAB> k <TAB> seeds (comma-separated) [<TAB> gpu_mb [<TAB> stages]];
'#' starts a comment; stages = base, refit or base,refit (default both). The file is re-read before every start, so tasks can be added, removed or reordered while the queue runs.
A task is skipped when all its folds are finished (run_cv skips finished stages anyway). Before starting a job the
runner waits while a process that is not a MorphoNup job uses the GPU (shared lab account).
  source ~/miniforge3/bin/activate tf
  setsid nohup python scripts/run_queue.py results/logs/benchmark_queue.tsv --parallel 2 >> results/logs/queue.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from morphonup import config, jobs  # noqa: E402


def say(msg):
    print(f"# {datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def tasks(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        out.append(dict(dataset=parts[0], k=int(parts[1]), seeds=[int(s) for s in parts[2].split(",")],
                        gpu_mb=int(parts[3]) if len(parts) > 3 and parts[3] not in ("", "-") else None,
                        stages=parts[4].split(",") if len(parts) > 4 else ["base", "refit"]))
    return out


def key(t):
    return f"{t['dataset']} k{t['k']} seeds {','.join(map(str, t['seeds']))} stages {','.join(t['stages'])}"


def finished(t):
    d = config.RESULTS_ROOT / t["dataset"] / f"k{t['k']}" / "full"
    last = t["stages"][-1]
    return all(len(list((d / f"seed_{s}").glob(f"fold_*/{last}/fold_summary.json"))) == t["k"] for s in t["seeds"])


def foreign_gpu_processes():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    foreign = []
    for pid in out.split():
        cmd = subprocess.run(["ps", "-o", "args=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if "scripts/run_cv.py" not in cmd and "scripts/run_explain.py" not in cmd:
            foreign.append(f"{pid} {cmd[:80]}")
    return foreign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args()
    running, done = {}, set()
    say(f"queue {args.queue}, {args.parallel} parallel job(s)")
    while True:
        for k_, (proc, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.write(f"# exit {proc.returncode} : {k_}\n"); fh.close()
                say(f"finished (exit {proc.returncode}): {k_}")
                done.add(k_); running.pop(k_)
        active = {k_.split(" seeds ")[0] for k_ in running}          # one job per dataset/k folder at a time
        todo = [t for t in tasks(args.queue) if key(t) not in running and key(t) not in done and not finished(t)
                and key(t).split(" seeds ")[0] not in active]
        if not todo and not running:
            say("queue finished"); return
        if todo and len(running) < args.parallel:
            busy = foreign_gpu_processes()
            if busy:
                say("GPU used by another process, waiting: " + "; ".join(busy)); time.sleep(300); continue
            t = todo[0]
            cmd = jobs._cv_args(t["dataset"], t["k"], t["seeds"], "full", t["gpu_mb"], t["stages"])
            log = jobs.log_path(t["dataset"], t["k"], t["seeds"])
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log, "a")
            fh.write(f"# launched {datetime.now().isoformat(timespec='seconds')} by run_queue: {' '.join(cmd)}\n"); fh.flush()
            env = {k_: v for k_, v in os.environ.items() if not (k_ == "CUDA_VISIBLE_DEVICES" and not v.strip())}
            running[key(t)] = (subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env), fh)
            say(f"started: {key(t)}")
            time.sleep(30)
            continue
        time.sleep(60)


if __name__ == "__main__":
    main()
