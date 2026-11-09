"""Launching and monitoring jobs from the notebooks (shared GPU account: the GPU is checked before every launch).

Jobs run detached (`setsid nohup`), so they survive the notebook kernel and the session; their logs are in
results/logs/. Finished stages are skipped, so relaunching the same job resumes it.
"""
import os
import shlex
import subprocess
from datetime import datetime

import pandas as pd

from . import config
from .cv import STAGES, run_dir


def gpu_status():
    """nvidia-smi summary and the compute processes currently on the GPU (empty list = free)."""
    query = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]
    apps = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    try:
        gpu = subprocess.run(query, capture_output=True, text=True, timeout=20).stdout.strip()
        procs = [line for line in subprocess.run(apps, capture_output=True, text=True, timeout=20).stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"nvidia-smi failed: {error}", None
    return gpu, procs


def running_jobs():
    """MorphoNup processes of this workspace (pid, command)."""
    out = subprocess.run(["pgrep", "-af", r"scripts/run_(cv|explain)\.py"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "pgrep" not in line]


def log_path(dataset_id, k, seeds, arm="full", kind="cv"):
    tag = "" if arm == "full" else f"_{arm}"
    return config.RESULTS_ROOT / "logs" / f"{kind}_{dataset_id.replace('/', '_')}_k{k}{tag}_s{'-'.join(map(str, seeds))}.log"


def _check_gpu(allow_busy_gpu):
    gpu, procs = gpu_status()
    if procs is None:
        raise RuntimeError(gpu)
    if procs and not allow_busy_gpu:
        raise RuntimeError("The GPU is in use (shared account). Processes:\n  " + "\n  ".join(procs)
                           + "\nCheck with your labmates, then pass allow_busy_gpu=True.")


def _detach(commands, log):
    """Run argument lists one after another in one detached shell; each command's exit code goes to the log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    steps = "; ".join(f"{' '.join(map(shlex.quote, c))}; echo \"# exit $? : {shlex.quote(' '.join(c[2:]))}\"" for c in commands)
    with open(log, "a") as fh:
        fh.write(f"# launched {datetime.now().isoformat(timespec='seconds')}: {len(commands)} command(s)\n")
    script = f"cd {shlex.quote(str(config.WORKSPACE))} && source ~/miniforge3/bin/activate tf && {{ {steps}; echo '# queue finished'; }}"
    # an empty CUDA_VISIBLE_DEVICES (a CPU-only helper shell) would silently make the job train on the CPU
    env = {k: v for k, v in os.environ.items() if not (k == "CUDA_VISIBLE_DEVICES" and not v.strip())}
    subprocess.run(["bash", "-c", f"setsid nohup bash -c {shlex.quote(script)} >> {shlex.quote(str(log))} 2>&1 < /dev/null &"],
                   check=True, env=env)
    return log


def _cv_args(dataset_id, k, seeds, arm, gpu_mb, stages=None):
    args = ["python", "scripts/run_cv.py", "--dataset", dataset_id, "--k", str(k), "--seeds", *map(str, seeds), "--arm", arm]
    return args + (["--gpu-mb", str(gpu_mb)] if gpu_mb else []) + (["--stages", *stages] if stages else [])


def launch(dataset_id, k, seeds, arm="full", allow_busy_gpu=False, gpu_mb=None):
    """Train one dataset/k/arm detached."""
    return launch_arms(dataset_id, k, seeds, [arm], allow_busy_gpu, gpu_mb)[0]


def launch_arms(dataset_id, k, seeds, arm_list, allow_busy_gpu=False, gpu_mb=None, chains=1):
    """Train arms of one dataset/k detached: `chains` parallel jobs, each running its share of the arms one after another
    (arms dealt round-robin). Different arms write to different folders; the same arm is never started twice."""
    if dataset_id not in config.DATASETS:
        raise KeyError(dataset_id)
    _check_gpu(allow_busy_gpu)
    busy = [j for j in running_jobs() if f"--dataset {dataset_id} " in j + " " and f"--k {k} " in j + " "
            and any(f"--arm {a} " in j + " " for a in arm_list)]
    if busy:
        raise RuntimeError("These arms of this dataset/k are already running:\n  " + "\n  ".join(busy))
    logs = []
    for chain in (arm_list[i::chains] for i in range(min(chains, len(arm_list)))):
        log = log_path(dataset_id, k, seeds, chain[0] if len(chain) == 1 else "arms_" + "+".join(chain))
        logs.append(_detach([_cv_args(dataset_id, k, seeds, a, gpu_mb) for a in chain], log))
    return logs


def arm_logs(dataset_id, k, seeds, arm_list=None):
    """Logs of the analysis-arm jobs of this dataset/k/seeds (single arms and chains), optionally only those that
    train one of `arm_list`."""
    full = log_path(dataset_id, k, seeds)
    head, tail = f"{full.name.split(f'_k{k}_')[0]}_k{k}_", f"_s{'-'.join(map(str, seeds))}.log"
    logs = []
    for p in sorted(full.parent.glob(f"{head}*{tail}")):
        tag = p.name[len(head):-len(tail)]
        if p != full and (arm_list is None or set(tag.removeprefix("arms_").split("+")) & set(arm_list)):
            logs.append(p)
    return logs


def launch_explain(dataset_id, k, seed, folds=None, n_targets=200, n_background=50, steps=32, allow_busy_gpu=False):
    """Explain the saved MorphoNup models of one seed detached (inference only, no training)."""
    _check_gpu(allow_busy_gpu)
    args = ["python", "scripts/run_explain.py", "--dataset", dataset_id, "--k", str(k), "--seed", str(seed),
            "--targets", str(n_targets), "--background", str(n_background), "--steps", str(steps)]
    args += ["--folds", *map(str, folds)] if folds else []
    return _detach([args], log_path(dataset_id, k, [seed], kind="explain"))


def progress(dataset_id, k, seeds, arm="full"):
    """Completed outer folds per seed and stage, and whether the pooled result exists."""
    root = run_dir(dataset_id, k, arm)
    rows = []
    for seed in seeds:
        row = dict(dataset=dataset_id, k=k, arm=arm, seed=seed)
        for stage in STAGES:
            row[f"{stage}_folds_done"] = len(list((root / f"seed_{seed}").glob(f"fold_*/{stage}/fold_summary.json")))
            row[f"{stage}_pooled"] = (root / f"seed_{seed}" / "pooled" / stage / "metrics.json").exists()
        row["status"] = "done" if row["refit_pooled"] else ("started" if row["base_folds_done"] else "not started")
        rows.append(row)
    return pd.DataFrame(rows)


def tail(path, lines=15):
    try:
        text = path.read_text().splitlines()
    except FileNotFoundError:
        return "(no log yet)"
    keep = [t for t in text if not t.startswith(("I0000", "W0000", "E0000", "WARNING: All log messages")) and "cuda" not in t.lower()
            and "tensorflow/core" not in t and "oneDNN" not in t]
    return "\n".join(keep[-lines:])
