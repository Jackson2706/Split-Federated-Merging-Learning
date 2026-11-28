"""Job queue for E-HSFP runs on one (shared) GPU. Standard library only.

  python run_queue.py --manifest gates.csv --dry-run
  python run_queue.py --manifest gates.csv --exp G1 --jobs 2 --device 0 --min-free-gb 5.5

Manifest columns: run_id,exp,priority,dataset,partition,method,seed,rounds,overrides
(overrides = space-separated key=value, mapped to CLI flags through KEY_MAP;
``rounds`` may be absent -> ROUNDS / 200).

A run is finished when <out-root>/<run_id>/DONE exists and is skipped on restart.
An unfinished run is started again with the same --out, so it resumes from its
checkpoint. Each attempt's stdout goes to <out-root>/<run_id>/stdout_try<N>.log;
START / DONE / FAIL lines (with timestamps and hours) go to <out-root>/queue_status.log.
"""
import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time

# ---- the two things to edit --------------------------------------------------
CMD_TEMPLATE = ("{python} main.py --task classification --method h-sfp --cfg {cfg} "
                "--seed {seed} --rounds {rounds} --out {out} --gpu-data 1 --set num_workers=2 "
                "{partition_args} {overrides}")
# dataset -> config. cifar100 loads the locked H-SFP setup (Step 1 of the C1/C2 check).
CONFIGS = {"cifar100": "configs/revision/hsfp_setup_cifar100.yaml",
           "ham10000": "configs/revision/ham10000_reference.yaml",
           "debug": "configs/debug.yaml"}
ROUNDS = {"debug": 3}  # default when the manifest has no rounds column: 200 (paper)
PARTITIONS = {"iid": "",
              "dir0.7": "--set partition=dirichlet --set dirichlet_alpha=0.7",
              "dir0.3": "--set partition=dirichlet --set dirichlet_alpha=0.3"}
KEY_MAP = {"alpha": "--alpha", "tau": "--max-delay", "p_train": "--p-train",
           "loss": "--packet-loss", "oracle": "--oracle", "oracle_every": "--oracle-every",
           "sampler": "--sampler", "scatter_fix": "--scatter-fix", "lambda_prc": "--lambda-prc",
           "prc": "--prc", "M_M": "--retention", "weighting": "--weighting",
           "drift_comp": "--drift-comp", "dc_window": "--dc-window",
           "dc_src_bonus": "--dc-src-bonus", "dc_min_weight": "--dc-min-weight",
           "alpha_mode": "--alpha-mode", "alpha_min": "--alpha-min"}
OUT_ROOT = "results"
# -------------------------------------------------------------------------------

LAUNCH_SETTLE_S = 90  # wait after each launch before reading free memory again


def build_cmd(r, out_root=OUT_ROOT, template=None, python=None):
    if r["method"] != "ehsfp":
        raise ValueError(f"{r['run_id']}: only E-HSFP runs are in scope (method={r['method']})")
    ov = []
    for kv in r["overrides"].split():
        k, v = kv.split("=", 1)
        if k not in KEY_MAP:
            raise KeyError(f"{r['run_id']}: override key '{k}' has no CLI flag in KEY_MAP")
        ov.append(f"{KEY_MAP[k]} {v}")
    out = os.path.join(out_root, r["run_id"])
    rounds = (r.get("rounds") or "").strip() or ROUNDS.get(r["dataset"], 200)
    fields = {**r, "out": out, "overrides": " ".join(ov), "python": python or shlex.quote(sys.executable),
              "cfg": CONFIGS[r["dataset"]], "rounds": rounds, "partition_args": PARTITIONS[r["partition"]]}
    return re.sub(" +", " ", (template or CMD_TEMPLATE).format(**fields)).strip(), out


def free_gb(device):
    """Free GPU memory in GB from nvidia-smi (None if it cannot be read)."""
    try:
        o = subprocess.check_output(["nvidia-smi", "-i", str(device), "--query-gpu=memory.free",
                                     "--format=csv,noheader,nounits"], text=True, timeout=30)
        return float(o.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs.csv")
    ap.add_argument("--exp", default="", help="comma list, e.g. G1,G0")
    ap.add_argument("--jobs", type=int, default=1, help="runs at once")
    ap.add_argument("--device", default="0", help="sets CUDA_VISIBLE_DEVICES for the children")
    ap.add_argument("--min-free-gb", type=float, default=0.0,
                    help="launch only when nvidia-smi reports at least this much free memory")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--max-priority", type=int, default=9)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--cmd-template", default=None, help="override CMD_TEMPLATE (tests)")
    ap.add_argument("--settle", type=float, default=LAUNCH_SETTLE_S)
    ap.add_argument("--poll", type=float, default=5.0)
    a = ap.parse_args(argv)

    rows = list(csv.DictReader(open(a.manifest)))
    want = set(a.exp.split(",")) if a.exp else None
    todo = []
    for r in sorted(rows, key=lambda r: int(r["priority"])):  # stable: manifest order within a priority
        if want and r["exp"] not in want:
            continue
        if int(r["priority"]) > a.max_priority:
            continue
        cmd, out = build_cmd(r, a.out_root, a.cmd_template)
        if os.path.exists(os.path.join(out, "DONE")):
            continue
        todo.append((r["run_id"], cmd, out))
    print(f"{len(todo)} runs to do, {a.jobs} at a time")
    if a.dry_run:
        for _, c, _ in todo:
            print(c)
        return 0

    os.makedirs(a.out_root, exist_ok=True)
    status = open(os.path.join(a.out_root, "queue_status.log"), "a")

    def log(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        status.write(line + "\n")
        status.flush()

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a.device),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    running, attempts, failed = {}, {}, []
    next_launch_ok = 0.0
    while todo or running:
        while todo and len(running) < a.jobs and time.time() >= next_launch_ok:
            if a.min_free_gb > 0:
                free = free_gb(a.device)
                if free is None or free < a.min_free_gb:
                    break  # wait; re-read after the poll sleep
            rid, cmd, out = todo.pop(0)
            os.makedirs(out, exist_ok=True)
            attempts[rid] = attempts.get(rid, 0) + 1
            f = open(os.path.join(out, f"stdout_try{attempts[rid]}.log"), "w")
            p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
            running[rid] = (p, f, cmd, out, time.time())
            log(f"START {rid} (try {attempts[rid]})")
            next_launch_ok = time.time() + a.settle
        time.sleep(a.poll)
        for rid in list(running):
            p, f, cmd, out, t0 = running[rid]
            if p.poll() is None:
                continue
            f.close()
            del running[rid]
            hrs = (time.time() - t0) / 3600
            if p.returncode == 0 and os.path.exists(os.path.join(out, "DONE")):
                log(f"DONE  {rid} {hrs:.2f} h")
            elif attempts[rid] <= a.retries:
                log(f"FAIL  {rid} rc={p.returncode} {hrs:.2f} h -> retry")
                todo.insert(0, (rid, cmd, out))
            else:
                log(f"FAIL  {rid} rc={p.returncode} {hrs:.2f} h -> giving up")
                failed.append(rid)
    log(f"queue finished; failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
