"""Simple job queue: runs the manifest with K jobs at once on one GPU.

Usage examples
  python run_queue.py --jobs 3                      # everything, priority order
  python run_queue.py --jobs 2 --exp E1             # only E1 (e.g., on your own A6000)
  python run_queue.py --jobs 3 --exp E3,E5,E4,E6,E7 # new-code runs on the rented A6000
  python run_queue.py --dry-run                     # print commands only

A run counts as finished when OUT_ROOT/<run_id>/DONE exists. Your training
script must create that file at the end. Rerunning the queue skips finished
runs, so it is safe to restart after an interruption. Your script should resume
from OUT_ROOT/<run_id>/ckpt if a checkpoint exists.
"""
import argparse, csv, os, re, shlex, subprocess, sys, time

# ---- EDIT THESE TWO ---------------------------------------------------------
CMD_TEMPLATE = ("{python} main.py --task classification --method h-sfp --cfg {cfg} "
                "--seed {seed} --rounds {rounds} --out {out} {partition_args} {overrides}")
# dataset -> paper reference config ("debug" is the CPU test config)
CONFIGS = {"cifar100": "configs/revision/cifar100_reference.yaml",
           "ham10000": "configs/revision/ham10000_reference.yaml",
           "debug": "configs/debug.yaml"}
ROUNDS = {"debug": 3}  # everything else: 200 (paper)
PARTITIONS = {"iid": "",
              "dir0.7": "--set partition=dirichlet --set dirichlet_alpha=0.7",
              "dir0.3": "--set partition=dirichlet --set dirichlet_alpha=0.3"}
# generic manifest key -> your CLI flag
KEY_MAP = {"alpha": "--alpha", "tau": "--max-delay", "p_train": "--p-train",
           "loss": "--packet-loss", "oracle": "--oracle", "sampler": "--sampler",
           "scatter_fix": "--scatter-fix", "lambda_prc": "--lambda-prc",
           "prc": "--prc", "M_M": "--retention", "weighting": "--weighting",
           "accum": "--accum-gamma"}
OUT_ROOT = "results/revision"
# Relative run cost for longest-first ordering (update after benchmarking).
DATASET_COST = {"cifar10": 1.0, "cifar100": 1.0, "ham10000": 0.5, "isic2018": 2.0}
ORACLE_COST = 1.15
# -----------------------------------------------------------------------------

def build_cmd(r, out_root=OUT_ROOT):
    if r["method"] != "ehsfp":
        raise ValueError(f"{r['run_id']}: only E-HSFP runs are in scope (method={r['method']})")
    ov = []
    for kv in r["overrides"].split():
        k, v = kv.split("=", 1)
        if k not in KEY_MAP:
            raise KeyError(f"{r['run_id']}: override key '{k}' has no CLI flag in KEY_MAP")
        ov.append(f"{KEY_MAP[k]} {v}")
    out = os.path.join(out_root, r["run_id"])
    fields = {**r, "out": out, "overrides": " ".join(ov), "python": shlex.quote(sys.executable),
              "cfg": CONFIGS[r["dataset"]], "rounds": ROUNDS.get(r["dataset"], 200),
              "partition_args": PARTITIONS[r["partition"]]}
    return re.sub(" +", " ", CMD_TEMPLATE.format(**fields)).strip(), out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs.csv")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--exp", default="", help="comma list, e.g. E1,E3")
    ap.add_argument("--max-priority", type=int, default=9)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--order", choices=["lpt", "manifest"], default="lpt",
                    help="lpt = within each priority, start the longest runs first "
                         "so the queue does not end waiting on one long run")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest)))
    want = set(a.exp.split(",")) if a.exp else None
    todo = []
    def est(r):
        c = DATASET_COST.get(r["dataset"], 1.0)
        return c * (ORACLE_COST if "oracle=1" in r["overrides"] else 1.0)
    key = ((lambda r: (int(r["priority"]), -est(r))) if a.order == "lpt"
           else (lambda r: int(r["priority"])))
    for r in sorted(rows, key=key):
        if want and r["exp"] not in want: continue
        if int(r["priority"]) > a.max_priority: continue
        if "TODO" in r["overrides"]:
            print(f"[skip] {r['run_id']}: fill TODO in make_manifest.py"); continue
        cmd, out = build_cmd(r, a.out_root)
        if os.path.exists(os.path.join(out, "DONE")): continue
        todo.append((r["run_id"], cmd, out))
    print(f"{len(todo)} runs to do, {a.jobs} at a time")
    if a.dry_run:
        for _, c, _ in todo: print(c)
        return

    running, attempts, failed = {}, {}, []
    os.makedirs(a.out_root, exist_ok=True)
    status = open(os.path.join(a.out_root, "queue_status.log"), "a")
    def log(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line); status.write(line + "\n"); status.flush()

    while todo or running:
        while todo and len(running) < a.jobs:
            rid, cmd, out = todo.pop(0)
            os.makedirs(out, exist_ok=True)
            attempts[rid] = attempts.get(rid, 0) + 1
            f = open(os.path.join(out, f"stdout_try{attempts[rid]}.log"), "w")
            p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
            running[rid] = (p, f, cmd, out, time.time())
            log(f"START {rid} (try {attempts[rid]})")
        time.sleep(5)
        for rid in list(running):
            p, f, cmd, out, t0 = running[rid]
            if p.poll() is None: continue
            f.close(); del running[rid]
            hrs = (time.time() - t0) / 3600
            if p.returncode == 0 and os.path.exists(os.path.join(out, "DONE")):
                log(f"DONE  {rid} {hrs:.2f} h")
            elif attempts[rid] <= a.retries:
                log(f"FAIL  {rid} rc={p.returncode} -> retry"); todo.append((rid, cmd, out))
            else:
                log(f"FAIL  {rid} rc={p.returncode} -> giving up"); failed.append(rid)
    log(f"queue finished; failed: {failed or 'none'}")

if __name__ == "__main__":
    main()
