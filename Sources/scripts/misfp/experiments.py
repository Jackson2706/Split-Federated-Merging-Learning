"""MiSFP experiment manifest + sequential launcher (resumable).

    python scripts/misfp/experiments.py status  --matrix pilot
    python scripts/misfp/experiments.py run     --matrix pilot            # sequential
    python scripts/misfp/experiments.py commands --matrix full            # print, do not run
    python scripts/misfp/experiments.py run     --matrix full --only <id> # launch one

States: pending | running | completed | failed. A run is "completed" only if its
run_dir/status.json says so and misfp_summary.json exists. Re-running ``run``
skips completed runs and retries failed/stale ones with a new attempt id
(``--set misfp_attempt=N`` changes the resolved-config hash, so the runner never
overwrites a previous run directory).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CFG = "configs/classification/misfp"
OUT_ROOT = os.environ.get("MISFP_OUT_ROOT", "/media/jackson/Data/misfp_runs")
MANIFEST_DIR = os.path.join(ROOT, "results", "misfp")

PILOT_EPOCHS = 10
FULL_EPOCHS = 60  # matches the journal campaign's CIFAR-100 budget
SEEDS = (0, 1, 2)
NONIID = {"iid": [], "dir0.1": ["partition=dirichlet", "dirichlet_alpha=0.1"],
          "dir0.7": ["partition=dirichlet", "dirichlet_alpha=0.7"]}


def _cfg(variant, extra=""):
    return f"{CFG}/cifar100_misfp_{variant}{extra}_resnet50_5_10.yaml"


def pilot_runs():
    runs = []
    for split, variants in (("iid", ("hsfp", "k1", "edge")), ("dir0.1", ("hsfp", "k1", "k2"))):
        for v in variants:
            runs.append({"id": f"pilot_{split}_{v}_s0", "cfg": _cfg(v), "seed": 0,
                         "sets": [f"epochs={PILOT_EPOCHS}", *NONIID[split]], "group": "pilot"})
    return runs


def full_runs():
    runs = []
    for split in NONIID:
        for seed in SEEDS:
            for v in ("hsfp", "k1", "edge", "k2", "k4", "adaptive"):
                runs.append({"id": f"full_{split}_{v}_s{seed}", "cfg": _cfg(v), "seed": seed,
                             "sets": [f"epochs={FULL_EPOCHS}", *NONIID[split]], "group": "natural_cost"})
            for v in ("k2", "adaptive"):  # matched per-packet byte budget vs K1 fp32
                runs.append({"id": f"full_{split}_{v}_fp16matched_s{seed}",
                             "cfg": _cfg(v, "_fp16_matched"), "seed": seed,
                             "sets": [f"epochs={FULL_EPOCHS}", *NONIID[split]], "group": "matched_budget"})
                for ratio in ("1.5", "2.0"):
                    runs.append({"id": f"full_{split}_{v}_budget{ratio}x_s{seed}", "cfg": _cfg(v), "seed": seed,
                                 "sets": [f"epochs={FULL_EPOCHS}", *NONIID[split],
                                          f"misfp_client_budget=match_k1_fp32:{ratio}",
                                          f"misfp_edge_budget=match_k1_fp32:{ratio}"],
                                 "group": "byte_budget_sweep"})
            for v in ("k1", "edge", "k2"):  # shared-snapshot extraction control
                runs.append({"id": f"full_{split}_{v}_snapshot_s{seed}", "cfg": _cfg(v), "seed": seed,
                             "sets": [f"epochs={FULL_EPOCHS}", *NONIID[split],
                                      "misfp_extraction=shared_snapshot"], "group": "shared_encoder"})
    return runs


MATRICES = {"pilot": pilot_runs, "full": full_runs}


def _manifest_path(matrix):
    return os.path.join(MANIFEST_DIR, f"manifest_{matrix}.json")


def _load(matrix):
    p = _manifest_path(matrix)
    state = json.load(open(p)) if os.path.exists(p) else {}
    for r in MATRICES[matrix]():
        state.setdefault(r["id"], {"state": "pending", "attempt": 0})
        state[r["id"]]["spec"] = r
    return state


def _save(matrix, state):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    tmp = _manifest_path(matrix) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    os.replace(tmp, _manifest_path(matrix))


def _refresh(entry):
    rd = entry.get("run_dir")
    if not rd:
        return
    st = os.path.join(rd, "status.json")
    if os.path.exists(st):
        s = json.load(open(st))["state"]
        if s == "completed" and os.path.exists(os.path.join(rd, "misfp_summary.json")):
            entry["state"] = "completed"
        elif s == "failed":
            entry["state"] = "failed"
        elif s == "running" and entry.get("pid") and not os.path.exists(f"/proc/{entry['pid']}"):
            entry["state"] = "failed"
            entry["error"] = "process vanished (stale running status)"


def command(entry, matrix):
    r = entry["spec"]
    out = os.path.join(OUT_ROOT, matrix, r["id"])
    sets = [*r["sets"], f"output_dir={out}"]
    if entry.get("attempt", 0) > 0:
        sets.append(f"misfp_attempt={entry['attempt']}")
    cmd = [sys.executable, "main.py", "--task", "classification", "--method", "misfp",
           "--cfg", r["cfg"], "--seed", str(r["seed"])]
    for s in sets:
        cmd += ["--set", s]
    return cmd, out


def do_run(matrix, only=None):
    state = _load(matrix)
    for rid, entry in state.items():
        if only and rid not in only:
            continue
        _refresh(entry)
        if entry["state"] == "completed":
            continue
        if entry["state"] in ("failed", "running"):
            entry["attempt"] = entry.get("attempt", 0) + 1
        cmd, out = command(entry, matrix)
        os.makedirs(out, exist_ok=True)
        log = os.path.join(out, f"attempt{entry.get('attempt', 0)}.log")
        entry.update(state="running", started=time.time(), log=log, command=" ".join(cmd),
                     run_dir=None, pid=None, error=None)
        _save(matrix, state)
        print(f"[misfp] launching {rid}: {' '.join(cmd)}", flush=True)
        with open(log, "w") as fh:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            entry["pid"] = proc.pid
            _save(matrix, state)
            rc = proc.wait()
        m = re.search(r"\[MiSFP\] run_dir=(\S+)", open(log).read())
        entry["run_dir"] = m.group(1) if m else None
        entry["returncode"] = rc
        entry["finished"] = time.time()
        _refresh(entry)
        if rc != 0 and entry["state"] != "completed":
            entry["state"] = "failed"
            entry["error"] = f"exit code {rc}; see {log}"
        _save(matrix, state)
        print(f"[misfp] {rid}: {entry['state']} ({(entry['finished'] - entry['started']) / 60:.1f} min)", flush=True)


def do_status(matrix):
    state = _load(matrix)
    counts = {}
    for rid, e in state.items():
        _refresh(e)
        counts[e["state"]] = counts.get(e["state"], 0) + 1
        print(f"{e['state']:10s} {rid:45s} {e.get('run_dir') or ''}")
    _save(matrix, state)
    print(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("run", "status", "commands"))
    ap.add_argument("--matrix", choices=sorted(MATRICES), default="pilot")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.action == "run":
        do_run(a.matrix, a.only)
    elif a.action == "status":
        do_status(a.matrix)
    else:
        state = _load(a.matrix)
        for rid, e in state.items():
            cmd, _ = command(e, a.matrix)
            print(f"# {rid} [{e['spec']['group']}]\n{' '.join(cmd)}")
        print(f"# total runs: {len(state)}")


if __name__ == "__main__":
    main()
