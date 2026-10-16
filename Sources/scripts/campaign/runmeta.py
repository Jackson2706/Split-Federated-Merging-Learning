#!/usr/bin/env python3
"""Per-run metadata for the lane scheduler (journal_run.sh::run_exp).

    runmeta.py start    <prov.json> --eid E --cmd "..." --lane L --gpu G
    runmeta.py finish   <prov.json> --rc N --status S
    runmeta.py validate --method M --out-dir D --log F --rounds R   # exit 0 = outputs valid
    runmeta.py classify --log F --rc N                              # prints transient|deterministic

Provenance records the exact command, the code revision plus a fingerprint of uncommitted source
changes, interpreter/library versions, the GPU, and timestamps. Validation is what gates the .done
marker: a zero exit code alone is not proof of a complete, well-formed run.
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git(*a):
    try:
        return subprocess.run(["git", "-C", SRC, *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def source_fingerprint():
    """sha256 over tracked + untracked *.py/*.yaml/*.sh content under Sources/ (excluding results/data)."""
    h = hashlib.sha256()
    files = _git("ls-files", "-co", "--exclude-standard", "--", "*.py", "*.yaml", "*.sh").splitlines()
    for f in sorted(files):
        if f.startswith(("results/", "data/", "outputs/")) or "/exp_" in f:
            continue
        p = os.path.join(SRC, f)
        if os.path.isfile(p):
            h.update(f.encode())
            with open(p, "rb") as fh:
                h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def cmd_start(a):
    try:
        import torch
        tv = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    except Exception as e:
        tv = {"torch": f"unavailable ({e})"}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    prov = {
        "eid": a.eid, "lane": a.lane, "command": a.cmd, "cuda_visible_devices": a.gpu,
        "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": socket.gethostname(),
        "git_head": _git("rev-parse", "HEAD"), "git_dirty_files": _git("status", "--porcelain").count("\n") + 1
        if _git("status", "--porcelain") else 0,
        "source_fingerprint": source_fingerprint(), "python": sys.version.split()[0],
        "platform": platform.platform(), "versions": tv, "gpus": gpu,
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF",
                                               "OUT_ROOT", "ROUNDS", "DIAG") if os.environ.get(k) is not None},
    }
    atomic_json(a.path, prov)


def cmd_finish(a):
    prov = json.load(open(a.path)) if os.path.exists(a.path) else {}
    prov.update(end=time.strftime("%Y-%m-%dT%H:%M:%S%z"), exit_code=a.rc, status=a.status)
    if a.validation:
        prov["validation"] = a.validation
    atomic_json(a.path, prov)


def _rows(csv_path):
    with open(csv_path) as fh:
        return max(0, sum(1 for line in fh if line.strip()) - 1)


def validate(method, out_dir, log, rounds):
    """Return (ok, reason)."""
    if not os.path.isdir(out_dir):
        return False, f"output dir missing: {out_dir}"
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    if re.search(r"Traceback \(most recent call last\)", text):
        return False, "traceback in log"
    if method == "misfp":
        sums = glob.glob(os.path.join(out_dir, "**", "misfp_summary.json"), recursive=True)
        if len(sums) != 1:
            return False, f"expected 1 misfp_summary.json, found {len(sums)}"
        rd = os.path.dirname(sums[0])
        st = json.load(open(os.path.join(rd, "status.json")))
        s = json.load(open(sums[0]))
        if st.get("state") != "completed":
            return False, f"status.json state={st.get('state')}"
        if int(s.get("epochs", -1)) != rounds:
            return False, f"summary epochs {s.get('epochs')} != {rounds}"
        fr = s.get("test_final_round") or {}
        if not all(isinstance(fr.get(k), (int, float)) for k in ("accuracy", "macro_f1")):
            return False, "final-round accuracy/macro_f1 missing"
        hist = os.path.join(rd, "history.csv")
        if os.path.exists(hist) and _rows(hist) < rounds:
            return False, f"history.csv has {_rows(hist)} rows < {rounds}"
        return True, f"misfp final acc={fr['accuracy']:.4f} f1={fr['macro_f1']:.4f}"
    if method == "h-sfp":
        mets = glob.glob(os.path.join(out_dir, "**", "metrics.json"), recursive=True)
        hists = glob.glob(os.path.join(out_dir, "**", "history.csv"), recursive=True)
        preds = glob.glob(os.path.join(out_dir, "**", "last_round_predictions.npz"), recursive=True)
        if not mets or not hists:
            return False, f"metrics.json ({len(mets)}) / history.csv ({len(hists)}) missing"
        n = max(_rows(h) for h in hists)
        if n < rounds:
            return False, f"history.csv has {n} rows < {rounds}"
        if f"Results after {rounds} global rounds" not in text:
            return False, "final results line missing from log"
        if not preds:
            return False, "last_round_predictions.npz missing"
        return True, f"h-sfp {n} history rows, predictions present"
    # other baselines: require the final-results line
    if f"Results after {rounds} global rounds" in text or "Total Run Time" in text:
        return True, "final results line present"
    return False, "no completion line in log"


TRANSIENT = [r"CUDA out of memory", r"OutOfMemoryError", r"CUDA error: unspecified launch failure",
             r"NCCL error", r"Connection reset", r"No space left on device", r"Killed",
             r"cudaErrorLaunchTimeout", r"DataLoader worker .* is killed"]


def classify(log, rc):
    text = open(log, errors="replace").read()[-20000:] if os.path.exists(log) else ""
    if rc in (137, 143, 124) or any(re.search(p, text) for p in TRANSIENT):
        return "transient"
    return "deterministic"


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="op", required=True)
    s = sp.add_parser("start"); s.add_argument("path"); s.add_argument("--eid"); s.add_argument("--cmd")
    s.add_argument("--lane", default=""); s.add_argument("--gpu", default="")
    f = sp.add_parser("finish"); f.add_argument("path"); f.add_argument("--rc", type=int)
    f.add_argument("--status"); f.add_argument("--validation", default="")
    v = sp.add_parser("validate"); v.add_argument("--method"); v.add_argument("--out-dir")
    v.add_argument("--log"); v.add_argument("--rounds", type=int)
    c = sp.add_parser("classify"); c.add_argument("--log"); c.add_argument("--rc", type=int)
    a = ap.parse_args()
    if a.op == "start":
        cmd_start(a)
    elif a.op == "finish":
        cmd_finish(a)
    elif a.op == "validate":
        ok, why = validate(a.method, a.out_dir, a.log, a.rounds)
        print(why)
        sys.exit(0 if ok else 1)
    else:
        print(classify(a.log, a.rc))


if __name__ == "__main__":
    main()
