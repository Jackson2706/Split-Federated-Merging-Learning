#!/usr/bin/env python3
"""Write the machine-readable campaign status, append the event log, refresh RUN_STATE.md's auto block.

    python scripts/campaign/campaign_status.py [--event "text"] [--builds builds.json] [--collector rc]

Outputs (all atomic):
  Sources/results/campaign/status.json     counts, running jobs, failures, resources, ETA, builds
  Sources/results/campaign/events.log      append-only, one line per event (Asia/Seoul timestamps)
  RUN_STATE.md                             block between AUTO-STATUS markers regenerated
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from zoneinfo import ZoneInfo

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = os.path.dirname(SRC)
RES = os.path.join(SRC, "results", "journal_v1")
MARK = os.path.join(RES, ".markers")
CAMP = os.path.join(SRC, "results", "campaign")
KST = ZoneInfo("Asia/Seoul")
DEADLINE = dt.datetime(2026, 10, 20, 23, 59, tzinfo=KST)
# fallback per-run hours on THIS machine until enough local runs finish (overwritten by measurements)
# 2026-10-15 pilots (round-1 s/round): light CIFAR-100 213 s @4-way, ~650 s @16-way; heavy E-HSFP 509 s @4-way;
# HAM E-HSFP 151 s solo / 255 s @8-way. Interpolated to 12 concurrent lanes x 60 rounds:
FALLBACK_H = {"misfp": 8.0, "cifar100_ehsfp": 20.0, "cifar100_hsfp": 8.0, "cifar10": 5.5, "ham10000": 5.5}


def now():
    return dt.datetime.now(KST)


def atomic(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def queue_items():
    items = []
    p = os.path.join(RES, "queue", "master.queue")
    for ln in open(p) if os.path.exists(p) else []:
        if ln.strip() and not ln.startswith("#"):
            F = ln.rstrip("\n").split("|")
            items.append(F)
    return items


def tier_of():
    tiers, cur = {}, None
    p = os.path.join(RES, "queue", "master.queue")
    for ln in open(p) if os.path.exists(p) else []:
        m = re.match(r"# ---- P(\d)", ln)
        if m:
            cur = int(m.group(1))
        elif ln.strip() and not ln.startswith("#"):
            tiers[ln.split("|")[1]] = cur
    return tiers


def cost_class(eid, F=None):
    if eid.startswith("misfp"):
        return "misfp"
    if "cifar100" in eid:
        line = "|".join(F) if F else ""
        heavy = ("ehsfp" in eid or "memory_mode=statistics" in line or "_abl_" in eid and "prc" in eid
                 or eid.startswith(("j3_dropout", "j3_stale", "j3_sv", "j3_memM", "j3_synth")))
        return "cifar100_ehsfp" if heavy else "cifar100_hsfp"
    return "ham10000" if "ham10000" in eid else "cifar10"


def measured_hours():
    """Mean wall hours per cost class from runs finished on this machine (provenance host match)."""
    host = os.uname().nodename
    by = defaultdict(list)
    for p in glob.glob(os.path.join(RES, "provenance", "*.json")):
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if j.get("host") != host or j.get("status") != "PASS" or not j.get("end"):
            continue
        t0 = dt.datetime.strptime(j["start"], "%Y-%m-%dT%H:%M:%S%z")
        t1 = dt.datetime.strptime(j["end"], "%Y-%m-%dT%H:%M:%S%z")
        by[cost_class(j["eid"], j.get("command", "").split())].append((t1 - t0).total_seconds() / 3600)
    return {k: sum(v) / len(v) for k, v in by.items()}, {k: len(v) for k, v in by.items()}


def running():
    out = []
    for lk in glob.glob(os.path.join(MARK, "*.lock")):
        eid = os.path.basename(lk)[:-5]
        try:
            pid = open(os.path.join(lk, "pid")).read().strip()
            started = open(os.path.join(lk, "started")).read().strip()
        except OSError:
            continue
        alive = os.path.exists(f"/proc/{pid}")
        log = os.path.join(RES, "logs", f"{eid}.log")
        ep = re.findall(r"EPOCH (\d+)/(\d+)", open(log, errors="replace").read()) if os.path.exists(log) else []
        out.append({"eid": eid, "pid": pid, "alive": alive, "started": started,
                    "round": f"{ep[-1][0]}/{ep[-1][1]}" if ep else "starting"})
    return out


def resources():
    r = {}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        r["gpus"] = [dict(zip(("index", "util_pct", "mem_used_MiB", "mem_total_MiB"), map(str.strip, l.split(","))))
                     for l in q.strip().splitlines()]
    except Exception as e:
        r["gpus"] = f"nvidia-smi failed: {e}"
    mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if ":" in l}
    r["ram_available_GiB"] = round(mem.get("MemAvailable", 0) / 2 ** 20, 1)
    du = shutil.disk_usage("/workspace")
    r["disk_free_GiB"] = round(du.free / 2 ** 30, 1)
    r["load1"] = os.getloadavg()[0]
    r["pressure"] = [w for w, c in (("disk<5GiB", du.free < 5 * 2 ** 30),
                                    ("ram<32GiB", r["ram_available_GiB"] < 32)) if c]
    return r


def lanes():
    p = os.path.join(RES, "queue", "LANES")
    names = [l.strip() for l in open(p) if l.strip() and not l.startswith("#")] if os.path.exists(p) else []
    alive = []
    for n in names:
        pf = os.path.join(RES, "queue", f"{n}.pid")
        try:
            pid = open(pf).read().strip()
            ok = os.path.exists(f"/proc/{pid}") and "queue_lane.sh" in open(f"/proc/{pid}/cmdline").read()
        except OSError:
            ok = False
        alive.append({"lane": n, "alive": ok})
    return alive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="")
    ap.add_argument("--builds", default="")
    ap.add_argument("--collector", default="")
    ap.add_argument("--next", default="")
    a = ap.parse_args()
    os.makedirs(CAMP, exist_ok=True)
    done = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(MARK, "*.done"))}
    failed = {os.path.basename(p)[:-7] for p in glob.glob(os.path.join(MARK, "*.failed"))}
    q = queue_items()
    tiers = tier_of()
    run = running()
    run_ids = {r["eid"] for r in run if r["alive"]}
    pending = [F for F in q if F[1] not in done and F[1] not in failed]
    paper = lambda e: "MiSFP" if e.startswith("misfp") else "E-HSFP"
    counts = defaultdict(lambda: defaultdict(int))
    for e in done:
        if e.startswith(("misfp", "j3_")):
            counts[paper(e)]["done_validated"] += 1
    for e in failed:
        counts[paper(e)]["failed"] += 1
    for F in pending:
        counts[paper(F[1])]["running" if F[1] in run_ids else "queued"] += 1
        counts[paper(F[1])][f"remaining_P{tiers.get(F[1])}"] += 1
    hours, nmeas = measured_hours()
    n_lanes = max(1, sum(1 for l in lanes() if l["alive"]))
    rem_h = sum(hours.get(cost_class(F[1], F), FALLBACK_H[cost_class(F[1], F)]) for F in pending)
    eta_h = rem_h / n_lanes          # crude: assumes per-run hours measured under the same concurrency
    eta = now() + dt.timedelta(hours=eta_h)
    by_tier = defaultdict(float)
    for F in pending:
        by_tier[tiers.get(F[1])] += hours.get(cost_class(F[1], F), FALLBACK_H[cost_class(F[1], F)]) / n_lanes
    cum, tier_eta = 0.0, {}
    for t in sorted(k for k in by_tier if k is not None):
        cum += by_tier[t]
        tier_eta[f"P{t}"] = (now() + dt.timedelta(hours=cum)).isoformat(timespec="minutes")
    builds = json.load(open(a.builds)) if a.builds and os.path.exists(a.builds) else {}
    prev = json.load(open(os.path.join(CAMP, "status.json"))) if os.path.exists(os.path.join(CAMP, "status.json")) else {}
    st = {
        "updated_kst": now().isoformat(timespec="seconds"),
        "deadline_kst": DEADLINE.isoformat(),
        "counts": {k: dict(v) for k, v in counts.items()},
        "running": run, "lanes": lanes(),
        "failed": sorted(failed),
        "resources": resources(),
        "measured_hours_per_run": {k: round(v, 2) for k, v in hours.items()}, "measured_n": nmeas,
        "remaining_gpu_lane_hours": round(rem_h, 1), "eta_all_kst": eta.isoformat(timespec="minutes"),
        "eta_by_tier_kst": tier_eta,
        "fits_deadline": eta <= DEADLINE,
        "last_collection_kst": now().isoformat(timespec="seconds") if a.collector != "" else prev.get("last_collection_kst"),
        "last_collection_rc": a.collector if a.collector != "" else prev.get("last_collection_rc"),
        "next_collection_kst": a.next or prev.get("next_collection_kst"),
        "builds": builds or prev.get("builds", {}),
    }
    atomic(os.path.join(CAMP, "status.json"), json.dumps(st, indent=2, default=str) + "\n")
    if a.event:
        with open(os.path.join(CAMP, "events.log"), "a") as fh:
            fh.write(f"{now().isoformat(timespec='seconds')} {a.event}\n")
    # RUN_STATE.md auto block
    rs = os.path.join(REPO, "RUN_STATE.md")
    if os.path.exists(rs):
        txt = open(rs).read()
        b, e = "<!-- AUTO-STATUS:BEGIN -->", "<!-- AUTO-STATUS:END -->"
        if b in txt and e in txt:
            c = st["counts"]
            lines = [b, f"_Auto-updated {st['updated_kst']} (KST) by scripts/campaign/campaign_status.py_", "",
                     "| Paper | validated done | running | queued | failed | remaining P1/P2/P3/P4 |", "|---|---|---|---|---|---|"]
            for p in ("E-HSFP", "MiSFP"):
                x = c.get(p, {})
                lines.append(f"| {p} | {x.get('done_validated', 0)} | {x.get('running', 0)} | {x.get('queued', 0)} | "
                             f"{x.get('failed', 0)} | {x.get('remaining_P1', 0)}/{x.get('remaining_P2', 0)}/"
                             f"{x.get('remaining_P3', 0)}/{x.get('remaining_P4', 0)} |")
            lines += ["", f"- Lanes alive: {n_lanes} / {len(st['lanes'])}; running: " +
                      (", ".join(f"`{r['eid']}` ({r['round']})" for r in run if r["alive"]) or "none"),
                      f"- Failed (need review): {', '.join(st['failed']) or 'none'}",
                      f"- Resources: {st['resources'].get('gpus')}; RAM avail {st['resources']['ram_available_GiB']} GiB; "
                      f"disk free {st['resources']['disk_free_GiB']} GiB; pressure: {st['resources']['pressure'] or 'none'}",
                      f"- Measured hours/run on this machine: {st['measured_hours_per_run'] or 'none yet (fallback estimates)'}",
                      f"- ETA all queued work: {st['eta_all_kst']} KST (by tier: {tier_eta}); fits deadline: {st['fits_deadline']}",
                      f"- Last collection: {st['last_collection_kst']} (rc {st['last_collection_rc']}); next: {st['next_collection_kst']}",
                      f"- LaTeX builds: {json.dumps(st['builds'])}", e]
            i, j = txt.index(b), txt.index(e) + len(e)
            atomic(rs, txt[:i] + "\n".join(lines) + txt[j:])
    print(json.dumps({"counts": st["counts"], "eta": st["eta_all_kst"], "fits": st["fits_deadline"]}))


if __name__ == "__main__":
    main()
