#!/usr/bin/env python3
"""Human-readable "what changed" report after each collection (user request 2026-10-16: report every update).

Compares the current state with the snapshot from the previous report and writes
  results/campaign/LATEST_UPDATE.md   (this collection only)
  results/campaign/UPDATES.md         (append-only history)
It reports: runs newly validated (with their final metric), new failures, table cells newly filled
(pending markers removed), figures rewritten, progress counts and ETA. Measured values are copied from the
collector outputs, never computed or estimated here.
"""
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import sys
import zoneinfo
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parent
CAMP = SRC / "results" / "campaign"
MK = SRC / "results" / "journal_v1" / ".markers"
SNAP = CAMP / ".update_report_state.json"
PAPERS = {"E-HSFP": REPO / "Submissions" / "E-HSFP Journal", "MiSFP": REPO / "Submissions" / "MiSFP Journal"}
PEND = {"E-HSFP": re.compile(r"\\pending\b"), "MiSFP": re.compile(r"\\MissingResult\b")}
KST = zoneinfo.ZoneInfo("Asia/Seoul")


def state():
    st = {"done": sorted(p.stem for p in MK.glob("*.done")), "failed": sorted(p.stem for p in MK.glob("*.failed")),
          "pending": {}, "figs": {}}
    for k, d in PAPERS.items():
        st["pending"][k] = {os.path.basename(f): len(PEND[k].findall(open(f, errors="replace").read()))
                            for f in sorted(glob.glob(str(d / "generated" / "*.tex")))}
        st["figs"][k] = {os.path.basename(f): hashlib.md5(open(f, "rb").read()).hexdigest()
                         for f in sorted(glob.glob(str(d / "figures" / "data" / "*.csv")))}
    # MiSFP figures have no data CSV and their PDFs embed a timestamp: track the result file they are drawn from
    m = PAPERS["MiSFP"] / "results" / "measurements.csv"
    if m.exists():
        st["figs"]["MiSFP"]["empirical_results + diagnostic_results (from measurements.csv)"] = \
            hashlib.md5(m.read_bytes()).hexdigest()
    return st


def metric(eid):
    if eid.startswith("misfp_"):
        cs = PAPERS["MiSFP"] / "generated" / "collector_status.json"
        for r in json.load(open(cs)).get("per_run", []) if cs.exists() else []:
            if r["eid"] == eid and r.get("accuracy") is not None:
                return f"final acc {r['accuracy']:.2f}%, macro-F1 {r['macro_f1']:.2f}%"
        return ""
    for r in csv.DictReader(open(SRC / "artifacts" / "per_run_metrics.csv")):
        if r["experiment_id"] == eid:
            for k, lab in (("final_acc", "final acc"), ("final_iou", "final IoU")):
                try:
                    return f"{lab} {100 * float(r[k]):.2f}%"
                except (TypeError, ValueError, KeyError):
                    pass
    return ""


def main():
    reason = sys.argv[1] if len(sys.argv) > 1 else "manual"
    now = datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    cur = state()
    prev = json.load(open(SNAP)) if SNAP.exists() else None
    status = json.load(open(CAMP / "status.json")) if (CAMP / "status.json").exists() else {}
    L = [f"## Update {now} (collection: {reason}, rc={status.get('last_collection_rc')})", ""]
    if prev is None:
        L.append("First report: baseline recorded; later reports list changes since this one.")
    new_done = sorted(set(cur["done"]) - set(prev["done"])) if prev else []
    new_fail = sorted(set(cur["failed"]) - set(prev["failed"])) if prev else []
    if new_done:
        L.append(f"**New validated runs ({len(new_done)}):**")
        L += [f"- `{e}`" + (f": {m}" if (m := metric(e)) else "") for e in new_done]
    if new_fail:
        L.append(f"**New FAILED runs ({len(new_fail)}):** " + ", ".join(f"`{e}`" for e in new_fail))
    if prev:
        for k in PAPERS:
            filled = {f: prev["pending"][k].get(f, 0) - n for f, n in cur["pending"][k].items()
                      if prev["pending"][k].get(f, 0) != n}
            if filled:
                tot = sum(cur["pending"][k].values())
                L.append(f"**{k} tables:** " + ", ".join(f"{f} {'+' if v > 0 else ''}{v} cells filled" if v > 0
                                                         else f"{f} {-v} cells re-opened" for f, v in filled.items())
                         + f" ({tot} cells still pending)")
            figs = [f for f, h in cur["figs"][k].items() if prev["figs"][k].get(f) != h]
            if figs:
                L.append(f"**{k} figures updated:** " + ", ".join(
                    f.replace(".csv", "").replace(".pdf", "") for f in figs))
        if not (new_done or new_fail) and not any(
                cur["pending"][k] != prev["pending"][k] or cur["figs"][k] != prev["figs"][k] for k in PAPERS):
            L.append("No new results since the previous report.")
    c = status.get("counts", {})
    L.append("")
    L.append("**Progress:** " + "; ".join(
        f"{k}: {v.get('done_validated', 0)} done, {v.get('running', 0)} running, {v.get('queued', 0)} queued"
        for k, v in c.items()) + f"; failed total {len(cur['failed'])}")
    L.append(f"**ETA (scheduler):** {status.get('eta_by_tier_kst')}; all work {status.get('eta_all_kst')}")
    b = status.get("builds", {})
    L.append("**PDFs:** " + "; ".join(f"{k}: {v.get('pages')} pp, {v.get('latex_errors')} LaTeX errors, "
                                     f"{len(v.get('undefined_references', []))} undefined refs" for k, v in b.items()))
    L.append("")
    txt = "\n".join(L) + "\n"
    (CAMP / "LATEST_UPDATE.md").write_text(txt)
    with open(CAMP / "UPDATES.md", "a") as fh:
        fh.write(txt + "\n")
    SNAP.write_text(json.dumps(cur))
    print(txt)


if __name__ == "__main__":
    main()
