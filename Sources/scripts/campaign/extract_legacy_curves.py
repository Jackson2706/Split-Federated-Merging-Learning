#!/usr/bin/env python3
"""Freeze the old host's per-round figure curves (2026-10-16).

The raw run directories of the 33 E-HSFP runs completed on the legacy RTX 3080 Ti host were not migrated,
but the figure data written from them by figures/make_figures.py was committed (commit 5ae3035,
`Submissions/E-HSFP Journal/figures/data/*.csv`: per-round mean, sample SD and n for every plotted series).
For a series with n=1 the values ARE that run's per-round curve; for n=3 they are the mean/SD over seeds 0-2.

This script reads those CSVs from git (never from the working tree, which make_figures.py overwrites), maps
every series back to its run-ID prefix with the same specs make_figures.py uses, verifies each accuracy
series against the legacy ledger (final-round value == ledger final_acc, or its seed mean for n>1), and
writes results/journal_v1/legacy/figure_curves_oldmachine.json. Any mismatch aborts without writing.
Values are stored unscaled (accuracy as a fraction) so make_figures.py applies its usual scaling.
"""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parent
COMMIT = "5ae3035"
DATA = "Submissions/E-HSFP Journal/figures/data"
OUT = SRC / "results" / "journal_v1" / "legacy" / "figure_curves_oldmachine.json"
LEDGER = SRC / "results" / "journal_v1" / "legacy" / "per_run_metrics_oldmachine_20261015.csv"
J = "j3"


def rows(name):
    txt = subprocess.run(["git", "-C", str(REPO), "show", f"{COMMIT}:{DATA}/{name}.csv"],
                         check=True, capture_output=True, text=True).stdout
    return list(csv.DictReader(io.StringIO(txt)))


def blocks(rs):
    """Split rows into consecutive per-series blocks (round restarts at 1)."""
    out = []
    for x in rs:
        if not out or out[-1]["key"] != (x["figure"], x["series"]) or int(x["round"]) == 1:
            out.append({"key": (x["figure"], x["series"]), "n": int(x["n"]), "rows": []})
        out[-1]["rows"].append(x)
    return out


def entry(b, prefix, what, figure):
    r = sorted(b["rows"], key=lambda x: int(x["round"]))
    assert [int(x["round"]) for x in r] == list(range(1, len(r) + 1)), (prefix, "rounds not contiguous")
    sc = 100.0 if what == "acc" else 1.0
    return {"prefix": prefix, "what": what, "n": b["n"], "figure": figure, "series": b["key"][1],
            "mean": [float(x["mean"]) / sc for x in r], "sd": [float(x["sd"]) / sc for x in r]}


def main():
    E = []
    # fig_conv_curves: panels cifar10, cifar100, ham10000 in this order; per panel the series that existed
    # are written in spec order (H-SFP/E-HSFP x IID/dir0.7/dir0.3). Map label -> prefix within each panel.
    lab2tag = {"IID": "", "dir(0.7)": "noniid_a0p7_", "dir(0.3)": "noniid_a0p3_"}
    panel, seen = 0, set()
    ds = ("cifar10", "cifar100", "ham10000")
    for b in blocks(rows("fig_conv_curves")):
        ml, pl = b["key"][1].split(" ", 1)
        if b["key"][1] in seen:          # label repeats -> next panel
            panel += 1; seen = set()
        seen.add(b["key"][1])
        m = "hsfp" if ml == "H-SFP" else "ehsfp"
        E.append(entry(b, f"{J}_{lab2tag[pl]}{ds[panel]}_{m}", "acc", "fig_conv_curves"))
    # fig_drift_rounds
    d = {"No event loss": "sv_cifar100_none", "Combined stress": "sv_cifar100_combined",
         r"$\tau$=5": "stale_cifar100_tau5", "p=0.5": "dropout_cifar100_p0p5"}
    for b in blocks(rows("fig_drift_rounds")):
        E.append(entry(b, f"{J}_{d[b['key'][1]]}", "drift", "fig_drift_rounds"))
    # dropout / staleness / memory (acc + drift)
    for name, f in (("fig_dropout_effect", lambda s: f"{J}_cifar100_ehsfp" if s == "p=0.2"
                     else f"{J}_dropout_cifar100_p{s[2:].replace('.', 'p')}"),
                    ("fig_staleness_effect", lambda s: f"{J}_cifar100_ehsfp" if s == r"$\tau$=0"
                     else f"{J}_stale_cifar100_tau{s.split('=')[1]}"),
                    ("fig_memory_effect", lambda s: f"{J}_memM_cifar100_M{s.split('=')[1]}")):
        for b in blocks(rows(name)):
            what = b["key"][0].split(":")[1]
            E.append(entry(b, f(b["key"][1]), what, name))

    # verify every accuracy series against the ledger
    L = {r["experiment_id"]: r for r in csv.DictReader(open(LEDGER))}
    bad = []
    for e in E:
        if e["what"] != "acc":
            continue
        fa = [float(L[f"{e['prefix']}_s{s}"]["final_acc"]) for s in range(e["n"])
              if f"{e['prefix']}_s{s}" in L and L[f"{e['prefix']}_s{s}"]["final_acc"]]
        if len(fa) != e["n"] or abs(sum(fa) / len(fa) - e["mean"][-1]) > 5e-5:
            bad.append((e["figure"], e["prefix"], e["n"], fa, e["mean"][-1]))
    if bad:
        print("VERIFY FAILED:", *bad, sep="\n  "); sys.exit(1)
    # drift-only series must belong to legacy-completed runs
    for e in E:
        for s in range(e["n"]):
            assert f"{e['prefix']}_s{s}" in L, (e["prefix"], s, "not a legacy run")
    # the same (prefix, what) appearing in several figures must carry identical data
    by = {}
    for e in E:
        k = (e["prefix"], e["what"])
        if k in by:
            assert by[k]["n"] == e["n"] and max(abs(a - b) for a, b in zip(by[k]["mean"], e["mean"])) < 1e-9, k
        by[k] = e
    out = {"source": f"git {COMMIT}:{DATA}/*.csv (written by make_figures.py from the legacy-host run dirs)",
           "verified": "accuracy series final round == legacy ledger final_acc (seed mean for n>1)",
           "curves": sorted(by.values(), key=lambda e: (e["prefix"], e["what"]))}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(SRC)}: {len(by)} curves "
          f"({sum(e['what'] == 'acc' for e in by.values())} acc verified, "
          f"{sum(e['what'] == 'drift' for e in by.values())} drift)")


if __name__ == "__main__":
    main()
