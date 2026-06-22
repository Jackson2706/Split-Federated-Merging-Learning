#!/usr/bin/env python3
"""
Generate LaTeX tables for the camera-ready experiments from collected summaries.

  - hetero    : results/camera_ready/hetero/summary_hetero.csv   -> table_hetero.tex
  - partial   : results/camera_ready/partial/summary_partial.csv -> table_partial.tex
  - profiling : profiling/profile_*_summary.json + wallclock.csv -> table_profiling.tex

(Covariance / Lstat / inversion scripts emit their own .tex directly.)

  python scripts/camera_ready/make_tables.py [--only hetero|partial|profiling|all]
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from camera_ready.io_utils import CR_ROOT, read_csv, fmt_mean_std
from camera_ready.latex import write_table

METHOD_ORDER = [("federated", "FedAvg"), ("splitfl", "SplitFed"),
                ("hsfl", "HSFL"), ("hierfl", "HierFL"), ("h-sfp", "H-SFP")]


def _cell(rows, pred):
    for r in rows:
        if pred(r):
            return fmt_mean_std(r.get("test_f1_mean", ""), r.get("test_f1_std", ""),
                                pct=True, prec=2)
    return "--"


def table_hetero():
    path = os.path.join(CR_ROOT, "hetero", "summary_hetero.csv")
    if not os.path.isfile(path):
        print(f"[make_tables] missing {path}; run collect first."); return
    rows = read_csv(path)
    settings = [("1.0", "1.0"), ("1.0", "0.1"), ("0.1", "1.0"), ("0.1", "0.1")]
    cols = [("setting", "Setting"), ("ae", "$\\alpha_{e}$"), ("ac", "$\\alpha_{c}$")]
    method_cols = [(mk, lbl) for mk, lbl in METHOD_ORDER
                   if any(r["method"] == mk for r in rows)]
    cols += [(mk, lbl) for mk, lbl in method_cols]

    out = []
    for i, (ae, ac) in enumerate(settings, 1):
        row = {"setting": f"S{i}", "ae": ae, "ac": ac}
        for mk, _ in method_cols:
            row[mk] = _cell(rows, lambda r, mk=mk, ae=ae, ac=ac:
                            r["method"] == mk and r["alpha_edge"] == ae
                            and r["alpha_client"] == ac)
        out.append(row)
    write_table(out, cols, os.path.join(CR_ROOT, "hetero", "table_hetero.tex"),
                caption="Hierarchical heterogeneity: test F1 (\\%) under two-level "
                        "Dirichlet ($\\alpha_e$=inter-edge, $\\alpha_c$=intra-edge), "
                        "CIFAR-100, mean$\\pm$std over seeds.",
                label="tab:hetero", escape=False)
    print("[make_tables] wrote table_hetero.tex")


def table_partial():
    path = os.path.join(CR_ROOT, "partial", "summary_partial.csv")
    if not os.path.isfile(path):
        print(f"[make_tables] missing {path}; run collect first."); return
    rows = read_csv(path)
    fracs = sorted({r["frac"] for r in rows}, key=float)
    dists = sorted({r["dist"] for r in rows})
    method_cols = [(mk, lbl) for mk, lbl in METHOD_ORDER
                   if any(r["method"] == mk for r in rows)]
    cols = [("frac", "Active \\%"), ("dist", "Distribution")] + method_cols

    out = []
    for dist in dists:
        for frac in fracs:
            row = {"frac": f"{int(float(frac)*100)}", "dist": dist}
            for mk, _ in method_cols:
                row[mk] = _cell(rows, lambda r, mk=mk, frac=frac, dist=dist:
                                r["method"] == mk and r["frac"] == frac
                                and r["dist"] == dist)
            out.append(row)
    write_table(out, cols, os.path.join(CR_ROOT, "partial", "table_partial.tex"),
                caption="Partial participation: test F1 (\\%) vs active-client ratio and "
                        "data distribution, CIFAR-100, mean$\\pm$std over seeds.",
                label="tab:partial", escape=False)
    print("[make_tables] wrote table_partial.tex")


def table_profiling():
    pdir = os.path.join(CR_ROOT, "profiling")
    summaries = glob.glob(os.path.join(pdir, "profile_*_summary.json"))
    if not summaries:
        print(f"[make_tables] no profiling summaries in {pdir}; run run_profiling.sh."); return

    # Wall-clock totals per method.
    wall = {}
    wpath = os.path.join(pdir, "wallclock.csv")
    if os.path.isfile(wpath):
        for r in read_csv(wpath):
            wall.setdefault(r["method"], []).append(float(r["wall_seconds"]))
    wmean = {m: sum(v) / len(v) for m, v in wall.items()}

    rows = []
    for spath in sorted(summaries):
        with open(spath) as f:
            summ = json.load(f)["summary"]
        tag = os.path.basename(spath)[len("profile_"):-len("_summary.json")]

        def mean_s(phase):
            return f"{summ[phase]['mean_s']:.4f}" if phase in summ else "--"
        rows.append({
            "dataset": tag,
            "pack": mean_s("client_pack"),
            "edge": mean_s("edge_process"),
            "cloud": mean_s("cloud_process"),
            "edge_agg": mean_s("edge_aggregate"),
            "hsfp_total": f"{wmean.get('h-sfp', 0):.1f}" if wmean.get("h-sfp") else "--",
            "split_total": f"{wmean.get('splitfl', 0):.1f}" if wmean.get("splitfl") else "--",
        })
    write_table(
        rows,
        columns=[("dataset", "Dataset"), ("pack", "pack/client/rnd (s)"),
                 ("edge_agg", "aggregate/edge/rnd (s)"), ("edge", "synth+train/edge/rnd (s)"),
                 ("hsfp_total", "H-SFP total (s)"), ("split_total", "SplitFed total (s)")],
        path=os.path.join(pdir, "table_profiling.tex"),
        caption="Runtime profiling of H-SFP packing/aggregation/synthesis phases "
                "(CUDA-synchronized) vs SplitFed total time.",
        label="tab:profiling", escape=False)
    print("[make_tables] wrote table_profiling.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["hetero", "partial", "profiling", "all"],
                    default="all")
    args = ap.parse_args()
    if args.only in ("hetero", "all"):
        table_hetero()
    if args.only in ("partial", "all"):
        table_partial()
    if args.only in ("profiling", "all"):
        table_profiling()


if __name__ == "__main__":
    main()
