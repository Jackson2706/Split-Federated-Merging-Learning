"""Gate 1 analysis: ref vs C1 vs C2 vs C1+C2 (CIFAR-100 IID, tau=5). Reads results/G1_*_tau5_s1.

  python analysis/gate1.py [--results results] [--out analysis/out]
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {"ref": "G1_ref_tau5_s1", "C1": "G1_C1_tau5_s1", "C2": "G1_C2_tau5_s1", "C1+C2": "G1_C1C2_tau5_s1"}


def table(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def median(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else None


def load(d):
    m = table(os.path.join(d, "metrics.csv"))
    o = table(os.path.join(d, "oracle.csv"))
    a = table(os.path.join(d, "alpha_log.csv"))
    s = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else {}
    acc = [(int(r["round"]), num(r["accuracy"])) for r in m if num(r["accuracy"]) is not None]
    by_round = {}
    for r in o:
        v = num(r.get("delta_proto"))
        if v is not None:
            by_round.setdefault(int(r["round"]), []).append(v)
    dp = sorted((k, sum(v) / len(v)) for k, v in by_round.items())
    return {"acc": acc, "dp": dp, "o": o, "a": a, "s": s,
            "t_round": mean([num(r["round_time_s"]) for r in m])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis/out")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    R = {k: load(os.path.join(A.results, v)) for k, v in RUNS.items()}
    rows = []
    for k, r in R.items():
        acc = [a for _, a in r["acc"]]
        o, al = r["o"], r["a"]
        alphas_mean = [num(x["alpha_mean"]) for x in al]
        alphas_med = [num(x["alpha_median"]) for x in al]
        rows.append({
            "run": k, "rounds": len(acc),
            "final_val_acc": acc[-1] if acc else None, "best_val_acc": max(acc) if acc else None,
            "mean_delta_proto": mean([v for _, v in r["dp"]]),
            "comp_err_ratio": median([num(x.get("comp_err_ratio")) for x in o]),
            "mix_err_actual": mean([num(x.get("mix_err_actual")) for x in o]),
            "mix_err_fixed": mean([num(x.get("mix_err_fixed")) for x in o]),
            "alpha_median": median(alphas_med),
            "alpha_min": min([x for x in alphas_mean if x is not None], default=None),
            "alpha_max": max([x for x in alphas_mean if x is not None], default=None),
            "peak_reserved_gb": r["s"].get("gpu_peak_reserved_gb"),
            "s_per_round": r["t_round"]})
    cols = list(rows[0])
    with open(os.path.join(A.out, "gate1_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    f = lambda x: "" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f(r[c]) for c in cols))

    for name, key, ylabel in (("delta_proto", "dp", "Delta_proto (mean over learners)"),
                              ("val_acc", "acc", "validation accuracy")):
        plt.figure(figsize=(6, 4))
        for k, r in R.items():
            if r[key]:
                plt.plot(*zip(*r[key]), label=k)
        plt.xlabel("round")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(A.out, f"gate1_{name}.png"), dpi=150)
        plt.close()

    ref, c12 = rows[0], rows[3]
    by = {r["run"]: r for r in rows}
    if None in (ref["mean_delta_proto"], c12["mean_delta_proto"], ref["final_val_acc"], c12["final_val_acc"]):
        print("\nDECISION: incomplete (runs missing)")
        return
    dd = 1 - c12["mean_delta_proto"] / ref["mean_delta_proto"]
    da = 100 * (c12["final_val_acc"] - ref["final_val_acc"])
    print(f"\nC1+C2 vs ref: time-averaged Delta_proto reduction {100 * dd:.1f}% (need >= 30%), "
          f"final val acc {da:+.2f} points (need >= +1.0)")
    if dd >= 0.30 and da >= 1.0:
        print("DECISION: GO")
        return
    c1_ok = by["C1"]["comp_err_ratio"] is not None and by["C1"]["comp_err_ratio"] < 0.7
    c2 = by["C2"]
    c2_ok = c2["mix_err_actual"] is not None and c2["mix_err_fixed"] is not None and c2["mix_err_actual"] < c2["mix_err_fixed"]
    if c1_ok or c2_ok:
        for name, ok in (("C1", c1_ok), ("C2", c2_ok)):
            if ok:
                gain = 100 * ((by[name]["final_val_acc"] or 0) - (ref["final_val_acc"] or 0))
                print(f"DECISION: PARTIAL - {name} passes its own check; final val acc vs ref {gain:+.2f} points")
    else:
        print("DECISION: STOP")


if __name__ == "__main__":
    main()
