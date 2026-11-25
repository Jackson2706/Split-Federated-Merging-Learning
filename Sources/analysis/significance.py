#!/usr/bin/env python3
"""Significance tests for Section VII.

1. Paired t-tests (per seed) of each new configuration against its experiment's
   reference (E3, E4, E5 per dataset, E6), with 95% CI and Holm correction per
   experiment. n = number of shared seeds (three).
2. Welch tests from the reported mean +- std of the existing Table IV contrasts
   (five runs each), and of the new E7 precision row (three seeds) against the
   reported Table IV(b) rows.
Writes analysis/out/paired.{csv,tex} and analysis/out/welch.{csv,tex}.
"""
import argparse
import math

from scipy import stats

from common import (E5_REFERENCE, REFERENCE, by_config, load_runs, mean_std, num, write_table)

# Reported Table IV values (mean, std; five runs). Source: sections/experiments.tex.
TABLE_IV = {
    "full": {"reference": (56.07, 0.89), "masking": (55.13, 1.07), "staleness": (53.47, 1.37)},
    "learned_cfg": {"reference": (55.36, 0.93), "masking": (54.63, 1.07), "staleness": (52.89, 1.31)},
    "learned_M0": {"reference": (55.36, 0.93), "masking": (54.63, 1.07), "staleness": (52.89, 1.31)},
    "count_M0": {"reference": (55.34, 0.99), "masking": (53.47, 1.23), "staleness": (51.71, 1.47)},
}
E7 = {"reference": "E7_ehsfp_cifar100_iid_M_M0_weightingprecision",
      "masking": "E7_ehsfp_cifar100_iid_M_M0_weightingprecision_p_train0p5",
      "staleness": "E7_ehsfp_cifar100_iid_M_M0_weightingprecision_tau10"}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def acc(run):
    return 100 * num(run["summary"].get("test_accuracy"))


def paired(groups):
    out = []
    for exp in ("E3", "E4", "E5", "E6"):
        tests = []
        for cid, runs in sorted(groups.items()):
            if not cid.startswith(exp + "_"):
                continue
            ref_id = (E5_REFERENCE.format(dataset=runs[0]["config"]["dataset"]) if exp == "E5"
                      else REFERENCE[exp])
            if cid == ref_id or ref_id not in groups:
                continue
            ref = {r["seed"]: acc(r) for r in groups[ref_id]}
            d = [acc(r) - ref[r["seed"]] for r in runs if r["seed"] in ref]
            if len(d) < 2:
                continue
            m, sd, n = mean_std(d)
            t = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
            p = stats.ttest_rel([acc(r) for r in runs if r["seed"] in ref],
                                [ref[r["seed"]] for r in runs if r["seed"] in ref]).pvalue
            tests.append([exp, cid, ref_id, n, m, m - t, m + t, p])
        for row, p_adj in zip(tests, holm([r[-1] for r in tests])):
            out.append(row + [p_adj])
    return out


def welch_rows(groups):
    rows = []
    contrasts = [("Full vs learned-weighting configuration", "full", "learned_cfg")]
    for label, a, b in contrasts:
        for cond in ("reference", "masking", "staleness"):
            (m1, s1), (m2, s2) = TABLE_IV[a][cond], TABLE_IV[b][cond]
            r = stats.ttest_ind_from_stats(m1, s1, 5, m2, s2, 5, equal_var=False)
            rows.append([label, cond, 5, 5, m1 - m2, r.statistic, r.pvalue, "reported"])
    for cond in ("reference", "masking", "staleness"):
        (m1, s1), (m2, s2) = TABLE_IV["learned_M0"][cond], TABLE_IV["count_M0"][cond]
        r = stats.ttest_ind_from_stats(m1, s1, 5, m2, s2, 5, equal_var=False)
        rows.append(["Learned vs sample-count weighting (M_M=0)", cond, 5, 5, m1 - m2, r.statistic, r.pvalue,
                     "reported"])
    for cond, cid in E7.items():
        if cid not in groups or len(groups[cid]) < 2:
            continue
        m1, s1, n1 = mean_std([acc(r) for r in groups[cid]])
        for other, label in (("learned_M0", "learned"), ("count_M0", "sample-count")):
            m2, s2 = TABLE_IV[other][cond]
            r = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, 5, equal_var=False)
            rows.append([f"Precision (new, {n1} seeds) vs {label} (M_M=0)", cond, n1, 5, m1 - m2,
                         r.statistic, r.pvalue, "new vs reported"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/revision")
    ap.add_argument("--out", default="analysis/out")
    a = ap.parse_args()
    groups = by_config(load_runs(a.root))
    f = lambda x: f"{x:.2f}"
    pr = paired(groups)
    write_table(f"{a.out}/paired", ["exp", "configuration", "reference", "n", "mean diff (pp)", "CI low",
                                    "CI high", "p", "p (Holm)"],
                [[e, c, r, n, f(m), f(lo), f(hi), f(p), f(ph)] for e, c, r, n, m, lo, hi, p, ph in pr])
    wr = welch_rows(groups)
    write_table(f"{a.out}/welch", ["contrast", "condition", "n1", "n2", "diff (pp)", "t", "p", "source"],
                [[c, k, n1, n2, f(d), f(t), f(p), s] for c, k, n1, n2, d, t, p, s in wr])
    print(f"{len(pr)} paired tests, {len(wr)} Welch tests -> {a.out}/paired.*, {a.out}/welch.*")


if __name__ == "__main__":
    main()
