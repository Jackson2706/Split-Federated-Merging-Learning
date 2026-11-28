#!/usr/bin/env python3
"""One row per configuration: mean +- std over seeds (sample std) and n seeds.

Fails if any configuration has fewer than --expected seeds (use --allow-partial
while runs are still in progress). Writes analysis/out/aggregate.{csv,tex}.
"""
import argparse

from common import EXPECTED_SEEDS, by_config, check_seeds, load_runs, mean_std, num, pm, write_table

METRICS = [("test_accuracy", 100), ("test_macro_f1", 100), ("final_accuracy", 100),
           ("relative_rounds", 1), ("selected_ece", 100), ("selected_nll", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/revision")
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--expected", type=int, default=EXPECTED_SEEDS)
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    groups = by_config(load_runs(a.root))
    short = check_seeds(groups, a.expected, a.allow_partial)
    rows = []
    for cid, runs in sorted(groups.items()):
        row = [cid, len(runs), " ".join(str(r["seed"]) for r in runs)]
        for key, scale in METRICS:
            m, sd, _ = mean_std([num(r["summary"].get(key)) * scale for r in runs])
            row.append(pm(m, sd, 0 if key == "relative_rounds" else 2))
        rows.append(row)
    header = ["configuration", "n", "seeds", "Acc (%)", "F1 (%)", "final Acc (%)", "Conv.", "ECE (%)", "NLL"]
    write_table(f"{a.out}/aggregate", header, rows)
    print(f"{len(rows)} configurations -> {a.out}/aggregate.csv/.tex"
          + (f"; PARTIAL: {short}" if short else ""))


if __name__ == "__main__":
    main()
