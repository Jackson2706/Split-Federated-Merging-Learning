#!/usr/bin/env python3
"""
Experiment 7: Baseline fairness & reproducibility summary.

Reads configs/camera_ready/baseline_fairness.yaml and emits:
  - results/camera_ready/fairness/baseline_fairness_summary.md  (human-readable)
  - results/camera_ready/fairness/baseline_fairness.tex         (LaTeX appendix)

  python scripts/camera_ready/gen_fairness.py
"""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from camera_ready.io_utils import cr_dir
from camera_ready.latex import write_table


def flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, list):
        out.append((prefix, ", ".join(str(x) for x in obj)))
    else:
        out.append((prefix, str(obj)))


def main():
    cfg_path = os.path.join(ROOT, "configs", "camera_ready", "baseline_fairness.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    rows = []
    flatten("", cfg, rows)

    out_dir = cr_dir("fairness")

    # Markdown summary
    md = ["# Baseline Fairness & Reproducibility Summary", "",
          "Common protocol applied to **all** methods (H-SFP and baselines) for the",
          "CIFAR-100 camera-ready experiments. Source: "
          "`configs/camera_ready/baseline_fairness.yaml`.", "",
          "| Hyperparameter | Value |", "|---|---|"]
    for k, v in rows:
        md.append(f"| `{k}` | {v} |")
    md += ["",
           "## Sanity checks",
           "- **Centralized oracle** (upper bound) and **tuned FedAvg** commands are in "
           "`results/camera_ready/README.md`.",
           "",
           "## Notes",
           "- All methods share dataset, client/edge counts, sampling ratio, local/global "
           "schedule, optimizer, batch size, and seeds.",
           "- Communication is accounted with identical formulas across methods "
           "(see `comm_accounting`).",
           ""]
    md_path = os.path.join(out_dir, "baseline_fairness_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))

    # LaTeX appendix table
    latex_rows = [{"param": k.replace("_", " ").replace(".", " / "),
                   "value": str(v).replace("_", "\\_")} for k, v in rows]
    write_table(
        latex_rows,
        columns=[("param", "Hyperparameter"), ("value", "Value")],
        path=os.path.join(out_dir, "baseline_fairness.tex"),
        caption="Shared experimental protocol for all methods (CIFAR-100).",
        label="tab:fairness", escape=False, column_spec="ll",
    )
    print(f"[fairness] wrote {md_path}")
    print(f"[fairness] wrote {os.path.join(out_dir, 'baseline_fairness.tex')}")


if __name__ == "__main__":
    main()
