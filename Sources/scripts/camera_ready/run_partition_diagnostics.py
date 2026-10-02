#!/usr/bin/env python3
"""
Partition diagnostics for Experiment 1 (hierarchical heterogeneity).

For each (alpha_edge, alpha_client) setting, build the two-level Dirichlet
partition on the dataset labels and save:
  - class histogram per edge / per client,
  - #classes per client, #samples per client,
  - PNG/PDF figures.

No model/training required. Fast.

  python scripts/camera_ready/run_partition_diagnostics.py \
      --cfg configs/camera_ready/smoke/hsfp_smoke.yaml --tag smoke
  python scripts/camera_ready/run_partition_diagnostics.py \
      --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml --tag cifar100
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hsfp_common as H

ROOT = H.ROOT
sys.path.insert(0, ROOT)
from camera_ready.partition import (two_level_dirichlet_partition,
                                    partition_diagnostics, plot_partition,
                                    _extract_labels)
from camera_ready.io_utils import cr_dir, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/camera_ready/smoke/hsfp_smoke.yaml")
    ap.add_argument("--num-edges", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="cifar100")
    ap.add_argument("--settings", nargs="+",
                    default=["1.0,1.0", "1.0,0.1", "0.1,1.0", "0.1,0.1"],
                    help="comma-separated alpha_edge,alpha_client pairs")
    args = ap.parse_args()

    cfg = H.load_config(os.path.join(ROOT, args.cfg) if not os.path.isabs(args.cfg) else args.cfg)
    num_users = int(cfg["num_users"])
    num_edges = args.num_edges or (cfg.get("mid_server", [5])[0])
    num_classes = int(cfg.get("num_classes", 100))

    print(f"[diag] loading labels ({cfg['dataset']}) ...")
    train_ds, _, _, _ = H.build_dataset(cfg)
    labels = _extract_labels(train_ds)

    out_dir = cr_dir("hetero", "partition_diagnostics")
    summary = []
    for s in args.settings:
        ae, ac = (float(v) for v in s.split(","))
        ug, c2e = two_level_dirichlet_partition(
            labels, num_users, num_edges, alpha_edge=ae, alpha_client=ac,
            seed=args.seed, num_classes=num_classes)
        diag = partition_diagnostics(ug, labels, num_classes=num_classes,
                                     client_to_edge=c2e)
        name = f"{args.tag}_ae{ae}_ac{ac}"
        write_json(diag, os.path.join(out_dir, f"diag_{name}.json"))
        plot_partition(diag, os.path.join(out_dir, f"diag_{name}"),
                       title=f"ae={ae}, ac={ac}")
        spc = list(diag["samples_per_client"].values())
        cpc = list(diag["classes_per_client"].values())
        summary.append({
            "alpha_edge": ae, "alpha_client": ac,
            "min_samples": min(spc), "max_samples": max(spc),
            "min_classes": min(cpc), "max_classes": max(cpc),
            "mean_classes": sum(cpc) / len(cpc),
        })
        print(f"  ae={ae} ac={ac}: classes/client in [{min(cpc)},{max(cpc)}] "
              f"samples/client in [{min(spc)},{max(spc)}]")

    write_json(summary, os.path.join(out_dir, f"diag_summary_{args.tag}.json"))
    print(f"[diag] wrote diagnostics + figures to {out_dir}")


if __name__ == "__main__":
    main()
