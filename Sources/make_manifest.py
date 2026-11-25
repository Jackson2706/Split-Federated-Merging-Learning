"""Generate runs.csv: every new run for the E-HSFP revision (84 runs, 3 seeds; E-HSFP experiments only).
Edit the CONSTANTS below, then: python make_manifest.py
Override keys are generic names; map them to your training script's flags
in run_queue.py (CMD_TEMPLATE / KEY_MAP)."""
import csv

SEEDS = [0, 1, 2]          # 3 seeds for all new experiments (report as 'three seeds')
E1_SEEDS = [0, 1, 2]       # set to [0, 1, 2, 3, 4] to pair with all 5 existing E-HSFP seeds
# Set True ONLY if you do not have per-seed E-HSFP results (seeds 0-2) for the
# 9 dataset/partition settings: adds E-HSFP reference runs to pair with E1.
E1_PAIR_EHSFP = False
# H-SFP reruns are out of scope (only E-HSFP experiments and paper TODOs are run).
INCLUDE_E1 = False
# Settings behind Table IV's "under source masking" / "under staleness" columns
# (still TODO in the paper - fill in before running E7).
MASK_P = 0.5       # Table IV "under source masking"
STALE_TAU = 10     # Table IV "under staleness"

rows = []
def add(exp, prio, dataset, partition, method, seed, oracle=False, **ov):
    ov = dict(ov)
    if oracle:
        ov["oracle"] = 1
    tag = "_".join(f"{k}{v}" for k, v in ov.items()) or "ref"
    run_id = f"{exp}_{method}_{dataset}_{partition}_{tag}_s{seed}".replace(".", "p")
    rows.append(dict(run_id=run_id, exp=exp, priority=prio, dataset=dataset,
                     partition=partition, method=method, seed=seed,
                     overrides=" ".join(f"{k}={v}" for k, v in ov.items())))

# E1 matched H-SFP (out of scope by default; set INCLUDE_E1 = True to add)
for ds in (["cifar100", "ham10000", "cifar10"] if INCLUDE_E1 else []):
    for part in ["iid", "dir0.7", "dir0.3"]:
        for s in E1_SEEDS:
            add("E1", 1, ds, part, "hsfp", s)

if E1_PAIR_EHSFP:
    for ds in ["cifar100", "ham10000", "cifar10"]:
        for part in ["iid", "dir0.7", "dir0.3"]:
            if ds == "cifar100" and part == "iid":
                continue  # covered by E3 alpha=0.3, tau=0 (reference config)
            for s in E1_SEEDS:
                add("E1", 1, ds, part, "ehsfp", s)

# E3 alpha sweep with drift oracle (must) -> also gives E2 diagnostics
for a in [0.1, 0.3, 0.5, 0.7, 1.0]:
    for s in SEEDS:
        add("E3", 1, "cifar100", "iid", "ehsfp", s, oracle=True, alpha=a, tau=0)
for a in [0.1, 0.3, 0.5, 0.7, 1.0]:
    for s in SEEDS:
        add("E3", 1, "cifar100", "iid", "ehsfp", s, oracle=True, alpha=a, tau=5)

# E5 severe heterogeneity, pass 1 (must)
for ds in ["cifar100", "ham10000"]:
    for var in [dict(), dict(sampler="mixture"), dict(scatter_fix=1),
                dict(sampler="mixture", scatter_fix=1)]:
        for s in SEEDS:
            add("E5", 1, ds, "dir0.3", "ehsfp", s, **var)

# E4 masking under real packet loss (should)
for p in [0.0, 0.2, 0.5]:
    for s in SEEDS:
        add("E4", 2, "cifar100", "iid", "ehsfp", s, oracle=True, p_train=p, loss=0.3)

# E6 PRC sensitivity + corrected PRC (should)
for lam in [0.0, 0.05, 0.5]:
    for s in SEEDS:
        add("E6", 2, "cifar100", "iid", "ehsfp", s, lambda_prc=lam)
for s in SEEDS:
    add("E6", 2, "cifar100", "iid", "ehsfp", s, lambda_prc=0.1, prc="corrected")

# E7 precision weighting, Table IV(b) row (should)
for cond in [dict(), dict(p_train=MASK_P), dict(tau=STALE_TAU)]:
    for s in SEEDS:
        add("E7", 2, "cifar100", "iid", "ehsfp", s, M_M=0, weighting="precision", **cond)

with open("runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

from collections import Counter
print("total", len(rows), dict(Counter(r["exp"] for r in rows)))
assert len({r["run_id"] for r in rows}) == len(rows), "duplicate run ids"
