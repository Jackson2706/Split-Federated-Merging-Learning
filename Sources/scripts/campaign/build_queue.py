#!/usr/bin/env python3
"""Build the single global priority queue drained by every lane (both papers).

    python scripts/campaign/build_queue.py            # writes results/journal_v1/queue/master.queue
    python scripts/campaign/build_queue.py --print    # show tiers/counts only

Queue line: stage|eid|task|dataset|method|cfg|seed|ablation|extra...  (see queue_lane.sh).
Runs already .done are omitted. Order inside the file is the execution priority:

  P1  missing main results      MiSFP Table main (H-SFP/K1/Adaptive x IID/Dir(0.1) x seeds 1-3);
                                E-HSFP seed-0 pending cells (non-IID, scalability, M=20, aggr memory+stale)
  P2  baselines / repetitions   MiSFP ablation variants (Edge/K2/K4); MiSFP Dir(0.7)
  P3  ablations / sensitivity   MiSFP budget sweep + shared-snapshot control on Dir(0.1);
                                E-HSFP figure seeds 1-2 (all sweep families incl. non-IID)
  P4  additional                MiSFP budget/snapshot on IID and Dir(0.7); min-support / boundary-sample sweep;
                                E-HSFP HAM non-IID seeds 3-4 (used by no table/figure)

E-HSFP lines are copied verbatim from the protocol-v3 queues of the previous machine (exact same
overrides), so new runs stay comparable with the 55 completed j3 runs.
MiSFP follows scripts/misfp/experiments.py (the authors' registered matrix) with the manuscript's
registered seeds {1,2,3} (experiments.py used 0,1,2; results/manifest.json and the paper say 1,2,3).
"""
import argparse
import os
import re

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
Q = os.path.join(SRC, "results", "journal_v1", "queue")
MARK = os.path.join(SRC, "results", "journal_v1", ".markers")
CFG = "configs/classification/misfp"
MISFP_SEEDS = (1, 2, 3)
SPLITS = {"iid": [], "dir0.1": ["partition=dirichlet", "dirichlet_alpha=0.1"],
          "dir0.7": ["partition=dirichlet", "dirichlet_alpha=0.7"]}


def done(eid):
    return os.path.exists(os.path.join(MARK, f"{eid}.done"))


def mline(eid, variant, seed, split, sets=(), cfgsuffix=""):
    cfg = f"{CFG}/cifar100_misfp_{variant}{cfgsuffix}_resnet50_5_10.yaml"
    extra = []
    # misfp_diag_pca_classes: diagnostic-only (held-out features saved for every class so the fidelity
    # table can be computed over all 100 classes instead of the default 4); never feeds training.
    for s in [*SPLITS[split], *sets, "misfp_diag_pca_classes=100"]:
        extra += ["--set", s]
    return "|".join(["misfp", eid, "classification", "cifar100", "misfp", cfg, str(seed), "", *extra])


def misfp_runs():
    """Yield (tier, line)."""
    out = []
    for split in ("iid", "dir0.1", "dir0.7"):
        for seed in MISFP_SEEDS:
            for v in ("hsfp", "k1", "edge", "k2", "k4", "adaptive"):
                main = v in ("hsfp", "k1", "adaptive")
                tier = (1 if main else 2) if split != "dir0.7" else 2
                out.append((tier, mline(f"misfp_{split}_{v}_s{seed}", v, seed, split)))
            sens_tier = 3 if split == "dir0.1" else 4
            for v in ("k2", "adaptive"):
                out.append((sens_tier, mline(f"misfp_{split}_{v}_fp16matched_s{seed}", v, seed, split,
                                             cfgsuffix="_fp16_matched")))
                for r in ("1.5", "2.0"):
                    out.append((sens_tier, mline(f"misfp_{split}_{v}_budget{r}x_s{seed}", v, seed, split,
                                                 [f"misfp_client_budget=match_k1_fp32:{r}",
                                                  f"misfp_edge_budget=match_k1_fp32:{r}"])))
            for v in ("k1", "edge", "k2"):
                out.append((sens_tier, mline(f"misfp_{split}_{v}_snapshot_s{seed}", v, seed, split,
                                             ["misfp_extraction=shared_snapshot"])))
    # diagnostics sweep (fig:diagnostics, tab:sensitivity rows 'min support' / 'boundary samples');
    # values are a protocol choice recorded in RUN_STATE.md for author confirmation
    for seed in MISFP_SEEDS:
        for ms in (1, 10):
            out.append((4, mline(f"misfp_dir0.1_adaptive_minsup{ms}_s{seed}", "adaptive", seed, "dir0.1",
                                 [f"misfp_min_component_support={ms}"])))
        for bs in (10, 200):
            out.append((4, mline(f"misfp_dir0.1_adaptive_bsamp{bs}_s{seed}", "adaptive", seed, "dir0.1",
                                 [f"misfp_boundary_samples_per_component={bs}"])))
    return out


def ehsfp_runs():
    lines = []
    for f in ("j3_cifar.queue", "ham_hold_v3.queue"):
        for ln in open(os.path.join(Q, f)):
            ln = ln.rstrip("\n")
            if ln and not ln.startswith("#"):
                lines.append(ln)
    seen, out = set(), []
    for ln in lines:
        eid = ln.split("|")[1]
        if eid in seen:
            continue
        seen.add(eid)
        seed = int(re.search(r"_s(\d+)$", eid).group(1))
        # sweep-table cells use seed 0 (fill_manuscript.py); figures use seeds 0-2 (make_figures.py);
        # seeds >= 3 of sweep runs feed neither (kept, lowest priority)
        tier = 1 if seed == 0 else (3 if seed <= 2 else 4)
        out.append((tier, ln))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    runs = [(t, l) for t, l in ehsfp_runs() + misfp_runs() if not done(l.split("|")[1])]
    # SCOPE (user decision 2026-10-15, cost $0.5/h): P1 + P2 + the P3 runs that feed MiSFP tab:sensitivity /
    # tab:fidelity (incl. the dir0.1 min-support / boundary-sample rows). Descoped: E-HSFP figure seeds 1-2
    # and all other P4. Descoped runs are listed in results/campaign/descoped.txt, never filled in.
    def in_scope(t, l):
        eid = l.split("|")[1]
        if t in (1, 2):
            return True
        if eid.startswith("misfp_dir0.1_"):
            return True
        if FIG_LIGHT.match(eid):
            return True
        return False
    # 2026-10-16 user: "focus on complete figure". Re-admit the cheap (non-CIFAR-100-E-HSFP) seeds 1-2 that
    # fig_conv_curves needs; the CIFAR-100 E-HSFP sweeps (~33-47 h each) cannot finish by the deadline.
    FIG_LIGHT = re.compile(r"^j3_noniid_a0p[37]_(?:(?:cifar10|ham10000)_(?:hsfp|ehsfp)|cifar100_hsfp)_s[12]$")
    descoped = [l.split("|")[1] for t, l in runs if not in_scope(t, l)]
    runs = [(4 if FIG_LIGHT.match(l.split("|")[1]) else 1 if t == 1 else 2 if t == 2 else 3, l) for t, l in runs if in_scope(t, l)]
    camp = os.path.join(SRC, "results", "campaign")
    os.makedirs(camp, exist_ok=True)
    with open(os.path.join(camp, "descoped.txt"), "w") as fh:
        fh.write("# Descoped 2026-10-15 by user decision (cost); not run, shown as pending/descoped\n"
                 + "\n".join(sorted(descoped)) + "\n")
    print("descoped", len(descoped))
    # within a tier: interleave cheap and expensive runs is left to lanes; keep seed-major order
    def key(tl):
        t, l = tl
        eid = l.split("|")[1]
        seed = int(re.search(r"_s(\d+)$", eid).group(1))
        return (t, seed if not eid.startswith("misfp") else seed - 1, eid.startswith("misfp"))
    runs.sort(key=key)
    # interleave the two papers inside each tier (round-robin) so neither paper's cells starve
    inter = []
    for t in sorted({t for t, _ in runs}):
        e = [x for x in runs if x[0] == t and not x[1].startswith("misfp|")]
        m = [x for x in runs if x[0] == t and x[1].startswith("misfp|")]
        for i in range(max(len(e), len(m))):
            inter += ([m[i]] if i < len(m) else []) + ([e[i]] if i < len(e) else [])
    runs = inter
    counts = {}
    for t, l in runs:
        k = (t, "misfp" if l.startswith("misfp|") else "ehsfp")
        counts[k] = counts.get(k, 0) + 1
    for k in sorted(counts):
        print(f"P{k[0]} {k[1]:6s} {counts[k]}")
    print("total", len(runs))
    if a.print:
        return
    body = ["# master.queue -- generated by scripts/campaign/build_queue.py; edit the generator, not this file.",
            "# Priority = file order. P1 main results, P2 baselines/repetitions, P3 ablation/sensitivity, P4 extra."]
    cur = None
    for t, l in runs:
        if t != cur:
            body.append(f"# ---- P{t}")
            cur = t
        body.append(l)
    tmp = os.path.join(Q, f".master.queue.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write("\n".join(body) + "\n")
    os.replace(tmp, os.path.join(Q, "master.queue"))
    if runs:  # new/pending work re-arms the "final collection after the last job" hook
        try:
            os.remove(os.path.join(MARK, "FINAL_COLLECTED"))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
