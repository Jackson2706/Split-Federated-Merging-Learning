"""Build results/misfp/PILOT_REPORT.md (+ CSV, PCA figure) from COMPLETED runs only.

    python scripts/misfp/report.py --matrix pilot
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "misfp")


def _load_rounds(rd):
    p = os.path.join(rd, "misfp_rounds.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def _hist_times(rd):
    p = os.path.join(rd, "history.csv")
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return [float(r["elapsed_time_s"]) for r in csv.DictReader(fh)]


def collect(matrix):
    man = json.load(open(os.path.join(OUT, f"manifest_{matrix}.json")))
    rows, pending = [], []
    for rid, e in sorted(man.items()):
        rd = e.get("run_dir")
        sp = os.path.join(rd, "misfp_summary.json") if rd else None
        if e.get("state") != "completed" or not sp or not os.path.exists(sp):
            pending.append((rid, e.get("state")))
            continue
        s = json.load(open(sp))
        rounds = _load_rounds(rd)
        last = rounds[-1] if rounds else {}
        disp = [d["mean_ratio"] for r in rounds for d in r.get("dispersion", []) if d.get("mean_ratio") is not None]
        fits = [r["client_fit_summary"] for r in rounds]
        edge_out = [e2["output"]["mean_components_per_class"] for r in rounds for e2 in r.get("edge", [])]
        hd = os.path.join(rd, "misfp_heldout_diagnostic.json")
        hd = json.load(open(hd)) if os.path.exists(hd) else {}
        times = _hist_times(rd)
        fb = {}
        for f in fits:
            for k, v in f.get("fallbacks", {}).items():
                fb[k] = fb.get(k, 0) + v
        n_fits = sum(f["class_fits"] for f in fits) or None
        split = rid.split("_")[1]
        rows.append({
            "id": rid, "split": split, "label": s["label"], "variant": s["variant"], "seed": s["seed"],
            "epochs": s["epochs"],
            "final_acc": s["test_final_round"]["accuracy"], "final_f1": s["test_final_round"]["macro_f1"],
            "final_worst10": s["test_final_round"]["worst_decile_class_accuracy"],
            "best_acc": (s.get("test_best_val_checkpoint") or {}).get("accuracy"),
            "best_round": (s.get("test_best_val_checkpoint") or {}).get("selected_round"),
            "val_acc_last": (s.get("validation_accuracy") or [None])[-1],
            "c2e_payload_B": s["prototype_payload_bytes"]["client_to_edge"],
            "e2c_payload_B": s["prototype_payload_bytes"]["edge_to_cloud"],
            "c2e_route_MB": s["communication_MB"]["client_to_edge_MB"],
            "e2c_route_MB": s["communication_MB"]["edge_to_cloud_MB"],
            "total_comm_MB": s["communication_MB"]["total_comm_MB"],
            "wall_h": s["wall_time_s"] / 3600, "mean_round_s": float(np.mean(times)) if times else None,
            "peak_cuda_GB": s["peak_cuda_allocated_bytes"] / 1e9,
            "mean_client_k": float(np.mean([f["mean_k"] for f in fits if f["mean_k"]])) if fits else None,
            "client_k1_fallback_frac": (fb.get("insufficient_support", 0) / n_fits) if n_fits else None,
            "edge_out_comps_per_class": float(np.mean(edge_out)) if edge_out else None,
            "dispersion_ratio": float(np.mean(disp)) if disp else None,
            "heldout_nll_mix": hd.get("mean_nll_mixture_per_dim"),
            "heldout_nll_single": hd.get("mean_nll_single_per_dim"),
            "time_local_fit_s": (last.get("totals") or {}).get("time_local_fit_s"),
            "time_edge_reestimate_s": (last.get("totals") or {}).get("time_edge_reestimate_s"),
            "syn_edge": (last.get("totals") or {}).get("synthetic_samples_edge"),
            "syn_cloud": (last.get("totals") or {}).get("synthetic_samples_cloud"),
            "infeasible": (last.get("totals") or {}).get("infeasible_budget_events"),
            "run_dir": rd,
        })
    return rows, pending


def _fmt(v, p=2, pct=False):
    if v is None:
        return "–"
    if pct:
        return f"{100 * v:.{p}f}"
    if isinstance(v, float):
        return f"{v:.{p}f}"
    return str(v)


def pca_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from misfp import ClassMixture, make_generator, sample_mixture
    cand = [r for r in rows if r["variant"] != "hsfp" and os.path.exists(os.path.join(r["run_dir"], "misfp_pca_payload.npz"))]
    if not cand:
        return None
    fig, axes = plt.subplots(len(cand), 3, figsize=(11, 3.2 * len(cand)), squeeze=False, constrained_layout=True)
    for i, r in enumerate(cand):
        z = np.load(os.path.join(r["run_dir"], "misfp_pca_payload.npz"))
        cls = sorted({k.split("_")[0] for k in z.files})[0]
        real = z[f"{cls}_real"].astype(np.float64)
        cm = ClassMixture(int(cls[1:]), z[f"{cls}_counts"], z[f"{cls}_means"], z[f"{cls}_vars"])
        mix, _ = sample_mixture(cm, 600, make_generator("pca", r["id"]))
        one, _ = sample_mixture(cm.collapse(), 600, make_generator("pca1", r["id"]))
        mu = real.mean(0)
        _, _, vt = np.linalg.svd(real - mu, full_matrices=False)  # ONE projection, fit on real
        P = vt[:2].T
        for j, (name, X) in enumerate((("held-out real (val)", real), (f"pooled mixture ({cm.R} comps)", mix.numpy()),
                                        ("moment-collapsed single", one.numpy()))):
            Y = (X - mu) @ P
            ax = axes[i, j]
            ax.scatter(Y[:, 0], Y[:, 1], s=5, alpha=0.4, color="#2a78d6", lw=0)
            ax.set_title(f"{r['label']} [{r['split']}] class {cls[1:]}: {name}", fontsize=8, loc="left")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7)
        lim = np.percentile(np.abs((real - mu) @ P), 99) * 1.6
        for ax in axes[i]:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
    fig.suptitle("Client->edge space, final round; PCA fitted on held-out real features, same projection per row",
                 fontsize=9, x=0.01, ha="left")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="pilot")
    a = ap.parse_args()
    rows, pending = collect(a.matrix)
    os.makedirs(OUT, exist_ok=True)
    if rows:
        with open(os.path.join(OUT, f"{a.matrix}_results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    L = [f"# MiSFP {a.matrix} report (exploratory; generated from completed runs only)\n"]
    L.append("CIFAR-100, ResNet split (client 128-d / edge 256-d), 200 clients, frac 0.1, 5 edges, t1=5/t2=10, "
             "validation = 10% held out from train (`val_from_train=0.1`), single seed. "
             "Headline = **final-round** test metrics through the composed model (no checkpoint selection). "
             "Single-seed differences are NOT significance-tested; metric std across seeds is unavailable.\n")
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        ref = {r["variant"]: r for r in rs}
        L.append(f"\n## Split: {split}\n")
        L.append("| Run | final acc % | Δ vs A | Δ vs B | macro-F1 % | worst-10% class acc % | best-val ckpt acc % (round) | "
                 "c→e payload KB | e→c payload KB | total comm MB | mean round s | peak CUDA GB |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            dA = (r["final_acc"] - ref["hsfp"]["final_acc"]) if "hsfp" in ref else None
            dB = (r["final_acc"] - ref["k1"]["final_acc"]) if "k1" in ref else None
            L.append(f"| {r['label']} | {_fmt(r['final_acc'], pct=True)} | {_fmt(dA, pct=True)} | {_fmt(dB, pct=True)} | "
                     f"{_fmt(r['final_f1'], pct=True)} | {_fmt(r['final_worst10'], pct=True)} | "
                     f"{_fmt(r['best_acc'], pct=True)} ({r['best_round']}) | "
                     f"{_fmt(r['c2e_payload_B'] / 1024 if r['c2e_payload_B'] else None, 0)} | "
                     f"{_fmt(r['e2c_payload_B'] / 1024 if r['e2c_payload_B'] else None, 0)} | "
                     f"{_fmt(r['total_comm_MB'], 1)} | {_fmt(r['mean_round_s'], 0)} | {_fmt(r['peak_cuda_GB'], 2)} |")
        L.append("\n| Run | mean client K | client K1 fallback frac | edge→cloud comps/class | between/within dispersion | "
                 "held-out NLL/dim mixture | held-out NLL/dim single | local fit s (cum) | edge re-est s (cum) | syn samples edge/cloud | infeasible budgets |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            if r["variant"] == "hsfp":
                L.append("| H-SFP | (1) | – | (1) | – | – | – | – | – | – | – |")
                continue
            L.append(f"| {r['label']} | {_fmt(r['mean_client_k'], 3)} | {_fmt(r['client_k1_fallback_frac'], 3)} | "
                     f"{_fmt(r['edge_out_comps_per_class'], 2)} | {_fmt(r['dispersion_ratio'], 3)} | "
                     f"{_fmt(r['heldout_nll_mix'], 3)} | {_fmt(r['heldout_nll_single'], 3)} | "
                     f"{_fmt(r['time_local_fit_s'], 1)} | {_fmt(r['time_edge_reestimate_s'], 1)} | "
                     f"{r['syn_edge']}/{r['syn_cloud']} | {r['infeasible']} |")
    if pending:
        L.append("\n## Not completed (excluded from all tables)\n")
        for rid, st in pending:
            L.append(f"- `{rid}`: {st}")
    fig = pca_figure(rows, os.path.join(OUT, f"{a.matrix}_pca.png")) if rows else None
    if fig:
        L.append(f"\nPCA figure: `{os.path.relpath(fig, ROOT)}`.\n")
    rt = [r["mean_round_s"] for r in rows if r["mean_round_s"]]
    if rt:
        from importlib import import_module
        sys.path.insert(0, os.path.join(ROOT, "scripts", "misfp"))
        ex = import_module("experiments")
        n_full = len(ex.full_runs())
        mean_rt = float(np.mean(rt))
        gpu_h = n_full * ex.FULL_EPOCHS * mean_rt / 3600
        L.append(f"\n## Cost projection\nMeasured mean round time across completed pilot runs: {mean_rt:.0f} s "
                 f"(shared GPU, concurrent campaign jobs). Full matrix: {n_full} runs × {ex.FULL_EPOCHS} rounds "
                 f"≈ {gpu_h:.0f} GPU-hours sequential (≈ {gpu_h / 24:.1f} days) at that throughput.\n")
    with open(os.path.join(OUT, f"{a.matrix.upper()}_REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
