"""Analysis scripts run end to end on a small debug results tree built through the queue."""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = {  # real configuration ids, debug dataset
    "E3_ehsfp_cifar100_iid_alpha0p3_tau0_oracle1": "alpha=0.3 tau=0 oracle=1 oracle_every=1",
    "E3_ehsfp_cifar100_iid_alpha1p0_tau0_oracle1": "alpha=1.0 tau=0 oracle=1 oracle_every=1",
    "E3_ehsfp_cifar100_iid_alpha0p3_tau5_oracle1": "alpha=0.3 tau=5 oracle=1 oracle_every=1",
    "E4_ehsfp_cifar100_iid_p_train0p2_loss0p3_oracle1": "p_train=0.2 loss=0.3 oracle=1 oracle_every=1",
    "E4_ehsfp_cifar100_iid_p_train0p0_loss0p3_oracle1": "p_train=0.0 loss=0.3 oracle=1 oracle_every=1",
    "E6_ehsfp_cifar100_iid_lambda_prc0p0": "lambda_prc=0.0",
    "E7_ehsfp_cifar100_iid_M_M0_weightingprecision": "M_M=0 weighting=precision",
}


def py(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_analysis_scripts_on_debug_runs(tmp_path):
    man = tmp_path / "runs.csv"
    with open(man, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "exp", "priority", "dataset", "partition", "method", "seed", "overrides"])
        for cid, ov in CONFIGS.items():
            for s in (0, 1):
                w.writerow([f"{cid}_s{s}", cid.split("_")[0], 1, "debug", "iid", "ehsfp", s, ov])
    res, out = tmp_path / "results", tmp_path / "out"
    py("run_queue.py", "--manifest", str(man), "--out-root", str(res), "--jobs", "4")
    assert len(list(res.glob("*/DONE"))) == 2 * len(CONFIGS)
    py("analysis/aggregate.py", "--root", str(res), "--out", str(out), "--expected", "2")
    py("analysis/significance.py", "--root", str(res), "--out", str(out))
    py("analysis/diagnostics.py", "--root", str(res), "--out", str(out))
    for f in ("aggregate.csv", "aggregate.tex", "paired.csv", "welch.csv", "d2_prop2_check.csv",
              "d3_alpha_sweep.csv", "d4_spearman.json", "diagnostics_table.csv", "fig_d1_drift_coverage.pdf",
              "fig_d2_prop2.pdf", "fig_d3_alpha_sweep.pdf"):
        assert (out / f).exists(), f
    paired = list(csv.DictReader(open(out / "paired.csv")))
    assert {r["exp"] for r in paired} == {"E3", "E4", "E6"} and all(r["n"] == "2" for r in paired)
    welch = list(csv.DictReader(open(out / "welch.csv")))
    assert [r["p"] for r in welch[:6]] == ["0.25", "0.48", "0.51", "0.97", "0.15", "0.22"]
    assert any(r["source"] == "new vs reported" for r in welch)
    table = list(csv.DictReader(open(out / "diagnostics_table.csv")))
    assert all(r["episodes without valid minibatch"] != "" for r in table)
    assert json.load(open(out / "d4_spearman.json"))["configurations"]["n"] == 5
    # strict mode refuses configurations with fewer seeds than expected
    r = subprocess.run([sys.executable, "analysis/aggregate.py", "--root", str(res), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "fewer than 3 seeds" in (r.stdout + r.stderr)
