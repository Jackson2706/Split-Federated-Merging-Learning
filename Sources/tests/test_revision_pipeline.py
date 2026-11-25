"""End-to-end revision checks on configs/debug.yaml (CPU, ~1 min total)."""
import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = json.load(open(os.path.join(ROOT, "tests", "golden", "debug_golden.json")))
# Paper reference condition on the debug model (preset + no execution events,
# mask at both interfaces, T_temp = 0.5, shared covariance, synchronized clients).
REF = ["--ablation", "full_e_hsfp", "--set", "ehsfp.use_serverless_simulation=false",
       "--set", "ehsfp.dropout_mode=both", "--set", "supcon_temperature=0.5",
       "--set", "synthesis=shared_cov", "--set", "client_sync=true"]
ZERO = ["--ablation", "full_e_hsfp", "--set", "ehsfp.use_serverless_simulation=false",
        "--set", "frac=1.0", "--set", "local_bs=5",  # 125 samples/client: no drop_last loss
        "--alpha", "1", "--p-train", "0", "--packet-loss", "0", "--max-delay", "0",
        "--lambda-prc", "0", "--weighting", "count", "--oracle", "1", "--oracle-every", "1"]
TRAIN_COLS = ("round", "accuracy", "macro_f1", "cloud_loss", "edge_loss", "client_loss", "no_valid_minibatch")


def run(out, *args, env=None, check=True):
    cmd = [sys.executable, "main.py", "--task", "classification", "--method", "h-sfp",
           "--cfg", "configs/debug.yaml", "--seed", "0", "--out", str(out), *args]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check:
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def rows(out, name="metrics.csv"):
    with open(os.path.join(out, name), newline="") as fh:
        return list(csv.DictReader(fh))


def training(out):
    return [{k: r[k] for k in TRAIN_COLS} for r in rows(out)]


@pytest.mark.parametrize("variant", sorted(GOLDEN))
def test_defaults_reproduce_pre_revision_golden(tmp_path, variant):
    g = GOLDEN[variant]
    run(tmp_path, *g["args"])
    m = rows(tmp_path)
    assert [float(r["accuracy"]) for r in m] == g["metrics"]["validation_accuracy"]
    assert [float(r["macro_f1"]) for r in m] == g["metrics"]["validation_f1"]
    assert [float(r["cloud_loss"]) for r in m] == g["metrics"]["cloud_loss"]
    assert json.load(open(tmp_path / "summary.json"))["test_accuracy"] == g["metrics"]["test_accuracy"]
    assert (tmp_path / "DONE").exists() and (tmp_path / "final.pt").exists()


def test_reference_is_deterministic(tmp_path):
    run(tmp_path / "a", *REF)
    run(tmp_path / "b", *REF)
    assert training(tmp_path / "a") == training(tmp_path / "b")


@pytest.mark.parametrize("extra", [[], ["--set", "client_sync=true", "--set", "synthesis=shared_cov"]])
def test_oracle_is_zero_without_perturbations(tmp_path, extra):
    run(tmp_path, *ZERO, *extra)
    orc = rows(tmp_path, "oracle.csv")
    assert len(orc) == 3 * 3  # 3 rounds x (2 edges + cloud)
    for r in orc:
        assert float(r["kappa"]) == 1.0, r
        assert float(r["delta_proto"]) < 1e-6, r


def test_oracle_does_not_change_training(tmp_path):
    stress = REF + ["--p-train", "0.5", "--packet-loss", "0.3", "--max-delay", "1"]
    run(tmp_path / "off", *stress)
    run(tmp_path / "on", *stress, "--oracle", "1", "--oracle-every", "1")
    assert training(tmp_path / "off") == training(tmp_path / "on")
    assert rows(tmp_path / "on", "oracle.csv") and rows(tmp_path / "on", "oracle_mask.csv")


def test_every_feature_runs_end_to_end(tmp_path):
    run(tmp_path, *REF, "--oracle", "1", "--oracle-every", "1", "--packet-loss", "0.3",
        "--sampler", "mixture", "--scatter-fix", "1", "--prc", "corrected", "--weighting", "precision",
        "--retention", "5", "--alpha", "0.5", "--gpu-data", "1", "--max-delay", "2")
    assert (tmp_path / "DONE").exists()
    s = json.load(open(tmp_path / "summary.json"))
    assert {"selected_ece", "selected_nll", "final_ece", "relative_rounds"} <= set(s)


def test_retention_zero_disables_bank_despite_preset(tmp_path):
    run(tmp_path, *REF, "--retention", "0")
    cfg = json.load(open(next((tmp_path / "run").glob("*/*/run_metadata.json"))))
    assert cfg["effective_ehsfp"]["use_episodic_memory"] is False


def test_resume_is_exact(tmp_path):
    args = (*REF, "--p-train", "0.5", "--max-delay", "1", "--set", "ckpt_every=1")
    run(tmp_path / "full", *args)
    r = run(tmp_path / "cut", *args, env={"EHSFP_STOP_AFTER_ROUND": "2"}, check=False)
    assert r.returncode != 0 and not (tmp_path / "cut" / "DONE").exists()
    assert len(rows(tmp_path / "cut")) == 2
    run(tmp_path / "cut", *args)
    assert training(tmp_path / "cut") == training(tmp_path / "full")
    a = json.load(open(tmp_path / "full" / "summary.json"))
    b = json.load(open(tmp_path / "cut" / "summary.json"))
    assert {k: v for k, v in a.items() if k != "total_time_s"} == {k: v for k, v in b.items() if k != "total_time_s"}


def test_queue_dry_run():
    q = os.path.join(ROOT, "run_queue.py")
    if not os.path.exists(q):
        pytest.skip("run_queue.py not in the repository yet")
    r = subprocess.run([sys.executable, q, "--dry-run"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "main.py" in r.stdout


def test_gpu_data_cache_matches_on_the_fly(tmp_path):
    cache = tmp_path / "train.pt"
    subprocess.run([sys.executable, "scripts/preprocess.py", "--cfg", "configs/debug.yaml", "--out", str(cache)],
                   cwd=ROOT, check=True, capture_output=True)
    run(tmp_path / "a", *REF, "--gpu-data", "1")
    run(tmp_path / "b", *REF, "--gpu-data", "1", "--set", f"gpu_data_cache={cache}")
    assert training(tmp_path / "a") == training(tmp_path / "b")
