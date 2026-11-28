"""C1/C2 end-to-end on configs/debug.yaml (CPU)."""
import csv
import json
import os

import pytest

from test_revision_pipeline import REF, ROOT, rows, run, training  # noqa: F401

# Baseline measured with the UNMODIFIED code on this server (the repo golden file was produced with
# other library versions and does not match even before any C1/C2 change).
BASE = json.load(open(os.path.join(ROOT, "tests", "golden", "local_server_baseline.json")))
ARGS = {"default": [], "reference": ["--ablation", "full_e_hsfp", "--set", "use_serverless_simulation=false",
        "--set", "synthesis=shared_cov", "--set", "client_sync=true", "--set", "dropout_mode=both",
        "--set", "supcon_temperature=0.5"],
        "stress": ["--ablation", "full_e_hsfp", "--set", "use_serverless_simulation=false", "--set", "synthesis=shared_cov",
                   "--set", "client_sync=true", "--set", "client_global_sync=true", "--set", "dropout_mode=both",
                   "--set", "staleness_tau=2", "--set", "stat_accumulation_decay=0.8"]}
C12 = ["--drift-comp", "rate", "--alpha-mode", "optimal"]
STRESS = REF + ["--max-delay", "2", "--p-train", "0.3"]


@pytest.mark.parametrize("variant", sorted(BASE))
def test_flags_off_matches_unmodified_baseline(tmp_path, variant):
    run(tmp_path, *ARGS[variant])
    assert training(tmp_path) == BASE[variant]


def test_c1_c2_run_and_log_columns(tmp_path):
    run(tmp_path, *STRESS, *C12, "--oracle", "1", "--oracle-every", "1", "--alpha", "0.5")
    orc = rows(tmp_path, "oracle.csv")
    assert {"comp_n", "comp_not_n", "comp_err_ratio", "mix_err_actual", "mix_err_fixed"} <= set(orc[0])
    al = rows(tmp_path, "alpha_log.csv")
    assert al and {"alpha_mean", "alpha_median", "fallback_share"} <= set(al[0])
    s = json.load(open(tmp_path / "summary.json"))
    assert {"gpu_peak_allocated_gb", "gpu_peak_reserved_gb", "dc_state_mb"} <= set(s)
    assert s["dc_state_mb"]


def test_oracle_on_off_identical_training_with_c1_c2(tmp_path):
    run(tmp_path / "off", *STRESS, *C12)
    run(tmp_path / "on", *STRESS, *C12, "--oracle", "1", "--oracle-every", "1")
    assert training(tmp_path / "off") == training(tmp_path / "on")


def test_resume_exact_with_c1_c2(tmp_path):
    args = (*STRESS, *C12, "--set", "ckpt_every=1")
    run(tmp_path / "full", *args)
    r = run(tmp_path / "cut", *args, env={"EHSFP_STOP_AFTER_ROUND": "2"}, check=False)
    assert r.returncode != 0 and not (tmp_path / "cut" / "DONE").exists()
    run(tmp_path / "cut", *args)
    assert training(tmp_path / "cut") == training(tmp_path / "full")
