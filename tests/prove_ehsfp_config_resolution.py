#!/usr/bin/env python3
"""CPU-only proof that CLI ablation injection survives H-SFP config resolution."""

import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
HSFP_DIR = ROOT / "classification" / "H-SFP"
CONFIG_PATH = ROOT / "configs" / "classification" / "h-sfp" / "cifar_our_resnet50_5_10.yaml"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HSFP_DIR))

from config import ConfigLoader  # noqa: E402
from ehsfp.config import get_ehsfp_config  # noqa: E402


def resolve_with_main_ablation_injection(ablation: str) -> tuple[dict, dict]:
    """Mirror main.py's YAML injection and the classification runner's loading."""
    with CONFIG_PATH.open() as stream:
        cfg_data = yaml.safe_load(stream) or {}
    cfg_data["ablation_mode"] = ablation
    cfg_data["seed"] = 0

    # main.py creates the temporary child beside the source config so its
    # relative ``base: default.yaml`` reference continues to resolve.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="proof_",
        dir=CONFIG_PATH.parent,
    ) as stream:
        yaml.safe_dump(cfg_data, stream, default_flow_style=False)
        stream.flush()
        resolved = ConfigLoader(stream.name).get_config()

    assert resolved["ablation_mode"] == ablation
    return resolved, get_ehsfp_config(resolved)


def main() -> None:
    baseline_resolved, baseline = resolve_with_main_ablation_injection("baseline_hsfp")
    full_resolved, full = resolve_with_main_ablation_injection("full_e_hsfp")

    differing = {key: (baseline[key], full[key]) for key in baseline if baseline[key] != full[key]}
    expected = {
        "ablation_mode",
        "aggregation_mode",
        "use_episodic_memory",
        "use_prc_loss",
        "use_prototype_dropout",
        "use_residual_generator",
        "use_serverless_simulation",
    }

    assert baseline_resolved is not full_resolved
    assert baseline["use_episodic_memory"] is False
    assert full["use_episodic_memory"] is True
    assert baseline["aggregation_mode"] == "average"
    assert full["aggregation_mode"] == "learnable_reliability"
    assert expected <= differing.keys()

    print(f"config: {CONFIG_PATH.relative_to(ROOT)}")
    print(f"resolved ablations: {baseline_resolved['ablation_mode']} != {full_resolved['ablation_mode']}")
    for key, values in sorted(differing.items()):
        print(f"{key}: {values[0]!r} -> {values[1]!r}")
    print("PASS: current config-resolution chain produces distinct effective E-HSFP configs")


if __name__ == "__main__":
    main()
