"""
E-HSFP configuration defaults and ablation presets.

All E-HSFP features are disabled by default so that the baseline H-SFP
behavior is preserved when no E-HSFP config keys are present.
"""

from typing import Dict, Any

EHSFP_DEFAULTS: Dict[str, Any] = {
    # --- Episodic Prototype Memory ---
    "use_episodic_memory": False,
    "memory_size": 500,
    "memory_replay_ratio": 0.3,       # alpha in Q_mix = alpha*current + (1-alpha)*memory
    "memory_top_k": 5,
    "max_prototype_age": 20,

    # --- Learnable Reliability-Aware Aggregation ---
    "aggregation_mode": "average",     # "average" | "learnable_reliability"
    "reliability_hidden_dim": 32,
    "reliability_lr": 1e-3,
    "reliability_weight_decay": 1e-4,

    # --- Prototype Replay Consistency Loss ---
    "use_prc_loss": False,
    "lambda_prc": 0.1,
    "prc_num_samples": 16,

    # --- Serverless Prototype Dropout ---
    "use_prototype_dropout": False,
    "prototype_dropout_rate": 0.2,
    "dropout_mode": "client_prototype",  # "client_prototype" | "edge_prototype" | "both"
    "use_dropout_consistency": False,
    "lambda_dropout": 0.1,

    # --- Serverless Episode Simulator ---
    "use_serverless_simulation": False,
    "cold_start_probability": 0.15,
    "function_timeout_probability": 0.05,
    "max_episode_duration": 30.0,
    "latency_mean": 0.5,
    "latency_std": 0.2,

    # --- Residual Prototype Generator ---
    "use_residual_generator": False,
    "generator_hidden_dim": 64,
    "generator_scale": 0.1,

    # --- Ablation mode (overrides above flags) ---
    "ablation_mode": None,  # set via --ablation CLI flag
}

ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline_hsfp": {
        "use_episodic_memory": False,
        "aggregation_mode": "average",
        "use_prc_loss": False,
        "use_prototype_dropout": False,
        "use_serverless_simulation": False,
        "use_residual_generator": False,
    },
    "hsfp_memory": {
        "use_episodic_memory": True,
        "aggregation_mode": "average",
        "use_prc_loss": False,
        "use_prototype_dropout": False,
    },
    "hsfp_memory_dropout": {
        "use_episodic_memory": True,
        "aggregation_mode": "average",
        "use_prc_loss": False,
        "use_prototype_dropout": True,
        "use_serverless_simulation": True,
    },
    "hsfp_memory_reliability": {
        "use_episodic_memory": True,
        "aggregation_mode": "learnable_reliability",
        "use_prc_loss": False,
        "use_prototype_dropout": False,
    },
    "hsfp_memory_reliability_prc": {
        "use_episodic_memory": True,
        "aggregation_mode": "learnable_reliability",
        "use_prc_loss": True,
        "use_prototype_dropout": False,
    },
    "full_e_hsfp": {
        "use_episodic_memory": True,
        "aggregation_mode": "learnable_reliability",
        "use_prc_loss": True,
        "use_prototype_dropout": True,
        "use_serverless_simulation": True,
        "use_residual_generator": True,
    },
}


def get_ehsfp_config(user_config: dict) -> dict:
    """Merge E-HSFP defaults with user config, applying ablation preset if set."""
    cfg = dict(EHSFP_DEFAULTS)
    # Load the resolved YAML first.  The repository defaults contain explicit
    # ``false`` values, so applying them after a preset silently disabled every
    # ablation selected by the CLI.
    for k in EHSFP_DEFAULTS:
        if k in user_config:
            cfg[k] = user_config[k]
    # An ablation is an effective configuration preset and therefore wins over
    # inherited YAML defaults (matching the comments in default.yaml).
    ablation = user_config.get("ablation_mode")
    if ablation:
        if ablation not in ABLATION_PRESETS:
            raise ValueError(f"Unknown E-HSFP ablation preset: {ablation}")
        cfg.update(ABLATION_PRESETS[ablation])
        cfg["ablation_mode"] = ablation
    return cfg
