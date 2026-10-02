"""MiSFP configuration: defaults, variant presets, validation.

Variants (all are the same method, MiSFP; ``hsfp`` runs the preserved
original H-SFP implementation as reference A):

  hsfp      A  original H-SFP (unchanged HierarchicalFL code path)
  k1        B  MiSFP-K1: new packet path, K=1 everywhere, exact moment pooling
  edge      C  MiSFP-Edge: clients send K=1; edges/cloud retain a mixture
  k2        D  MiSFP-K2: fixed local K=2 + bounded hierarchical merging
  k4        E  MiSFP-K4: fixed local K=4 + bounded hierarchical merging
  adaptive  F  MiSFP-Adaptive: adaptive local K (cap 4) + budgeted merging
"""

from __future__ import annotations

import copy
from typing import Any, Dict

VARIANT_LABELS = {
    "hsfp": "H-SFP",
    "k1": "MiSFP-K1",
    "edge": "MiSFP-Edge",
    "k2": "MiSFP-K2",
    "k4": "MiSFP-K4",
    "adaptive": "MiSFP-Adaptive",
}

MISFP_DEFAULTS: Dict[str, Any] = {
    "misfp_variant": "k1",
    # --- local construction (client) ---
    "misfp_local_k_mode": "fixed",  # fixed | adaptive
    "misfp_local_k": 1,  # fixed K, or cap for adaptive
    "misfp_min_component_support": 3,
    "misfp_kmeans_iters": 50,
    "misfp_kmeans_n_init": 3,
    "misfp_reservoir_size": 4096,
    "misfp_adaptive_val_fraction": 0.3,
    "misfp_adaptive_min_val": 4,
    "misfp_adaptive_min_improvement": 0.02,
    "misfp_adaptive_penalty_per_component": 0.0,
    "misfp_nll_rel_var_floor": 0.01,
    "misfp_nll_abs_var_floor": 1.0e-6,
    "misfp_sample_var_floor": 1.0e-12,
    # --- hierarchical aggregation ---
    "misfp_pooling": "moment",  # moment | baseline_average (K=1 regression only)
    "misfp_edge_pool_cap": 1,  # comps/class kept at edge INPUT (None = retain all)
    "misfp_edge_pool_safety_cap": 32,  # hard bound on edge-input comps/class
    "misfp_boundary_mode": "propagate",  # propagate | refit
    "misfp_boundary_samples_per_component": 50,  # propagate: MC draws per comp
    "misfp_boundary_samples_per_class": 200,  # refit: MC draws per class
    "misfp_boundary_refit_k_mode": "adaptive",
    "misfp_boundary_refit_k": 4,
    "misfp_edge_out_cap": 1,  # comps/class in the edge->cloud packet
    "misfp_cloud_pool_cap": None,  # comps/class kept at the cloud (None = all)
    # --- communication budgets (per packet, per round) ---
    "misfp_precision": "float32",  # float32 | float16
    "misfp_client_budget": None,  # None | int bytes | "match_k1_fp32[:ratio]"
    "misfp_edge_budget": None,  # same grammar, edge->cloud packet
    "misfp_on_infeasible": "warn",  # warn | raise
    # --- representation compatibility ---
    "misfp_extraction": "local",  # local | shared_snapshot
    # --- diagnostics ---
    "misfp_heldout_diagnostic": True,
    "misfp_diag_pca_classes": 4,
}

VARIANT_PRESETS: Dict[str, Dict[str, Any]] = {
    "hsfp": {},
    "k1": {
        "misfp_local_k_mode": "fixed", "misfp_local_k": 1,
        "misfp_edge_pool_cap": 1, "misfp_edge_out_cap": 1, "misfp_cloud_pool_cap": 1,
    },
    "edge": {
        "misfp_local_k_mode": "fixed", "misfp_local_k": 1,
        "misfp_edge_pool_cap": None, "misfp_edge_out_cap": 4, "misfp_cloud_pool_cap": None,
    },
    "k2": {
        "misfp_local_k_mode": "fixed", "misfp_local_k": 2,
        "misfp_edge_pool_cap": None, "misfp_edge_out_cap": 4, "misfp_cloud_pool_cap": None,
    },
    "k4": {
        "misfp_local_k_mode": "fixed", "misfp_local_k": 4,
        "misfp_edge_pool_cap": None, "misfp_edge_out_cap": 4, "misfp_cloud_pool_cap": None,
    },
    "adaptive": {
        "misfp_local_k_mode": "adaptive", "misfp_local_k": 4,
        "misfp_edge_pool_cap": None, "misfp_edge_out_cap": 4, "misfp_cloud_pool_cap": None,
    },
}

# E-HSFP mechanisms stay outside MiSFP; enabling them with a MiSFP packet path is
# rejected rather than silently combined (they assume one mean/std per class).
_EHSFP_OFF = {
    "use_episodic_memory": False,
    "use_prc_loss": False,
    "use_prototype_dropout": False,
    "use_serverless_simulation": False,
    "use_residual_generator": False,
    "aggregation_mode": "average",
}


def _parse_budget(value):
    if value in (None, "", "none", "None"):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value <= 0:
            raise ValueError("byte budget must be positive")
        return ("absolute", int(value))
    s = str(value)
    if s.startswith("match_k1_fp32"):
        ratio = float(s.split(":", 1)[1]) if ":" in s else 1.0
        if ratio <= 0:
            raise ValueError("budget ratio must be positive")
        return ("match_k1_fp32", ratio)
    raise ValueError(f"unrecognised budget {value!r}")


def get_misfp_config(config: Dict[str, Any], ehsfp_cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """Resolve defaults <- variant preset <- explicit misfp_* config keys."""
    variant = config.get("misfp_variant", MISFP_DEFAULTS["misfp_variant"])
    if variant not in VARIANT_PRESETS:
        raise ValueError(f"unknown misfp_variant {variant!r}; choose from {sorted(VARIANT_PRESETS)}")
    cfg = copy.deepcopy(MISFP_DEFAULTS)
    cfg.update(VARIANT_PRESETS[variant])
    explicit = {k: v for k, v in config.items() if k.startswith("misfp_")}
    cfg.update(explicit)
    cfg["misfp_variant"] = variant
    cfg["misfp_label"] = VARIANT_LABELS[variant]
    if variant == "hsfp":
        return cfg

    if cfg["misfp_local_k_mode"] not in ("fixed", "adaptive"):
        raise ValueError("misfp_local_k_mode must be fixed|adaptive")
    if cfg["misfp_pooling"] not in ("moment", "baseline_average"):
        raise ValueError("misfp_pooling must be moment|baseline_average")
    if cfg["misfp_pooling"] == "baseline_average" and (
        cfg["misfp_local_k"] != 1 or cfg["misfp_edge_pool_cap"] != 1
        or cfg["misfp_edge_out_cap"] != 1 or cfg["misfp_cloud_pool_cap"] != 1
    ):
        raise ValueError("misfp_pooling=baseline_average is only defined for the K=1 path")
    if cfg["misfp_boundary_mode"] not in ("propagate", "refit"):
        raise ValueError("misfp_boundary_mode must be propagate|refit")
    if cfg["misfp_precision"] not in ("float32", "float16"):
        raise ValueError("misfp_precision must be float32|float16")
    if cfg["misfp_extraction"] not in ("local", "shared_snapshot"):
        raise ValueError("misfp_extraction must be local|shared_snapshot")
    if cfg["misfp_on_infeasible"] not in ("warn", "raise"):
        raise ValueError("misfp_on_infeasible must be warn|raise")
    for key in ("misfp_edge_pool_cap", "misfp_edge_out_cap", "misfp_cloud_pool_cap"):
        if cfg[key] is not None and int(cfg[key]) < 1:
            raise ValueError(f"{key} must be >= 1 or null")
    cfg["_client_budget"] = _parse_budget(cfg["misfp_client_budget"])
    cfg["_edge_budget"] = _parse_budget(cfg["misfp_edge_budget"])

    if ehsfp_cfg is not None:
        bad = {k: ehsfp_cfg.get(k) for k, v in _EHSFP_OFF.items() if ehsfp_cfg.get(k, v) != v}
        if bad:
            raise ValueError(
                "MiSFP variants run without E-HSFP mechanisms (memory, reliability, PRC, "
                f"dropout, serverless, generator); got {bad}. Use --method h-sfp for E-HSFP.")
    for key, neutral in (("prototype_space", "raw"), ("staleness_tau", 0),
                         ("partial_edge_probability", 0.0)):
        val = config.get(key)
        if val not in (None, neutral, "", 0, 0.0) and val != neutral:
            raise ValueError(f"MiSFP requires {key}={neutral!r}; got {val!r}")
    if config.get("cold_start_defers_packet"):
        raise ValueError("MiSFP does not support cold_start_defers_packet")
    return cfg


def public_view(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe resolved config (drops parsed private fields)."""
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    out["client_budget_parsed"] = list(cfg["_client_budget"]) if cfg.get("_client_budget") else None
    out["edge_budget_parsed"] = list(cfg["_edge_budget"]) if cfg.get("_edge_budget") else None
    return out
