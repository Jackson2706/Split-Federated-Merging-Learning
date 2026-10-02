#!/usr/bin/env bash
# ============================================================================
# run_dropout_staleness.sh — Robustness experiments
#
# Part A: Prototype dropout rate sweep (0.0, 0.1, 0.3, 0.5, 0.7)
# Part B: Staleness (max_prototype_age) sweep (0, 1, 3, 5, 10)
#
# Both parts use full E-HSFP with the swept parameter overridden via a
# temporary YAML config.
#
# Usage:
#   ./run_dropout_staleness.sh
#   GPU_ID=2 ./run_dropout_staleness.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

check_env
ensure_results_dirs

# Helper: create a temp config with a YAML key overridden
# Usage: override_yaml ORIGINAL_CFG KEY VALUE -> prints temp file path
override_yaml() {
    local original="$1" key="$2" value="$3"
    local tmp
    tmp=$(mktemp "${RESULTS_ROOT}/.tmp_cfg_XXXXXX.yaml")
    python3 -c "
import yaml, sys
with open('$original') as f:
    d = yaml.safe_load(f) or {}
d['$key'] = $value
with open('$tmp', 'w') as f:
    yaml.dump(d, f, default_flow_style=False)
"
    echo "$tmp"
}

# ── Configs to sweep ──────────────────────────────────────────────────────
CLS_CFG="configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"
SEG_CFG="configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml"

# ============================================================================
# Part A: Dropout Rate Sweep
# ============================================================================
log_info "=== DROPOUT RATE SWEEP ==="

DROPOUT_RATES=(0.0 0.1 0.3 0.5 0.7)

for rate in "${DROPOUT_RATES[@]}"; do
    rate_tag="dr$(echo "$rate" | tr '.' 'p')"

    # Classification
    tmp_cfg=$(override_yaml "${PROJECT_ROOT}/${CLS_CFG}" "prototype_dropout_rate" "$rate")
    for seed in $SEEDS; do
        eid=$(make_experiment_id "dropout" "cls" "$rate_tag" "s${seed}")
        run_one "$eid" classification h-sfp "$tmp_cfg" "$seed" "full_e_hsfp" || true
    done
    rm -f "$tmp_cfg"

    # Segmentation
    tmp_cfg=$(override_yaml "${PROJECT_ROOT}/${SEG_CFG}" "prototype_dropout_rate" "$rate")
    for seed in $SEEDS; do
        eid=$(make_experiment_id "dropout" "seg" "$rate_tag" "s${seed}")
        run_one "$eid" segmentation h-sfp "$tmp_cfg" "$seed" "full_e_hsfp" || true
    done
    rm -f "$tmp_cfg"
done

# ============================================================================
# Part B: Staleness (max_prototype_age) Sweep
# ============================================================================
log_info "=== STALENESS SWEEP ==="

STALENESS_VALUES=(0 1 3 5 10)

for tau in "${STALENESS_VALUES[@]}"; do
    tau_tag="tau${tau}"

    # Classification
    tmp_cfg=$(override_yaml "${PROJECT_ROOT}/${CLS_CFG}" "max_prototype_age" "$tau")
    for seed in $SEEDS; do
        eid=$(make_experiment_id "staleness" "cls" "$tau_tag" "s${seed}")
        run_one "$eid" classification h-sfp "$tmp_cfg" "$seed" "full_e_hsfp" || true
    done
    rm -f "$tmp_cfg"

    # Segmentation
    tmp_cfg=$(override_yaml "${PROJECT_ROOT}/${SEG_CFG}" "max_prototype_age" "$tau")
    for seed in $SEEDS; do
        eid=$(make_experiment_id "staleness" "seg" "$tau_tag" "s${seed}")
        run_one "$eid" segmentation h-sfp "$tmp_cfg" "$seed" "full_e_hsfp" || true
    done
    rm -f "$tmp_cfg"
done

print_summary "DROPOUT & STALENESS"
