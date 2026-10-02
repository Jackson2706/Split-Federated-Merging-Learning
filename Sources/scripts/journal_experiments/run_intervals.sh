#!/usr/bin/env bash
# ============================================================================
# run_intervals.sh — Aggregation interval experiments
#
# Tests different (Ic, Ie) interval pairs for both H-SFP and E-HSFP:
#   (5, 10)   — frequent aggregation
#   (10, 20)  — moderate
#   (25, 50)  — infrequent
#
# These correspond to existing config files with _5_10, _10_20, _25_50 suffixes.
#
# Usage:
#   ./run_intervals.sh
#   GPU_ID=0 SEEDS="0 1 2" ./run_intervals.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

check_env
ensure_results_dirs

log_info "=== AGGREGATION INTERVAL EXPERIMENTS ==="

# ── Classification: ResNet50 on CIFAR ──────────────────────────────────────
CLS_INTERVAL_CFGS=(
    "configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"
    "configs/classification/h-sfp/cifar_our_resnet50_10_20.yaml"
    "configs/classification/h-sfp/cifar_our_resnet50_25_50.yaml"
)

# H-SFP reference
log_info "--- Intervals: H-SFP Classification (CIFAR ResNet50) ---"
for cfg in "${CLS_INTERVAL_CFGS[@]}"; do
    run_seeds "intv_hsfp" classification h-sfp "$cfg"
done

# E-HSFP full
log_info "--- Intervals: E-HSFP Classification (CIFAR ResNet50) ---"
for cfg in "${CLS_INTERVAL_CFGS[@]}"; do
    run_seeds "intv_ehsfp" classification h-sfp "$cfg" "full_e_hsfp"
done

# ── Classification: ResNet50 on HAM10000 ───────────────────────────────────
CLS_HAM_CFGS=(
    "configs/classification/h-sfp/ham10000_our_resnet50_5_10.yaml"
    "configs/classification/h-sfp/ham10000_our_resnet50_10_20.yaml"
    "configs/classification/h-sfp/ham10000_our_resnet50_25_50.yaml"
)

log_info "--- Intervals: H-SFP Classification (HAM10000 ResNet50) ---"
for cfg in "${CLS_HAM_CFGS[@]}"; do
    run_seeds "intv_hsfp" classification h-sfp "$cfg"
done

log_info "--- Intervals: E-HSFP Classification (HAM10000 ResNet50) ---"
for cfg in "${CLS_HAM_CFGS[@]}"; do
    run_seeds "intv_ehsfp" classification h-sfp "$cfg" "full_e_hsfp"
done

# ── Segmentation: ISIC ResNet50 ───────────────────────────────────────────
SEG_INTERVAL_CFGS=(
    "configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml"
    "configs/segmentation/h-sfp/isic_our_resnet50_10_20.yaml"
    "configs/segmentation/h-sfp/isic_our_resnet50_25_50.yaml"
)

log_info "--- Intervals: H-SFP Segmentation (ISIC ResNet50) ---"
for cfg in "${SEG_INTERVAL_CFGS[@]}"; do
    run_seeds "intv_hsfp" segmentation h-sfp "$cfg"
done

log_info "--- Intervals: E-HSFP Segmentation (ISIC ResNet50) ---"
for cfg in "${SEG_INTERVAL_CFGS[@]}"; do
    run_seeds "intv_ehsfp" segmentation h-sfp "$cfg" "full_e_hsfp"
done

print_summary "INTERVALS"
