#!/usr/bin/env bash
# ============================================================================
# run_ablation.sh — E-HSFP component ablation study
#
# Runs all 6 ablation presets across datasets.  H-SFP (baseline_hsfp) is
# included as the first row; each successive preset adds one component.
#
# Ablation ladder:
#   1. baseline_hsfp              (= vanilla H-SFP, no E-HSFP features)
#   2. hsfp_memory                (+ episodic memory)
#   3. hsfp_memory_dropout        (+ memory + prototype dropout)
#   4. hsfp_memory_reliability    (+ memory + reliability aggregation)
#   5. hsfp_memory_reliability_prc (+ memory + reliability + PRC loss)
#   6. full_e_hsfp                (all E-HSFP features)
#
# Usage:
#   ./run_ablation.sh
#   GPU_ID=1 SEEDS="0 1 2 3 4" ./run_ablation.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

check_env
ensure_results_dirs

log_info "=== ABLATION STUDY ==="

ABLATION_MODES=(
    "baseline_hsfp"
    "hsfp_memory"
    "hsfp_memory_dropout"
    "hsfp_memory_reliability"
    "hsfp_memory_reliability_prc"
    "full_e_hsfp"
)

# ── Classification ─────────────────────────────────────────────────────────
# Use representative configs: ResNet50 on CIFAR and HAM10000
CLS_CFGS=(
    "configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"
    "configs/classification/h-sfp/ham10000_our_resnet50_5_10.yaml"
)

log_info "--- Ablation: Classification ---"
for ablation in "${ABLATION_MODES[@]}"; do
    for cfg in "${CLS_CFGS[@]}"; do
        run_seeds "abl" classification h-sfp "$cfg" "$ablation"
    done
done

# ── Segmentation ──────────────────────────────────────────────────────────
SEG_CFGS=(
    "configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml"
)

log_info "--- Ablation: Segmentation ---"
for ablation in "${ABLATION_MODES[@]}"; do
    for cfg in "${SEG_CFGS[@]}"; do
        run_seeds "abl" segmentation h-sfp "$cfg" "$ablation"
    done
done

print_summary "ABLATION"
