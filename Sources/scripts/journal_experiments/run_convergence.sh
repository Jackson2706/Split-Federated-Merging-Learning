#!/usr/bin/env bash
# ============================================================================
# run_convergence.sh — Convergence curves for H-SFP (reference) & E-HSFP
#
# Runs H-SFP and full E-HSFP across all datasets/backbones, plus baselines.
# 5 seeds each. Output: results/journal/convergence/
#
# Usage:
#   ./run_convergence.sh
#   GPU_ID=0 SEEDS="0 1 2" ./run_convergence.sh   # fewer seeds
#   DRY_RUN=1 ./run_convergence.sh                 # preview commands
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

check_env
ensure_results_dirs

log_info "=== CONVERGENCE EXPERIMENTS ==="

# ============================================================================
# 1. H-SFP Reference (no ablation — vanilla H-SFP)
# ============================================================================
log_info "--- H-SFP Reference (classification) ---"

HSFP_CLS_CFGS=(
    "configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"
    "configs/classification/h-sfp/cifar_our_alexnet_5_10.yaml"
    "configs/classification/h-sfp/ham10000_our_resnet50_5_10.yaml"
    "configs/classification/h-sfp/ham10000_our_vgg_5_10.yaml"
)
for cfg in "${HSFP_CLS_CFGS[@]}"; do
    run_seeds "conv_hsfp" classification h-sfp "$cfg"
done

log_info "--- H-SFP Reference (segmentation) ---"

HSFP_SEG_CFGS=(
    "configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml"
)
for cfg in "${HSFP_SEG_CFGS[@]}"; do
    run_seeds "conv_hsfp" segmentation h-sfp "$cfg"
done

# ============================================================================
# 2. E-HSFP (full_e_hsfp ablation)
# ============================================================================
log_info "--- E-HSFP full (classification) ---"

for cfg in "${HSFP_CLS_CFGS[@]}"; do
    run_seeds "conv_ehsfp" classification h-sfp "$cfg" "full_e_hsfp"
done

log_info "--- E-HSFP full (segmentation) ---"

for cfg in "${HSFP_SEG_CFGS[@]}"; do
    run_seeds "conv_ehsfp" segmentation h-sfp "$cfg" "full_e_hsfp"
done

# ============================================================================
# 3. Baselines — Federated (FedAvg, FedProx, FedNova, FedSGD)
# ============================================================================
log_info "--- Federated Baselines (classification) ---"

FED_CLS_CFGS=(
    "configs/classification/federated/cifar_fedavg_resnet50.yaml"
    "configs/classification/federated/cifar_fedprox_resnet50.yaml"
    "configs/classification/federated/cifar_fednova_resnet50.yaml"
    "configs/classification/federated/cifar_fedsgd_resnet50.yaml"
    "configs/classification/federated/ham10000_fedavg_resnet50.yaml"
    "configs/classification/federated/ham10000_fedprox_resnet50.yaml"
    "configs/classification/federated/ham10000_fednova_resnet50.yaml"
    "configs/classification/federated/ham10000_fedsgd_resnet50.yaml"
)
for cfg in "${FED_CLS_CFGS[@]}"; do
    run_seeds "conv_fed" classification federated "$cfg"
done

log_info "--- Federated Baselines (segmentation) ---"

FED_SEG_CFGS=(
    "configs/segmentation/federated/isic_fedavg_resnet50.yaml"
    "configs/segmentation/federated/isic_fedprox_resnet50.yaml"
    "configs/segmentation/federated/isic_fednova_resnet50.yaml"
    "configs/segmentation/federated/isic_fedsgd_resnet50.yaml"
)
for cfg in "${FED_SEG_CFGS[@]}"; do
    run_seeds "conv_fed" segmentation federated "$cfg"
done

# ============================================================================
# 4. Baselines — HierFL
# ============================================================================
log_info "--- HierFL Baselines ---"

HIERFL_CLS_CFGS=(
    "configs/classification/hierfl/cifar_hierfl_resnet50.yaml"
    "configs/classification/hierfl/ham10000_hierfl_resnet50.yaml"
)
for cfg in "${HIERFL_CLS_CFGS[@]}"; do
    run_seeds "conv_hierfl" classification hierfl "$cfg"
done

run_seeds "conv_hierfl" segmentation hierfl "configs/segmentation/hierfl/isic_hierfl_resnet50.yaml"

# ============================================================================
# 5. Baselines — SplitFL
# ============================================================================
log_info "--- SplitFL Baselines ---"

SPLITFL_CLS_CFGS=(
    "configs/classification/splitfl/cifar_splitfed_resnet50.yaml"
    "configs/classification/splitfl/ham10000_splitfed_resnet50.yaml"
)
for cfg in "${SPLITFL_CLS_CFGS[@]}"; do
    run_seeds "conv_splitfl" classification splitfl "$cfg"
done

# No ISIC splitfl config — skip segmentation splitfl

# ============================================================================
# 6. Baselines — HeteroSFL
# ============================================================================
log_info "--- HeteroSFL Baselines ---"

run_seeds "conv_heterosfl" classification hetero-sfl "configs/classification/hetero-sfl/cifar_heteroSFL_resnet18.yaml"
run_seeds "conv_heterosfl" classification hetero-sfl "configs/classification/hetero-sfl/ham10000_heteroSFL_resnet50.yaml"
run_seeds "conv_heterosfl" segmentation hetero-sfl "configs/segmentation/hetero-sfl/isic_heteroSFL_resnet50.yaml"

# ============================================================================
# 7. Baselines — HSFL
# ============================================================================
log_info "--- HSFL Baselines ---"

HSFL_CLS_CFGS=(
    "configs/classification/hsfl/cifar_our_resnet50_5_10.yaml"
    "configs/classification/hsfl/ham10000_our_resnet50_5_10.yaml"
)
for cfg in "${HSFL_CLS_CFGS[@]}"; do
    run_seeds "conv_hsfl" classification hsfl "$cfg"
done

run_seeds "conv_hsfl" segmentation hsfl "configs/segmentation/hsfl/cifar_our_resnet50_5_10.yaml"

# ============================================================================
# Missing baselines (FedProto, FedGen, FedDF) — warn and skip
# ============================================================================
log_warn "FedProto, FedGen, FedDF are not implemented — skipping."

print_summary "CONVERGENCE"
