#!/usr/bin/env bash
# ============================================================================
# run_smoke.sh — Quick smoke test (1 seed, 1 dataset per task)
#
# Verifies that H-SFP and all E-HSFP ablation modes run without crashing.
# Usage:
#   ./run_smoke.sh                  # default GPU 0
#   GPU_ID=1 ./run_smoke.sh        # use GPU 1
#   DRY_RUN=1 ./run_smoke.sh       # just print commands
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Override to 1 seed for smoke test
SEEDS="0"

check_env
ensure_results_dirs

log_info "=== SMOKE TEST ==="

# ── Classification: CIFAR ResNet50 (Ic=5, Ie=10) ──────────────────────────
SMOKE_CLS_CFG="configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"

# H-SFP reference (no ablation)
run_seeds "smoke" classification h-sfp "$SMOKE_CLS_CFG"

# E-HSFP ablation modes
ABLATION_MODES=("baseline_hsfp" "hsfp_memory" "hsfp_memory_dropout" "hsfp_memory_reliability" "hsfp_memory_reliability_prc" "full_e_hsfp")
for ablation in "${ABLATION_MODES[@]}"; do
    run_seeds "smoke" classification h-sfp "$SMOKE_CLS_CFG" "$ablation"
done

# ── Segmentation: ISIC ResNet50 (Ic=5, Ie=10) ─────────────────────────────
SMOKE_SEG_CFG="configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml"

run_seeds "smoke" segmentation h-sfp "$SMOKE_SEG_CFG"

for ablation in "${ABLATION_MODES[@]}"; do
    run_seeds "smoke" segmentation h-sfp "$SMOKE_SEG_CFG" "$ablation"
done

# ── One baseline per task ──────────────────────────────────────────────────
run_seeds "smoke" classification federated "configs/classification/federated/cifar_fedavg_resnet50.yaml"
run_seeds "smoke" segmentation federated "configs/segmentation/federated/isic_fedavg_resnet50.yaml"

print_summary "SMOKE TEST"
