#!/usr/bin/env bash
# ============================================================================
# Run all H-SFP experiments (ECCV 2026)
#
# Usage:
#   ./run.sh                    # run everything
#   ./run.sh classification     # run only classification experiments
#   ./run.sh segmentation       # run only segmentation experiments
#   ./run.sh --wandb            # run everything with W&B logging
#   ./run.sh classification --wandb --wandb-project MyProject
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
TASK_FILTER="all"
WANDB_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --wandb)
            WANDB_ARGS+=" --wandb"
            ;;
        --wandb-project=*)
            WANDB_ARGS+=" --wandb-project ${arg#*=}"
            ;;
        --wandb-entity=*)
            WANDB_ARGS+=" --wandb-entity ${arg#*=}"
            ;;
        classification|segmentation|all)
            TASK_FILTER="$arg"
            ;;
    esac
done
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

TOTAL=0
PASSED=0
FAILED=0
FAILED_LIST=()

run_experiment() {
    local task="$1"
    local method="$2"
    local cfg="$3"
    local ablation="${4:-}"

    local name
    name="$(basename "$cfg" .yaml)"
    local dataset
    dataset="$(echo "$name" | cut -d'_' -f1)"
    local ablation_suffix=""
    local ablation_flag=""
    if [[ -n "$ablation" ]]; then
        ablation_suffix="_${ablation}"
        ablation_flag="--ablation $ablation"
    fi
    local log_file="${LOG_DIR}/${task}_${dataset}_${method}_${name}${ablation_suffix}.log"

    TOTAL=$((TOTAL + 1))
    echo "========================================"
    echo "[${TOTAL}] task=${task}  method=${method}  ablation=${ablation:-none}"
    echo "     cfg=${cfg}"
    echo "     log=${log_file}"
    echo "========================================"

    if python main.py --task "$task" --method "$method" --cfg "$cfg" $ablation_flag $WANDB_ARGS 2>&1 | tee "$log_file"; then
        PASSED=$((PASSED + 1))
        echo "[PASS] ${task}/${method}/${name}${ablation_suffix}"
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("${task}/${method}/${name}${ablation_suffix}")
        echo "[FAIL] ${task}/${method}/${name}${ablation_suffix}"
    fi
    echo ""
}

# ============================================================================
# Classification
# ============================================================================
if [[ "$TASK_FILTER" == "all" || "$TASK_FILTER" == "classification" ]]; then

    echo "############################################################"
    echo "# CLASSIFICATION — H-SFP (Primary Method)"
    echo "############################################################"
    for cfg in configs/classification/h-sfp/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment classification h-sfp "$cfg"
    done

    echo "############################################################"
    echo "# CLASSIFICATION — Federated Baselines"
    echo "############################################################"
    for cfg in configs/classification/federated/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment classification federated "$cfg"
    done

    echo "############################################################"
    echo "# CLASSIFICATION — HierFL"
    echo "############################################################"
    for cfg in configs/classification/hierfl/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment classification hierfl "$cfg"
    done

    echo "############################################################"
    echo "# CLASSIFICATION — SplitFL"
    echo "############################################################"
    for cfg in configs/classification/splitfl/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment classification splitfl "$cfg"
    done

    echo "############################################################"
    echo "# CLASSIFICATION — HeteroSFL"
    echo "############################################################"
    for cfg in configs/classification/hetero-sfl/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment classification hetero-sfl "$cfg"
    done

    echo "############################################################"
    echo "# CLASSIFICATION — HSFL"
    echo "############################################################"
    for cfg in configs/classification/hsfl/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment classification hsfl "$cfg"
    done

fi

# ============================================================================
# Segmentation
# ============================================================================
if [[ "$TASK_FILTER" == "all" || "$TASK_FILTER" == "segmentation" ]]; then

    echo "############################################################"
    echo "# SEGMENTATION — H-SFP (Primary Method)"
    echo "############################################################"
    for cfg in configs/segmentation/h-sfp/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment segmentation h-sfp "$cfg"
    done

    echo "############################################################"
    echo "# SEGMENTATION — Federated Baselines"
    echo "############################################################"
    for cfg in configs/segmentation/federated/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment segmentation federated "$cfg"
    done

    echo "############################################################"
    echo "# SEGMENTATION — HierFL"
    echo "############################################################"
    for cfg in configs/segmentation/hierfl/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment segmentation hierfl "$cfg"
    done

    echo "############################################################"
    echo "# SEGMENTATION — SplitFL"
    echo "############################################################"
    for cfg in configs/segmentation/splitfl/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment segmentation splitfl "$cfg"
    done

    echo "############################################################"
    echo "# SEGMENTATION — HeteroSFL"
    echo "############################################################"
    for cfg in configs/segmentation/hetero-sfl/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment segmentation hetero-sfl "$cfg"
    done

    echo "############################################################"
    echo "# SEGMENTATION — HSFL"
    echo "############################################################"
    for cfg in configs/segmentation/hsfl/*.yaml; do
        [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
        run_experiment segmentation hsfl "$cfg"
    done

fi

# ============================================================================
# E-HSFP Ablation Studies
# ============================================================================
if [[ "$TASK_FILTER" == "all" || "$TASK_FILTER" == "ablation" ]]; then

    ABLATION_MODES=("baseline_hsfp" "hsfp_memory" "hsfp_memory_dropout" "hsfp_memory_reliability" "hsfp_memory_reliability_prc" "full_e_hsfp")

    echo "############################################################"
    echo "# E-HSFP ABLATION — Classification"
    echo "############################################################"
    for ablation in "${ABLATION_MODES[@]}"; do
        for cfg in configs/classification/h-sfp/*.yaml; do
            [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
            run_experiment classification h-sfp "$cfg" "$ablation"
        done
    done

    echo "############################################################"
    echo "# E-HSFP ABLATION — Segmentation"
    echo "############################################################"
    for ablation in "${ABLATION_MODES[@]}"; do
        for cfg in configs/segmentation/h-sfp/*.yaml; do
            [[ "$(basename "$cfg")" == "default.yaml" ]] && continue
            run_experiment segmentation h-sfp "$cfg" "$ablation"
        done
    done

fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "                    EXPERIMENT SUMMARY"
echo "============================================================"
echo "  Total:  ${TOTAL}"
echo "  Passed: ${PASSED}"
echo "  Failed: ${FAILED}"
echo "  Logs:   ${LOG_DIR}/"
if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    echo ""
    echo "  Failed experiments:"
    for f in "${FAILED_LIST[@]}"; do
        echo "    - ${f}"
    done
fi
echo "============================================================"
