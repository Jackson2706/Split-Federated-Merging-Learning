#!/usr/bin/env bash
# ============================================================================
# run_all_journal.sh — Master launcher for all E-HSFP journal experiments
#
# Runs every experiment group sequentially. Each sub-script has its own
# resume support, so you can safely re-run this after a crash.
#
# Usage:
#   ./run_all_journal.sh                              # run everything
#   GPU_ID=0 ./run_all_journal.sh                     # specific GPU
#   WANDB_ARGS="--wandb --wandb-project E-HSFP" ./run_all_journal.sh
#   DRY_RUN=1 ./run_all_journal.sh                    # preview all commands
#   ./run_all_journal.sh --reset                      # clear all markers first
#   ./run_all_journal.sh convergence ablation          # run specific groups
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Parse arguments
RUN_GROUPS=()
DO_RESET=0

for arg in "$@"; do
    case "$arg" in
        --reset)
            DO_RESET=1
            ;;
        smoke|convergence|ablation|dropout_staleness|intervals)
            RUN_GROUPS+=("$arg")
            ;;
        *)
            log_warn "Unknown argument: $arg"
            ;;
    esac
done

# Default: run all groups
if [[ ${#RUN_GROUPS[@]} -eq 0 ]]; then
    RUN_GROUPS=(smoke convergence ablation dropout_staleness intervals)
fi

# Reset markers if requested
if [[ "$DO_RESET" -eq 1 ]]; then
    reset_markers "*"
fi

ensure_results_dirs

OVERALL_START=$(date +%s)

for group in "${RUN_GROUPS[@]}"; do
    log_info "=========================================="
    log_info "  Starting group: ${group}"
    log_info "=========================================="

    case "$group" in
        smoke)
            bash "${SCRIPT_DIR}/run_smoke.sh"
            ;;
        convergence)
            bash "${SCRIPT_DIR}/run_convergence.sh"
            ;;
        ablation)
            bash "${SCRIPT_DIR}/run_ablation.sh"
            ;;
        dropout_staleness)
            bash "${SCRIPT_DIR}/run_dropout_staleness.sh"
            ;;
        intervals)
            bash "${SCRIPT_DIR}/run_intervals.sh"
            ;;
    esac
done

OVERALL_END=$(date +%s)
ELAPSED=$(( OVERALL_END - OVERALL_START ))

echo ""
echo "============================================================"
echo "  ALL JOURNAL EXPERIMENTS COMPLETE"
echo "  Total wall time: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
echo "  Results: ${RESULTS_ROOT}/"
echo "============================================================"
