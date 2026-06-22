#!/usr/bin/env bash
# ============================================================================
# Master launcher for ALL camera-ready experiments.
#
# Smoke (fast end-to-end sanity, 1 seed, tiny configs):
#   SMOKE=1 ./scripts/camera_ready/run_all_camera_ready.sh
# Full:
#   ./scripts/camera_ready/run_all_camera_ready.sh
#
# Honors: SMOKE, SEEDS, METHODS, DRY_RUN, RESET (see common.sh).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"

CFG="configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"
TAG="cifar100"
COV_EXTRA=""
LSTAT_EXTRA=""
INV_EXTRA=""
if [[ "$SMOKE" == "1" ]]; then
    CFG="configs/camera_ready/smoke/hsfp_smoke.yaml"
    TAG="smoke"
    COV_EXTRA="--max-classes 10 --probe-per-class 30 --syn-per-class 30 --clf-epochs 3"
    LSTAT_EXTRA="--num-clients 2 --batch 64"
    INV_EXTRA="--num-samples 3 --iters 80"
fi

log_info "=== [1/7] Hierarchical heterogeneity ==="
bash "${HERE}/run_hetero.sh"
log_info "=== partition diagnostics ==="
python "${HERE}/run_partition_diagnostics.py" --cfg "$CFG" --tag "$TAG"

log_info "=== [2/7] Partial participation ==="
bash "${HERE}/run_partial.sh"

log_info "=== [3/7] Covariance ablation ==="
python "${HERE}/run_covariance.py" --cfg "$CFG" --tag "$TAG" $COV_EXTRA

log_info "=== [4/7] Lstat estimation ==="
python "${HERE}/run_lstat.py" --cfg "$CFG" --tag "$TAG" $LSTAT_EXTRA

log_info "=== [5/7] Runtime profiling ==="
bash "${HERE}/run_profiling.sh"

log_info "=== [6/7] Inversion / exposure ==="
python "${HERE}/run_inversion.py" --cfg "$CFG" --tag "$TAG" $INV_EXTRA

log_info "=== [7/7] Baseline fairness summary ==="
python "${HERE}/gen_fairness.py"

log_info "=== Aggregate + tables ==="
python "${HERE}/collect_camera_ready.py"
python "${HERE}/make_tables.py"

log_info "All camera-ready experiments dispatched. Outputs under results/camera_ready/."
