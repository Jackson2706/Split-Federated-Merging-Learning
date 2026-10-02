#!/usr/bin/env bash
# ============================================================================
# Experiment 5: Packing/unpacking & synthesis runtime profiling.
# Runs H-SFP with profile=true (fine-grained CUDA-synchronized phase timers) and
# SplitFed for a total-time comparison. Wall-clock for each run is appended to
# results/camera_ready/profiling/wallclock.csv; H-SFP per-phase breakdown is
# written by the in-pipeline profiler to profiling/profile_<tag>_*.{csv,json}.
# Addresses reviewer concern: packing/unpacking computation overhead.
#
# Smoke:  SMOKE=1 ./scripts/camera_ready/run_profiling.sh
# Full :  ./scripts/camera_ready/run_profiling.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASET="${DATASET:-cifar100}"
PROF_DIR="${RESULTS_ROOT}/profiling"
mkdir -p "$PROF_DIR"
WALL="${PROF_DIR}/wallclock.csv"
[[ -f "$WALL" ]] || echo "method,seed,dataset,smoke,wall_seconds" > "$WALL"

seed="${PROF_SEED:-0}"

timed_run() {
    local exp_id="$1" method="$2"; shift 2
    local extra=("$@")
    read -r flag cfg < <(cfg_for "$method")
    local logf="${LOGS_DIR}/${exp_id}.log"
    local cmd=(python main.py --task classification --method "$flag" --cfg "$cfg"
               --seed "$seed" --set "dataset=${DATASET}" "${extra[@]}")
    if [[ -n "${EXTRA_SET:-}" ]]; then for kv in $EXTRA_SET; do cmd+=(--set "$kv"); done; fi
    log_info "RUN  $exp_id : ${cmd[*]}"
    if [[ "$DRY_RUN" == "1" ]]; then echo "DRY_RUN: ${cmd[*]}" | tee "$logf"; return 0; fi
    cd "$PROJECT_ROOT"
    local t0 t1
    t0=$(date +%s.%N)
    "${cmd[@]}" >"$logf" 2>&1 || log_error "run failed: $exp_id (see $logf)"
    t1=$(date +%s.%N)
    echo "${method},${seed},${DATASET},${SMOKE},$(echo "$t1 - $t0" | bc)" >> "$WALL"
}

# H-SFP with fine-grained profiling enabled.
timed_run "profiling_h-sfp_s${seed}" "h-sfp" \
    --set "profile=true" --set "profile_tag=hsfp_${DATASET}"

# SplitFed total-time reference.
timed_run "profiling_splitfl_s${seed}" "splitfl"

log_info "Profiling complete. Wall-clock: $WALL"
log_info "H-SFP phase breakdown: ${PROF_DIR}/profile_hsfp_${DATASET}_summary.json"
log_info "Table: python scripts/camera_ready/make_tables.py --only profiling"
