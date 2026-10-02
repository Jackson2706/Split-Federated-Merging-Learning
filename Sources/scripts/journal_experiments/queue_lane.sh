#!/usr/bin/env bash
# =============================================================================
# queue_lane.sh — drain a prioritized run queue through journal_run.sh's run_exp
#
#   scripts/journal_experiments/queue_lane.sh <queue_file> [lane_name]
#
# A queue line is   stage|eid|task|dataset|method|cfg|seed|ablation|extra...
# (as printed by `EMIT_QUEUE=1 ./journal_run.sh <stage>`). Lines are taken in
# file order; '#' lines and blanks are ignored. The file is re-read after every
# run, so items can be appended or re-prioritized while lanes are running.
#
# Several lanes may drain the same file concurrently: run_exp takes an atomic
# per-run lock (results/journal_v1/.markers/<eid>.lock), skips runs that are
# .done, and a lock whose owner PID is dead is taken over. A run that fails is
# marked .failed and is NOT retried by the lane (bounded retries — inspect,
# then `./journal_run.sh retry_failed`).
#
# Honors every journal_run.sh environment variable (OUT_ROOT, ROUNDS, GPU_ID,
# MIN_FREE_GB, ...). STOP file: touch results/journal_v1/.markers/STOP_<lane>
# to let the lane exit after its current run; the STOP_<lane> file is consumed,
# so the cron supervisor (start_lanes.sh every 10 min) restarts the lane with the
# current lane_env.sh. STOP_ALL is persistent: lanes exit and stay down until it
# is removed. To retire one lane permanently, remove it from queue/LANES.
# =============================================================================
set -uo pipefail
QUEUE="${1:?usage: queue_lane.sh <queue_file> [lane_name]}"
LANE="${2:-lane0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
[[ "$QUEUE" = /* ]] || QUEUE="$ROOT/$QUEUE"

# shellcheck source=/dev/null
source "$ROOT/journal_run.sh"   # helpers only; main() is not invoked when sourced
check_env

next_item() {
    local line eid
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        eid="$(cut -d'|' -f2 <<< "$line")"
        [[ -f "${MARKER_DIR}/${eid}.done" || -f "${MARKER_DIR}/${eid}.failed" ]] && continue
        if [[ -d "${MARKER_DIR}/${eid}.lock" ]]; then
            local owner; owner="$(cat "${MARKER_DIR}/${eid}.lock/pid" 2>/dev/null)"
            [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null && continue
        fi
        echo "$line"; return 0
    done < "$QUEUE"
    return 1
}

log_info "lane ${LANE} pid $$ draining ${QUEUE}"
while true; do
    if [[ -f "${MARKER_DIR}/STOP_ALL" ]]; then
        log_info "lane ${LANE}: STOP_ALL present — exiting"; break
    fi
    if [[ -f "${MARKER_DIR}/STOP_${LANE}" ]]; then
        rm -f "${MARKER_DIR}/STOP_${LANE}"
        log_info "lane ${LANE}: STOP_${LANE} consumed — exiting (supervisor restarts it)"; break
    fi
    # Pick up environment edits (e.g. DIAG) between runs.
    # shellcheck source=/dev/null
    [[ -f "$ROOT/results/journal_v1/queue/lane_env.sh" ]] && source "$ROOT/results/journal_v1/queue/lane_env.sh"
    item="$(next_item)" || { log_info "lane ${LANE}: queue drained"; break; }
    IFS='|' read -r -a F <<< "$item"
    extra=()
    for ((i = 8; i < ${#F[@]}; i++)); do [[ -n "${F[$i]}" ]] && extra+=("${F[$i]}"); done
    run_exp "${F[0]}" "${F[1]}" "${F[2]}" "${F[3]}" "${F[4]}" "${F[5]}" "${F[6]}" "${F[7]:-}" "${extra[@]}"
    sleep 5   # let the GPU allocator settle before the next run
done
log_info "lane ${LANE}: passed=${_PASSED} failed=${_FAILED} skipped=${_SKIPPED}"
