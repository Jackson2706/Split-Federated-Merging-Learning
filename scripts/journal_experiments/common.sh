#!/usr/bin/env bash
# ============================================================================
# common.sh — Shared functions for E-HSFP journal experiment scripts
# ============================================================================

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="${PROJECT_ROOT}/results/journal"

# ── Defaults (overridable via env) ─────────────────────────────────────────
GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
WANDB_ARGS="${WANDB_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

# ── Counters ───────────────────────────────────────────────────────────────
_TOTAL=0
_PASSED=0
_FAILED=0
_SKIPPED=0
_FAILED_LIST=()

# ── Output dirs ────────────────────────────────────────────────────────────
ensure_results_dirs() {
    local subdirs=(convergence ablation dropout staleness intervals communication logs)
    for d in "${subdirs[@]}"; do
        mkdir -p "${RESULTS_ROOT}/${d}"
    done
}

# ── Logging ────────────────────────────────────────────────────────────────
_ts() { date "+%Y-%m-%d %H:%M:%S"; }

log_info()  { echo "[$(_ts)] [INFO]  $*"; }
log_warn()  { echo "[$(_ts)] [WARN]  $*" >&2; }
log_error() { echo "[$(_ts)] [ERROR] $*" >&2; }

# ── Marker helpers (resume support) ────────────────────────────────────────
_marker_dir() {
    echo "${RESULTS_ROOT}/.markers"
}

_marker_path() {
    local experiment_id="$1"
    echo "$(_marker_dir)/${experiment_id}"
}

is_done() {
    local experiment_id="$1"
    [[ -f "$(_marker_path "$experiment_id").done" ]]
}

mark_done() {
    local experiment_id="$1"
    mkdir -p "$(_marker_dir)"
    touch "$(_marker_path "$experiment_id").done"
    rm -f "$(_marker_path "$experiment_id").failed"
}

mark_failed() {
    local experiment_id="$1"
    mkdir -p "$(_marker_dir)"
    touch "$(_marker_path "$experiment_id").failed"
}

# Build a unique experiment ID from components
make_experiment_id() {
    # Usage: make_experiment_id task method cfg_basename [seed] [extra...]
    local parts=("$@")
    local id
    id=$(IFS=_; echo "${parts[*]}")
    # Sanitize: replace slashes, spaces, dots with underscores
    echo "${id//[\/. ]/_}"
}

# ── Check environment ──────────────────────────────────────────────────────
check_env() {
    cd "$PROJECT_ROOT"

    if ! command -v python &>/dev/null; then
        log_error "python not found in PATH"
        exit 1
    fi

    if ! python -c "import torch" &>/dev/null; then
        log_error "PyTorch not installed. Run: pip install -r requirements.txt"
        exit 1
    fi

    if [[ -n "$WANDB_ARGS" ]]; then
        if ! python -c "import wandb" &>/dev/null; then
            log_warn "wandb not installed but WANDB_ARGS set. Disabling W&B."
            WANDB_ARGS=""
        fi
    fi

    log_info "Environment OK (GPU_ID=${GPU_ID}, seeds=[${SEEDS}])"
}

# ── Core runner ────────────────────────────────────────────────────────────
# run_one EXPERIMENT_ID TASK METHOD CFG SEED [ABLATION] [EXTRA_ARGS...]
#
# Runs a single experiment with resume support.
# Returns 0 on success, 1 on failure, 2 if skipped (already done).
run_one() {
    local experiment_id="$1"
    local task="$2"
    local method="$3"
    local cfg="$4"
    local seed="$5"
    shift 5
    local ablation="${1:-}"
    shift  # always consume the ablation slot (may be empty)
    local extra_args=("$@")

    _TOTAL=$((_TOTAL + 1))

    # Resume: skip if already done
    if is_done "$experiment_id"; then
        _SKIPPED=$((_SKIPPED + 1))
        log_info "[SKIP] ${experiment_id} (already done)"
        return 2
    fi

    # Check config exists
    local cfg_path="${PROJECT_ROOT}/${cfg}"
    if [[ ! -f "$cfg_path" ]]; then
        _SKIPPED=$((_SKIPPED + 1))
        log_warn "[SKIP] Config not found: ${cfg}"
        return 2
    fi

    local log_file="${RESULTS_ROOT}/logs/${experiment_id}.log"
    mkdir -p "$(dirname "$log_file")"

    echo "========================================"
    log_info "[${_TOTAL}] ${experiment_id}"
    log_info "  task=${task} method=${method} seed=${seed} ablation=${ablation:-none}"
    log_info "  cfg=${cfg}"
    echo "========================================"

    if [[ "$DRY_RUN" == "1" ]]; then
        log_info "[DRY-RUN] Would run: python main.py --task ${task} --method ${method} --cfg ${cfg} --seed ${seed} ${ablation:+--ablation $ablation} ${extra_args[*]:-}"
        _PASSED=$((_PASSED + 1))
        mark_done "$experiment_id"
        return 0
    fi

    local cmd=(
        python main.py
        --task "$task"
        --method "$method"
        --cfg "$cfg"
        --seed "$seed"
    )
    [[ -n "$ablation" ]] && cmd+=(--ablation "$ablation")
    [[ -n "$WANDB_ARGS" ]] && cmd+=($WANDB_ARGS)
    [[ ${#extra_args[@]} -gt 0 ]] && cmd+=("${extra_args[@]}")

    cd "$PROJECT_ROOT"
    if CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" 2>&1 | tee "$log_file"; then
        _PASSED=$((_PASSED + 1))
        mark_done "$experiment_id"
        log_info "[PASS] ${experiment_id}"
        return 0
    else
        _FAILED=$((_FAILED + 1))
        _FAILED_LIST+=("$experiment_id")
        mark_failed "$experiment_id"
        log_error "[FAIL] ${experiment_id}"
        return 1
    fi
}

# ── Convenience: run across seeds ──────────────────────────────────────────
# run_seeds PREFIX TASK METHOD CFG [ABLATION] [EXTRA_ARGS...]
run_seeds() {
    local prefix="$1"
    local task="$2"
    local method="$3"
    local cfg="$4"
    shift 4
    local ablation="${1:-}"
    [[ -n "$ablation" ]] && shift || true
    local extra_args=("$@")

    local cfg_base
    cfg_base="$(basename "$cfg" .yaml)"

    for seed in $SEEDS; do
        local eid
        eid=$(make_experiment_id "$prefix" "$task" "$method" "$cfg_base" "s${seed}")
        run_one "$eid" "$task" "$method" "$cfg" "$seed" "$ablation" "${extra_args[@]}" || true
    done
}

# ── Summary ────────────────────────────────────────────────────────────────
print_summary() {
    local label="${1:-Experiment}"
    echo ""
    echo "============================================================"
    echo "  ${label} SUMMARY"
    echo "============================================================"
    echo "  Total:   ${_TOTAL}"
    echo "  Passed:  ${_PASSED}"
    echo "  Skipped: ${_SKIPPED}"
    echo "  Failed:  ${_FAILED}"
    echo "  Results: ${RESULTS_ROOT}/"
    if [[ ${#_FAILED_LIST[@]} -gt 0 ]]; then
        echo ""
        echo "  Failed experiments:"
        for f in "${_FAILED_LIST[@]}"; do
            echo "    - ${f}"
        done
    fi
    echo "============================================================"
}

# ── Reset markers (for re-running) ────────────────────────────────────────
reset_markers() {
    local pattern="${1:-*}"
    local marker_dir
    marker_dir="$(_marker_dir)"
    if [[ -d "$marker_dir" ]]; then
        find "$marker_dir" -name "${pattern}.done" -delete 2>/dev/null || true
        find "$marker_dir" -name "${pattern}.failed" -delete 2>/dev/null || true
        log_info "Cleared markers matching '${pattern}'"
    fi
}
