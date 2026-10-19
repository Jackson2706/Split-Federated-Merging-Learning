#!/usr/bin/env bash
# ============================================================================
# common.sh — Shared helpers for camera-ready experiment launchers.
# Mirrors scripts/journal_experiments/common.sh (markers/resume/logging).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="${PROJECT_ROOT}/results/camera_ready"
LOGS_DIR="${RESULTS_ROOT}/logs"
MARKERS_DIR="${RESULTS_ROOT}/.markers"

# Defaults (override via env)
SMOKE="${SMOKE:-0}"                 # 1 => use tiny smoke configs + 1 seed
SEEDS="${SEEDS:-0 1 2 3 4}"
[[ "$SMOKE" == "1" ]] && SEEDS="${SEEDS_SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESET="${RESET:-0}"                 # 1 => ignore .done markers and re-run
WANDB_ARGS="${WANDB_ARGS:-}"

mkdir -p "$LOGS_DIR" "$MARKERS_DIR"

_ts() { date "+%Y-%m-%d %H:%M:%S"; }
log_info()  { echo "[$(_ts)] [INFO]  $*"; }
log_warn()  { echo "[$(_ts)] [WARN]  $*" >&2; }
log_error() { echo "[$(_ts)] [ERROR] $*" >&2; }

# Map a method name to its (full | smoke) base config and main.py --method flag.
# Usage: cfg_for <method>  -> echoes "<method_flag> <cfg_path>"
cfg_for() {
    local method="$1" cfg flag
    case "$method" in
        h-sfp|hsfp)   flag="h-sfp"
            cfg="configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"
            [[ "$SMOKE" == "1" ]] && cfg="configs/camera_ready/smoke/hsfp_smoke.yaml" ;;
        hsfl)         flag="hsfl"
            cfg="configs/classification/hsfl/cifar_our_resnet50_5_10.yaml"
            [[ "$SMOKE" == "1" ]] && cfg="configs/camera_ready/smoke/hsfl_smoke.yaml" ;;
        hierfl)       flag="hierfl"
            cfg="configs/classification/hierfl/cifar_hierfl_resnet50.yaml"
            [[ "$SMOKE" == "1" ]] && cfg="configs/camera_ready/smoke/hierfl_smoke.yaml" ;;
        splitfl|splitfed) flag="splitfl"
            cfg="configs/classification/splitfl/cifar_splitfed_resnet50.yaml"
            [[ "$SMOKE" == "1" ]] && cfg="configs/camera_ready/smoke/splitfl_smoke.yaml" ;;
        federated|fedavg) flag="federated"
            cfg="configs/classification/federated/cifar_fedavg_resnet50.yaml"
            [[ "$SMOKE" == "1" ]] && cfg="configs/camera_ready/smoke/federated_smoke.yaml" ;;
        *) log_error "unknown method: $method"; return 1 ;;
    esac
    echo "$flag $cfg"
}

# Run one experiment with resume markers and logging.
# Usage: run_main <exp_id> <method> <seed> [extra main.py args...]
run_main() {
    local exp_id="$1"; shift
    local method="$1"; shift
    local seed="$1"; shift
    local extra=("$@")

    local done_marker="${MARKERS_DIR}/${exp_id}.done"
    if [[ "$RESET" != "1" && -f "$done_marker" ]]; then
        log_info "SKIP (done): $exp_id"
        return 0
    fi

    read -r flag cfg < <(cfg_for "$method")
    local logf="${LOGS_DIR}/${exp_id}.log"
    local cmd=(python main.py --task classification --method "$flag" --cfg "$cfg"
               --seed "$seed" "${extra[@]}")
    # Global budget/overrides applied uniformly to every run, e.g.
    #   EXTRA_SET="epochs=30 ssl_epochs_client=3 t1=3 t2=6"
    if [[ -n "${EXTRA_SET:-}" ]]; then for kv in $EXTRA_SET; do cmd+=(--set "$kv"); done; fi
    [[ -n "$WANDB_ARGS" ]] && cmd+=($WANDB_ARGS)

    log_info "RUN  $exp_id"
    log_info "  cmd: ${cmd[*]}"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY_RUN: ${cmd[*]}" | tee "$logf"
        return 0
    fi

    cd "$PROJECT_ROOT"
    if "${cmd[@]}" >"$logf" 2>&1; then
        touch "$done_marker"; rm -f "${MARKERS_DIR}/${exp_id}.failed"
        log_info "DONE $exp_id"
    else
        touch "${MARKERS_DIR}/${exp_id}.failed"
        log_error "FAILED $exp_id (see $logf)"
    fi
}
