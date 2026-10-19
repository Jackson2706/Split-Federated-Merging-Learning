#!/usr/bin/env bash
# ============================================================================
# run_preflight_smoke.sh — E-HSFP journal campaign pre-flight smoke gate
#
# Per the governing campaign brief, Section 6 ("Before expensive final runs"):
#   1. run two-round smoke tests for every dataset/method family
# before any 5-seed frozen-protocol run is queued.
#
# Covers the 5 datasets in Section 8.1's main matched comparison, EXCLUDING
# ImageNet-1K (excluded from campaign scope by explicit user decision,
# 2026-07-23 — see docs/optimization_loop/DECISIONS.md):
#   CIFAR-10 (AlexNet), CIFAR-100 (ResNet-50/18), HAM10000 (ResNet-50),
#   ISIC-2018 (ResNet50-U-Net)
# x 6 method families each (federated, hierfl, hsfl, splitfl, hetero-sfl,
# h-sfp) = 24 smoke runs total.
#
# Each run is bounded to 2 epochs/rounds (--set epochs=2), just enough to
# confirm: config loads, dataset loads, one full client->edge->cloud round
# executes, no crash, no NaN. This is NOT a real result — do not read any
# accuracy/IoU value out of these logs as a reportable number.
#
# Resumable: skips any (dataset, method) pair already marked done. Safe to
# re-run after a partial failure.
#
# Usage:
#   ./run_preflight_smoke.sh
#   GPU_ID=0 EPOCHS=2 ./run_preflight_smoke.sh
#   DRY_RUN=1 ./run_preflight_smoke.sh   # print commands only
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-2}"
DRY_RUN="${DRY_RUN:-0}"
TIMEOUT_S="${TIMEOUT_S:-900}"

STATUS_DIR="docs/optimization_loop/logs/preflight_smoke"
MARKER_DIR="${STATUS_DIR}/.markers"
mkdir -p "$STATUS_DIR" "$MARKER_DIR"
PROGRESS_FILE="${STATUS_DIR}/PROGRESS.txt"
[[ -f "$PROGRESS_FILE" ]] || echo "Pre-flight smoke gate start $(date -Is)" > "$PROGRESS_FILE"

_ts() { date "+%Y-%m-%d %H:%M:%S"; }
log_info()  { echo "[$(_ts)] [INFO]  $*"; }
log_warn()  { echo "[$(_ts)] [WARN]  $*" >&2; }
log_error() { echo "[$(_ts)] [ERROR] $*" >&2; }

_TOTAL=0; _PASSED=0; _FAILED=0; _SKIPPED=0
_FAILED_LIST=()

# ── Method x dataset config map ────────────────────────────────────────────
# format: "task|dataset_label|method|config_path"
RUNS=(
  # CIFAR-10 (AlexNet) — classification
  "classification|cifar10|federated|configs/classification/federated/cifar_fedavg_alexnet.yaml"
  "classification|cifar10|hierfl|configs/classification/hierfl/cifar_hierfl_alexnet.yaml"
  "classification|cifar10|hsfl|configs/classification/hsfl/cifar_our_alexnet_5_10.yaml"
  "classification|cifar10|splitfl|configs/classification/splitfl/cifar_splitfed_alexnet.yaml"
  "classification|cifar10|hetero-sfl|configs/classification/hetero-sfl/cifar_heteroSFL_alexnet.yaml"
  "classification|cifar10|h-sfp|configs/classification/h-sfp/cifar_our_alexnet_5_10.yaml"

  # CIFAR-100 (ResNet-50/18) — classification
  "classification|cifar100|federated|configs/classification/federated/cifar_fedavg_resnet50.yaml"
  "classification|cifar100|hierfl|configs/classification/hierfl/cifar_hierfl_resnet50.yaml"
  "classification|cifar100|hsfl|configs/classification/hsfl/cifar_our_resnet50_5_10.yaml"
  "classification|cifar100|splitfl|configs/classification/splitfl/cifar_splitfed_resnet50.yaml"
  "classification|cifar100|hetero-sfl|configs/classification/hetero-sfl/cifar_heteroSFL_resnet18.yaml"
  "classification|cifar100|h-sfp|configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"

  # HAM10000 (ResNet-50) — classification
  "classification|ham10000|federated|configs/classification/federated/ham10000_fedavg_resnet50.yaml"
  "classification|ham10000|hierfl|configs/classification/hierfl/ham10000_hierfl_resnet50.yaml"
  "classification|ham10000|hsfl|configs/classification/hsfl/ham10000_our_resnet50_5_10.yaml"
  "classification|ham10000|splitfl|configs/classification/splitfl/ham10000_splitfed_resnet50.yaml"
  "classification|ham10000|hetero-sfl|configs/classification/hetero-sfl/ham10000_heteroSFL_resnet50.yaml"
  "classification|ham10000|h-sfp|configs/classification/h-sfp/ham10000_our_resnet50_5_10.yaml"

  # ISIC-2018 (ResNet50-U-Net) — segmentation
  "segmentation|isic2018|federated|configs/segmentation/federated/isic_fedavg_resnet50.yaml"
  "segmentation|isic2018|hierfl|configs/segmentation/hierfl/isic_hierfl_resnet50.yaml"
  "segmentation|isic2018|hsfl|configs/segmentation/hsfl/isic_our_resnet50_5_10.yaml"
  "segmentation|isic2018|splitfl|configs/segmentation/splitfl/isic_splitfed_resnet50.yaml"
  "segmentation|isic2018|hetero-sfl|configs/segmentation/hetero-sfl/isic_heteroSFL_resnet50.yaml"
  "segmentation|isic2018|h-sfp|configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml"

  # ImageNet-1K deliberately EXCLUDED — campaign-scope decision, 2026-07-23.
)

for entry in "${RUNS[@]}"; do
  IFS='|' read -r task dataset_label method cfg <<< "$entry"
  eid="smoke_${task}_${dataset_label}_${method}"
  marker="${MARKER_DIR}/${eid}.done"
  _TOTAL=$((_TOTAL + 1))

  if [[ -f "$marker" ]]; then
    _SKIPPED=$((_SKIPPED + 1))
    log_info "[SKIP] ${eid} (already done)"
    continue
  fi

  if [[ ! -f "$PROJECT_ROOT/$cfg" ]]; then
    _FAILED=$((_FAILED + 1))
    _FAILED_LIST+=("${eid} (config not found: ${cfg})")
    log_error "[MISSING CONFIG] ${eid}: ${cfg}"
    echo "$eid MISSING_CONFIG $(date -Is)" >> "$PROGRESS_FILE"
    continue
  fi

  log_file="${STATUS_DIR}/${eid}.log"
  cmd=(python main.py --task "$task" --method "$method" --cfg "$cfg"
       --set epochs="$EPOCHS" --set output_dir="outputs/preflight_smoke/${eid}")

  if [[ "$DRY_RUN" == "1" ]]; then
    log_info "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} ${cmd[*]}"
    _PASSED=$((_PASSED + 1))
    continue
  fi

  echo "========================================"
  log_info "[${_TOTAL}] ${eid}"
  log_info "  cfg=${cfg} epochs=${EPOCHS}"
  echo "========================================"

  if CUDA_VISIBLE_DEVICES="$GPU_ID" timeout "$TIMEOUT_S" "${cmd[@]}" > "$log_file" 2>&1; then
    _PASSED=$((_PASSED + 1))
    touch "$marker"
    log_info "[PASS] ${eid}"
    echo "$eid PASS $(date -Is)" >> "$PROGRESS_FILE"
  else
    exit_code=$?
    _FAILED=$((_FAILED + 1))
    _FAILED_LIST+=("${eid} (exit ${exit_code}, see ${log_file})")
    log_error "[FAIL] ${eid} exit=${exit_code}"
    echo "$eid FAIL exit=${exit_code} $(date -Is)" >> "$PROGRESS_FILE"
  fi
done

echo ""
echo "============================================================"
echo "  PRE-FLIGHT SMOKE GATE SUMMARY"
echo "============================================================"
echo "  Total:   ${_TOTAL}"
echo "  Passed:  ${_PASSED}"
echo "  Skipped: ${_SKIPPED}"
echo "  Failed:  ${_FAILED}"
echo "  Logs:    ${STATUS_DIR}/"
if [[ ${#_FAILED_LIST[@]} -gt 0 ]]; then
  echo ""
  echo "  Failed/missing:"
  for f in "${_FAILED_LIST[@]}"; do
    echo "    - ${f}"
  done
fi
echo "============================================================"

if [[ "$DRY_RUN" == "1" ]]; then
  # Dry runs preview commands only; never write the completion marker a real
  # run relies on, or a later real invocation could be mistaken for already
  # finished.
  exit 0
fi

if [[ $_FAILED -gt 0 ]]; then
  echo "ALL_DONE_WITH_FAILURES $(date -Is)" > "${STATUS_DIR}/DONE.marker"
  exit 1
else
  echo "ALL_DONE_CLEAN $(date -Is)" > "${STATUS_DIR}/DONE.marker"
  exit 0
fi
