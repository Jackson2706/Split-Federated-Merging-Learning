#!/usr/bin/env bash
# =============================================================================
# journal_run.sh — E-HSFP journal-extension campaign runner  (H-SFP repo)
#
# ONE self-contained, resumable launcher for every experiment the journal
# extension still needs. Run it manually, stage by stage, on the host GPU.
#
#   ./journal_run.sh                  # print the plan + wall-time estimates, run NOTHING
#   ./journal_run.sh plan             # same as above
#   ./journal_run.sh s1_parity        # run one stage
#   ./journal_run.sh s1_parity s2_main
#   ./journal_run.sh core             # parity + main table + ablation only
#   ./journal_run.sh all              # every stage, in dependency order
#   ./journal_run.sh runnable         # what can and cannot run, and why
#   ./journal_run.sh status           # what is done / failed / pending so far
#   ./journal_run.sh coverage         # what prior work already covers, if anything
#   ./journal_run.sh retry_failed     # clear .failed markers so they re-run
#
# It does NOT source scripts/journal_experiments/common.sh on purpose — see
# "KNOWN TRAPS" below; that helper silently skips sweeps. This file is
# standalone and depends only on `python`, `main.py`, and the configs.
#
# -----------------------------------------------------------------------------
# WHY A NEW RESULTS ROOT (results/journal_v1, not results/journal)
# -----------------------------------------------------------------------------
# results/journal/.markers holds 40 `.done` markers from the 2026-07-09..14
# seed-0 pass. Those runs predate every fix in Section 5 of
# ALL_SIMULATION_RESULTS.md (fp16 NaN collapse, real macro-F1, unified
# communication accounting, prediction artifacts, history.csv). Reusing that
# marker dir would silently SKIP every seed-0 run in this campaign and mix
# pre-fix numbers into post-fix means. This script therefore writes to a fresh
# root and never reads the old markers.
#
# -----------------------------------------------------------------------------
# KNOWN TRAPS THIS SCRIPT WORKS AROUND (all verified in the code, 2026-07-31)
# -----------------------------------------------------------------------------
# 1. `prepare_run_dir()` (ehsfp/integrity.py:100) RAISES FileExistsError if
#    base/<architecture_id>/<config_hash> already exists. A run that is killed
#    or crashes leaves that directory behind, so the naive retry dies instantly
#    with "Refusing to overwrite existing run directory". We detect that exact
#    message in the log, archive the stale dir to <dir>.stale-<ts>, and retry
#    once automatically.
# 2. scripts/journal_experiments/run_dropout_staleness.sh passes an ABSOLUTE
#    temp-config path into run_one(), which then tests "${PROJECT_ROOT}/${cfg}"
#    -> never a file -> every dropout/staleness run is SKIPPED, not run. That is
#    why results/journal/dropout and .../staleness are empty. We use main.py's
#    own `--set key=value` instead, which writes its temp YAML next to the
#    original config so the `base:` chain still resolves.
# 3. An `--ablation` preset has HIGHEST precedence and overwrites the six
#    on/off keys it names (ehsfp/config.py:ABLATION_PRESETS). Scalar knobs
#    (prototype_dropout_rate, max_prototype_age, memory_size, ...) are NOT in
#    any preset, so `--set` survives alongside `--ablation`. But
#    `aggregation_mode` IS in the presets — so the aggregation-rule stage
#    (s9) deliberately passes NO --ablation and sets the flags by hand.
# 4. `--set mid_server=[5]` does not work: main.py's _coerce() leaves "[5]" a
#    string and the runner then indexes a string. Any list-valued key goes
#    through write_child_cfg() below instead.
# 5. `mid_server[0]` and `num_edges` are two independently-read keys
#    (JOURNAL_SIMULATION_RESULTS.md 3.3). The scalability stage always sets
#    both together.
#
# -----------------------------------------------------------------------------
# PRIOR WORK — what is already done and therefore NOT scheduled here
# -----------------------------------------------------------------------------
# The ECCV 2026 rebuttal / camera-ready campaign (2026-06-22..26, 65 runs,
# ~48 GPU-hours) is inventoried in docs/ECCV_REBUTTAL_SIMULATIONS.md. Summary of
# what that buys this campaign — `./journal_run.sh coverage` reprints it:
#
#   REUSED, never rerun here: all 7 analysis experiments (covariance ablation,
#   Lstat sensitivity, phase profiling, feature inversion, fairness manifest,
#   partition diagnostics). They answer ECCV reviewer questions, are not part of
#   the journal's numeric matrix, and have no stage below.
#
#   NOT reusable: the 63 completed hetero + partial training runs. They ran at
#   30 rounds / seed 0 / ssl_epochs=3 / local_ep=3 / t1=3,t2=6 — not the frozen
#   protocol — on pre-fix code, with a `test_f1` column that actually holds
#   top-1 accuracy for h-sfp/hsfl/splitfl, and a collapsed H-SFP arm (3.5-6.5%).
#   Pooling them with frozen-protocol runs is indefensible, so nothing below is
#   skipped on their account.
#
#   STILL MISSING from ECCV itself: 2 runs that died of CUDA OOM at frac=1.0 IID
#   (partial_federated / partial_hierfl). Stage s12_eccv_recover reruns exactly
#   those two, at the ECCV protocol, with local_bs halved. It is the only stage
#   here that deliberately does NOT use the frozen protocol.
#
#   The ECCV set also covers partial participation (frac sweep), a dimension the
#   journal brief does not request — so this script has no stage for it.
#
# -----------------------------------------------------------------------------
# HONEST SCOPE LIMITS — read before trusting any output
# -----------------------------------------------------------------------------
# * Residual prototype generator is a NO-OP. Both hierarchies hard-code
#   `self.residual_generator = None` (classification/H-SFP/hierarchy.py:471,
#   segmentation/H-SFP/hierarchy.py:328) regardless of
#   use_residual_generator. So `full_e_hsfp` runs 5 of its 6 advertised
#   components. Stage s13_synthesis is therefore GATED OFF and will refuse to
#   run until that is wired in. Do not claim a prototype-synthesis ablation.
#   The `full_e_hsfp` row is honest as "E-HSFP minus prototype synthesis".
# * The OTHER five components are verified genuinely wired in BOTH tasks
#   (audited 2026-08-02) — each has a real construction site and a real
#   application site, so no sweep below has silently identical arms:
#     episodic memory        cls hierarchy.py:427   seg hierarchy.py:290
#     prototype dropout      cls :456 applied :825,:972
#                            seg :315 applied :569,:687   (client + edge)
#     PRC loss               cls :896,:1026         seg :628,:725
#     serverless simulation  cls :465,:1251+        seg :323,:958+
#     reliability aggregation / sample_count_weighted: both, via
#       ehsfp/aggregation.py with a real support_map from the callers.
#   This is why the ISIC-2018 arms of s5/s6/s9 are worth the GPU time.
# * Segmentation has NO Dirichlet partitioner (`partition_from_config` is
#   imported by all 6 classification methods and by none of the segmentation
#   ones). The non-IID stage is classification-only; ISIC-2018 non-IID cells
#   cannot be produced without new code.
# * Segmentation H-SFP ignores `output_dir` (segmentation/H-SFP/runner.py:41
#   hard-codes Figure/data/runs). Its artifacts land under the method dir, not
#   outputs/journal_v1/. Runs still do not collide, because the run dir is keyed
#   by resolved_config_hash and the seed is part of the config.
# * ImageNet-1K is out of scope by the user's 2026-07-23 decision.
# * `sample_count_weighted` IS implemented (ehsfp/aggregation.py:112) and the
#   callers do pass a real support_map — Section 20 item 11 of
#   JOURNAL_SIMULATION_RESULTS.md is stale on this point.
#
# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
#   GPU_ID=0            CUDA device
#   SEEDS="0 1 2 3 4"   frozen protocol seeds (JOURNAL_SIMULATION_RESULTS 3.1)
#   ROUNDS=60           global rounds (`epochs`). 60 = every full-scale run in
#                       this repo so far. The frozen protocol table nominally
#                       says 200 for classification; ROUNDS=200 switches, at
#                       ~3.3x the wall time. NEVER mix budgets inside one table
#                       — the value is recorded per row in the manifest.
#   DRY_RUN=1           print commands, touch no markers, run nothing
#   SKIP_SEG=1          skip every segmentation run
#   TIMEOUT_S=0         per-run timeout in seconds (0 = none)
#   MIN_FREE_GB=15      abort before a run if the disk is below this
#   FORCE=1             ignore the s13 implementation gate (do not use)
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT" || exit 1

GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
ROUNDS="${ROUNDS:-60}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SEG="${SKIP_SEG:-0}"
TIMEOUT_S="${TIMEOUT_S:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-15}"
FORCE="${FORCE:-0}"

RESULTS_ROOT="results/journal_v1"
MARKER_DIR="${RESULTS_ROOT}/.markers"
LOG_DIR="${RESULTS_ROOT}/logs"
# Run artifacts can live on another filesystem (the root disk was at 99%).
# Default unchanged; the completion campaign sets
# OUT_ROOT=/media/jackson/Data/ehsfp_campaign/outputs_journal_v1, which holds a
# checksum-verified copy of every earlier outputs/journal_v1 run.
OUT_ROOT="${OUT_ROOT:-outputs/journal_v1}"
MANIFEST="${RESULTS_ROOT}/manifest.csv"

_TOTAL=0; _PASSED=0; _FAILED=0; _SKIPPED=0
_FAILED_LIST=()
_PLAN_RUNS=0; _PLAN_SECONDS=0
PLAN_ONLY=0
EMIT_QUEUE="${EMIT_QUEUE:-0}"

# ── logging ──────────────────────────────────────────────────────────────────
_ts()       { date "+%Y-%m-%d %H:%M:%S"; }
log_info()  { echo "[$(_ts)] [INFO]  $*"; }
log_warn()  { echo "[$(_ts)] [WARN]  $*" >&2; }
log_error() { echo "[$(_ts)] [ERROR] $*" >&2; }
log_stage() { echo; echo "############################################################"; \
              echo "#  $*"; echo "############################################################"; }

hms() {  # seconds -> "12h 34m"
    local s=${1%.*}
    printf '%dh %02dm' $((s / 3600)) $(((s % 3600) / 60))
}

# ── rough per-run cost model (seconds at 60 rounds) ──────────────────────────
# Measured from results/journal/logs/*.log where available, extrapolated from
# the 10-round proxy runs otherwise. Planning aid only — not a result.
est_s() {
    local task="$1" dataset="$2" method="$3" base
    case "${task}:${dataset}:${method}" in
        classification:cifar10:h-sfp)        base=4450 ;;
        classification:cifar10:federated)    base=2500 ;;
        classification:cifar10:hierfl)       base=2000 ;;
        classification:cifar10:hsfl)         base=900  ;;
        classification:cifar10:splitfl)      base=1100 ;;
        classification:cifar10:hetero-sfl)   base=1200 ;;
        classification:cifar100:h-sfp)       base=24200 ;;
        classification:cifar100:federated)   base=3450 ;;
        classification:cifar100:hierfl)      base=2570 ;;
        classification:cifar100:hsfl)        base=1160 ;;
        classification:cifar100:splitfl)     base=1500 ;;
        classification:cifar100:hetero-sfl)  base=1500 ;;
        classification:ham10000:h-sfp)       base=11100 ;;
        classification:ham10000:federated)   base=4700 ;;
        classification:ham10000:hierfl)      base=10100 ;;
        classification:ham10000:hsfl)        base=2150 ;;
        classification:ham10000:splitfl)     base=3570 ;;
        classification:ham10000:hetero-sfl)  base=3000 ;;
        segmentation:isic2018:h-sfp)         base=20200 ;;
        segmentation:isic2018:federated)     base=8000 ;;
        segmentation:isic2018:hierfl)        base=8000 ;;
        segmentation:isic2018:hsfl)          base=6000 ;;
        segmentation:isic2018:splitfl)       base=6000 ;;
        segmentation:isic2018:hetero-sfl)    base=6000 ;;
        *)                                   base=6000 ;;
    esac
    echo $(( base * ROUNDS / 60 ))
}

# ── guards ───────────────────────────────────────────────────────────────────
free_gb() { df -BG --output=avail "${1:-$PROJECT_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9'; }
# The disk that actually receives checkpoints/predictions.
out_free_gb() { mkdir -p "$OUT_ROOT" 2>/dev/null; free_gb "$OUT_ROOT"; }
# Logs/markers stay in the repo; they are small, so only a low floor applies.
MIN_REPO_FREE_GB="${MIN_REPO_FREE_GB:-2}"

check_env() {
    command -v python >/dev/null 2>&1 || { log_error "python not on PATH"; exit 1; }
    python -c "import torch, yaml" 2>/dev/null || {
        log_error "PyTorch/PyYAML missing — pip install -r requirements.txt"; exit 1; }
    if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        log_warn "CUDA not available to torch — runs will be CPU-bound and effectively never finish."
    fi
    local avail repo_avail; avail="$(out_free_gb)"; repo_avail="$(free_gb)"
    log_info "env ok | gpu=${GPU_ID} seeds=[${SEEDS}] rounds=${ROUNDS} out_root=${OUT_ROOT} out_free=${avail}G repo_free=${repo_avail}G"
    if [[ -n "$avail" && "$avail" -lt "$MIN_FREE_GB" ]]; then
        log_error "only ${avail}G free under ${OUT_ROOT}, need >= ${MIN_FREE_GB}G. Free space or lower MIN_FREE_GB."
        exit 1
    fi
    if [[ -n "$repo_avail" && "$repo_avail" -lt "$MIN_REPO_FREE_GB" ]]; then
        log_error "only ${repo_avail}G free on the repo disk (logs/markers), need >= ${MIN_REPO_FREE_GB}G."
        exit 1
    fi
    mkdir -p "$MARKER_DIR" "$LOG_DIR" "$OUT_ROOT"
    [[ -f "$MANIFEST" ]] || echo \
"start_ts,end_ts,runtime_s,stage,experiment_id,task,dataset,method,ablation,seed,rounds,cfg,overrides,exit_code,status,log_path" \
        > "$MANIFEST"
}

# Write a temp child config next to the original, with extra keys applied.
# Needed only for list-valued keys that `--set` cannot express (trap 4).
# Usage: write_child_cfg <cfg> <tag> "key: value" ["key: value" ...] -> prints path
write_child_cfg() {
    local cfg="$1" tag="$2"; shift 2
    local out="${cfg%.yaml}__jrun_${tag}.yaml"
    python - "$cfg" "$out" "$@" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    d = yaml.safe_load(f) or {}
for kv in sys.argv[3:]:
    k, v = kv.split(":", 1)
    d[k.strip()] = yaml.safe_load(v)
with open(dst, "w") as f:
    yaml.dump(d, f, default_flow_style=False)
PY
    echo "$out"
}

# ── the core runner ──────────────────────────────────────────────────────────
# run_exp <stage> <eid> <task> <dataset> <method> <cfg> <seed> <ablation|""> [--set k=v ...]
run_exp() {
    local stage="$1" eid="$2" task="$3" dataset="$4" method="$5" cfg="$6" seed="$7" ablation="$8"
    shift 8
    local extra=("$@")

    if [[ "$EMIT_QUEUE" == "1" ]]; then
        # One queue line per not-yet-done run: stage|eid|task|dataset|method|cfg|seed|ablation|extra...
        [[ "$SKIP_SEG" == "1" && "$task" == "segmentation" ]] && return 0
        [[ -f "${MARKER_DIR}/${eid}.done" ]] && return 0
        local IFS='|'
        echo "${stage}|${eid}|${task}|${dataset}|${method}|${cfg}|${seed}|${ablation}|${extra[*]}"
        return 0
    fi
    if [[ "$PLAN_ONLY" == "1" ]]; then
        [[ "$SKIP_SEG" == "1" && "$task" == "segmentation" ]] && return 0
        [[ -f "${MARKER_DIR}/${eid}.done" ]] && return 0
        _PLAN_RUNS=$((_PLAN_RUNS + 1))
        _PLAN_SECONDS=$((_PLAN_SECONDS + $(est_s "$task" "$dataset" "$method")))
        return 0
    fi

    _TOTAL=$((_TOTAL + 1))

    if [[ "$SKIP_SEG" == "1" && "$task" == "segmentation" ]]; then
        _SKIPPED=$((_SKIPPED + 1)); log_info "[SKIP] ${eid} (SKIP_SEG=1)"; return 2
    fi
    if [[ -f "${MARKER_DIR}/${eid}.done" ]]; then
        _SKIPPED=$((_SKIPPED + 1)); log_info "[SKIP] ${eid} (already done)"; return 2
    fi
    if [[ ! -f "$cfg" ]]; then
        _SKIPPED=$((_SKIPPED + 1)); log_warn "[SKIP] ${eid} — config missing: ${cfg}"; return 2
    fi
    # Run lock: several GPU lanes may drain the same queue. mkdir is atomic.
    # A lock whose owner PID is gone (host reboot, killed lane) is stale.
    local lock="${MARKER_DIR}/${eid}.lock"
    if [[ "$DRY_RUN" != "1" ]]; then
        if ! mkdir "$lock" 2>/dev/null; then
            local owner; owner="$(cat "${lock}/pid" 2>/dev/null)"
            if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
                _SKIPPED=$((_SKIPPED + 1)); log_info "[SKIP] ${eid} (running in lane pid ${owner})"; return 2
            fi
            log_warn "stale lock for ${eid} (owner ${owner:-unknown} not alive) — taking it over"
        fi
        echo "$$" > "${lock}/pid"; date -Is > "${lock}/started"
    fi

    local out_dir="${OUT_ROOT}/${eid}"
    local log_file="${LOG_DIR}/${eid}.log"
    # Never overwrite an earlier attempt's log (e.g. a run killed by a host
    # reboot): keep it as evidence next to the new one.
    if [[ "$DRY_RUN" != "1" && -f "$log_file" && ! -f "${MARKER_DIR}/${eid}.done" ]]; then
        mv "$log_file" "${log_file%.log}.attempt-$(date -r "$log_file" +%Y%m%dT%H%M%S).log"
    fi
    local cmd=(python main.py --task "$task" --method "$method" --cfg "$cfg"
               --seed "$seed" --set "epochs=${ROUNDS}" --set "output_dir=${out_dir}")
    [[ -n "$ablation" ]] && cmd+=(--ablation "$ablation")
    [[ ${#extra[@]} -gt 0 ]] && cmd+=("${extra[@]}")
    # Per-round oracle drift / reliability-weight / tier-memory diagnostics
    # (classification H-SFP only; diagnostics never feed back into training).
    if [[ "${DIAG:-0}" == "1" && "$method" == "h-sfp" && "$task" == "classification" ]]; then
        cmd+=(--set "diagnostics_dir=${out_dir}/diagnostics")
    fi

    local overrides="epochs=${ROUNDS}"
    [[ ${#extra[@]} -gt 0 ]] && overrides="${overrides} ${extra[*]}"

    if [[ "$DRY_RUN" == "1" ]]; then
        log_info "[DRY-RUN] CUDA_VISIBLE_DEVICES=${GPU_ID} ${cmd[*]}"
        _PASSED=$((_PASSED + 1)); return 0
    fi

    local avail; avail="$(out_free_gb)"
    if [[ -n "$avail" && "$avail" -lt "$MIN_FREE_GB" ]]; then
        rm -f "${MARKER_DIR}/${eid}.lock/pid" "${MARKER_DIR}/${eid}.lock/started"; rmdir "${MARKER_DIR}/${eid}.lock" 2>/dev/null
        log_error "STOPPING: disk down to ${avail}G (< ${MIN_FREE_GB}G) before ${eid}."
        log_error "Free space, then re-run this stage — finished runs are skipped automatically."
        exit 1
    fi

    echo "========================================"
    log_info "[${_TOTAL}] ${eid}"
    log_info "  ${task}/${dataset}/${method} seed=${seed} ablation=${ablation:-none} rounds=${ROUNDS}"
    log_info "  cfg=${cfg}"
    log_info "  est ~$(hms "$(est_s "$task" "$dataset" "$method")")"
    echo "========================================"

    local start_ts end_ts t0 t1 rc runtime status
    start_ts="$(date -Is)"; t0="$(date +%s)"

    if [[ "$TIMEOUT_S" -gt 0 ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_ID" timeout "$TIMEOUT_S" "${cmd[@]}" 2>&1 | tee "$log_file"
    else
        CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" 2>&1 | tee "$log_file"
    fi
    rc=${PIPESTATUS[0]}

    # Trap 1: stale run dir from a previous crash -> archive it and retry once.
    if [[ $rc -ne 0 ]] && grep -q "Refusing to overwrite existing run directory" "$log_file"; then
        local stale
        stale="$(grep -o "Refusing to overwrite existing run directory [^;]*" "$log_file" \
                 | head -1 | sed 's/^Refusing to overwrite existing run directory //' | xargs)"
        if [[ -n "$stale" && -d "$stale" ]]; then
            log_warn "stale run dir from an earlier attempt: ${stale}"
            log_warn "archiving to ${stale}.stale-$(date +%s) and retrying once"
            mv "$stale" "${stale}.stale-$(date +%s)"
            if [[ "$TIMEOUT_S" -gt 0 ]]; then
                CUDA_VISIBLE_DEVICES="$GPU_ID" timeout "$TIMEOUT_S" "${cmd[@]}" 2>&1 | tee "$log_file"
            else
                CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" 2>&1 | tee "$log_file"
            fi
            rc=${PIPESTATUS[0]}
        fi
    fi

    t1="$(date +%s)"; end_ts="$(date -Is)"; runtime=$((t1 - t0))

    if [[ $rc -eq 0 ]]; then
        status="PASS"
        # This campaign has a documented history of silent NaN collapse
        # (fp16 prototype-std overflow). Flag it; never silently accept.
        if grep -qiE '(loss|iou|dice|acc)[^,]{0,24}\bnan\b' "$log_file"; then
            status="PASS_NAN_WARNING"
            log_warn "NaN seen in a loss/metric line of ${eid} — inspect before using this row."
        fi
        _PASSED=$((_PASSED + 1))
        touch "${MARKER_DIR}/${eid}.done"; rm -f "${MARKER_DIR}/${eid}.failed"
        log_info "[${status}] ${eid} in $(hms "$runtime")"
    else
        status="FAIL_exit${rc}"
        _FAILED=$((_FAILED + 1)); _FAILED_LIST+=("${eid} (exit ${rc})")
        touch "${MARKER_DIR}/${eid}.failed"
        log_error "[FAIL] ${eid} exit=${rc} after $(hms "$runtime") — ${log_file}"
    fi

    rm -f "${MARKER_DIR}/${eid}.lock/pid" "${MARKER_DIR}/${eid}.lock/started"
    rmdir "${MARKER_DIR}/${eid}.lock" 2>/dev/null
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s\n' \
        "$start_ts" "$end_ts" "$runtime" "$stage" "$eid" "$task" "$dataset" "$method" \
        "${ablation:-none}" "$seed" "$ROUNDS" "$cfg" "$overrides" "$rc" "$status" "$log_file" \
        >> "$MANIFEST"
    return 0
}

# run_seeds <stage> <prefix> <task> <dataset> <method> <cfg> <ablation|""> [--set ...]
run_seeds() {
    local stage="$1" prefix="$2" task="$3" dataset="$4" method="$5" cfg="$6" ablation="$7"
    shift 7
    local seed eid
    for seed in $SEEDS; do
        eid="${prefix}_s${seed}"
        run_exp "$stage" "$eid" "$task" "$dataset" "$method" "$cfg" "$seed" "$ablation" "$@"
    done
}

# =============================================================================
# CONFIG MAP — the 24 (dataset, method) pairs the 2026-07-24 pre-flight smoke
# gate proved runnable end-to-end. "task|dataset|method|config"
# =============================================================================
MAIN_MATRIX=(
  "classification|cifar10|federated|configs/classification/federated/cifar_fedavg_alexnet.yaml"
  "classification|cifar10|hierfl|configs/classification/hierfl/cifar_hierfl_alexnet.yaml"
  "classification|cifar10|hsfl|configs/classification/hsfl/cifar_our_alexnet_5_10.yaml"
  "classification|cifar10|splitfl|configs/classification/splitfl/cifar_splitfed_alexnet.yaml"
  "classification|cifar10|hetero-sfl|configs/classification/hetero-sfl/cifar_heteroSFL_alexnet.yaml"

  "classification|cifar100|federated|configs/classification/federated/cifar_fedavg_resnet50.yaml"
  "classification|cifar100|hierfl|configs/classification/hierfl/cifar_hierfl_resnet50.yaml"
  "classification|cifar100|hsfl|configs/classification/hsfl/cifar_our_resnet50_5_10.yaml"
  "classification|cifar100|splitfl|configs/classification/splitfl/cifar_splitfed_resnet50.yaml"
  "classification|cifar100|hetero-sfl|configs/classification/hetero-sfl/cifar_heteroSFL_resnet18.yaml"

  "classification|ham10000|federated|configs/classification/federated/ham10000_fedavg_resnet50.yaml"
  "classification|ham10000|hierfl|configs/classification/hierfl/ham10000_hierfl_resnet50.yaml"
  "classification|ham10000|hsfl|configs/classification/hsfl/ham10000_our_resnet50_5_10.yaml"
  "classification|ham10000|splitfl|configs/classification/splitfl/ham10000_splitfed_resnet50.yaml"
  "classification|ham10000|hetero-sfl|configs/classification/hetero-sfl/ham10000_heteroSFL_resnet50.yaml"

  "segmentation|isic2018|federated|configs/segmentation/federated/isic_fedavg_resnet50.yaml"
  "segmentation|isic2018|hierfl|configs/segmentation/hierfl/isic_hierfl_resnet50.yaml"
  "segmentation|isic2018|hsfl|configs/segmentation/hsfl/isic_our_resnet50_5_10.yaml"
  "segmentation|isic2018|splitfl|configs/segmentation/splitfl/isic_splitfed_resnet50.yaml"
  "segmentation|isic2018|hetero-sfl|configs/segmentation/hetero-sfl/isic_heteroSFL_resnet50.yaml"
)

# H-SFP / E-HSFP primary configs, one per dataset. "task|dataset|config"
HSFP_CFG=(
  "classification|cifar10|configs/classification/h-sfp/cifar_our_alexnet_5_10.yaml"
  "classification|cifar100|configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"
  "classification|ham10000|configs/classification/h-sfp/ham10000_our_resnet50_5_10.yaml"
  "segmentation|isic2018|configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml"
)

CLS_CFG="configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"   # CIFAR-100
SEG_CFG="configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml"      # ISIC-2018

ALL_PRESETS=(baseline_hsfp hsfp_memory hsfp_memory_dropout
             hsfp_memory_reliability hsfp_memory_reliability_prc full_e_hsfp)

# =============================================================================
# STAGES
# =============================================================================

# ── s0: pre-flight ───────────────────────────────────────────────────────────
s0_preflight() {
    log_stage "s0_preflight — unit tests + 2-round crash-freedom gate"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        echo "  ~1-2 h: pytest suite + 24 two-round smoke runs (skipped if already marked)"
        return 0
    fi
    if [[ "$DRY_RUN" != "1" ]]; then
        log_info "running the metric/communication/logging test suite"
        python -m pytest tests/ -q 2>&1 | tail -20
        [[ ${PIPESTATUS[0]} -ne 0 ]] && log_warn "tests reported failures — read the output above before continuing"
    fi
    if [[ -x scripts/journal_experiments/run_preflight_smoke.sh ]]; then
        GPU_ID="$GPU_ID" DRY_RUN="$DRY_RUN" scripts/journal_experiments/run_preflight_smoke.sh
    else
        log_warn "scripts/journal_experiments/run_preflight_smoke.sh not found — skipping the smoke gate"
    fi
}

# ── s1: conference parity (BLOCKING) ─────────────────────────────────────────
# The open blocker in ALL_SIMULATION_RESULTS.md Section 6: H-SFP scores ~11% on
# CIFAR-100 against a 55.10% published reference. Until that is explained, no
# 5-seed number from this repo is trustworthy. One seed, full budget, per
# dataset. Read the table this prints before launching s2.
s1_parity() {
    log_stage "s1_parity — 1 seed x 4 datasets, H-SFP vs the ECCV reference (BLOCKING)"
    local entry task dataset cfg
    for entry in "${HSFP_CFG[@]}"; do
        IFS='|' read -r task dataset cfg <<< "$entry"
        run_exp s1_parity "parity_${dataset}_hsfp" "$task" "$dataset" h-sfp "$cfg" 0 baseline_hsfp
    done
    [[ "$PLAN_ONLY" == "1" ]] && return 0
    cat <<'EOF'

  ---------------------------------------------------------------
  PARITY GATE — compare the runs above against the ECCV reference:
      CIFAR-10    67.44 +- 0.85 %
      CIFAR-100   55.10 +- 0.95 %
      HAM10000    80.86 +- 0.75 %
      ISIC-2018   68.3  +- 0.5  % IoU / 79.5 +- 0.4 % Dice
  A result near 11% on CIFAR-100 is a FAILED parity check, not a
  journal number. Do not start s2_main until this is explained.
  ---------------------------------------------------------------
EOF
}

# ── s2: main matched comparison ──────────────────────────────────────────────
s2_main() {
    log_stage "s2_main — 5 baselines + H-SFP + E-HSFP, 4 datasets, ${SEEDS// /,}"
    local entry task dataset method cfg
    for entry in "${MAIN_MATRIX[@]}"; do
        IFS='|' read -r task dataset method cfg <<< "$entry"
        run_seeds s2_main "main_${dataset}_${method}" "$task" "$dataset" "$method" "$cfg" ""
    done
    for entry in "${HSFP_CFG[@]}"; do
        IFS='|' read -r task dataset cfg <<< "$entry"
        run_seeds s2_main "main_${dataset}_hsfp"  "$task" "$dataset" h-sfp "$cfg" baseline_hsfp
        run_seeds s2_main "main_${dataset}_ehsfp" "$task" "$dataset" h-sfp "$cfg" full_e_hsfp
    done
}

# ── s2b: the other FedX variants ─────────────────────────────────────────────
s2b_fedvariants() {
    log_stage "s2b_fedvariants — FedProx / FedNova / FedSGD"
    local v
    for v in fedprox fednova fedsgd; do
        run_seeds s2b "fed_cifar10_${v}"  classification cifar10  federated \
            "configs/classification/federated/cifar_${v}_alexnet.yaml" ""
        run_seeds s2b "fed_cifar100_${v}" classification cifar100 federated \
            "configs/classification/federated/cifar_${v}_resnet50.yaml" ""
        run_seeds s2b "fed_ham10000_${v}" classification ham10000 federated \
            "configs/classification/federated/ham10000_${v}_resnet50.yaml" ""
        run_seeds s2b "fed_isic2018_${v}" segmentation isic2018 federated \
            "configs/segmentation/federated/isic_${v}_resnet50.yaml" ""
    done
}

# ── s3: component ablation ───────────────────────────────────────────────────
s3_ablation() {
    log_stage "s3_ablation — 6 presets x 3 datasets x ${SEEDS// /,}"
    local entry task dataset cfg preset
    for entry in "${HSFP_CFG[@]}"; do
        IFS='|' read -r task dataset cfg <<< "$entry"
        [[ "$dataset" == "cifar10" && "${ABL_INCLUDE_CIFAR10:-0}" != "1" ]] && continue
        for preset in "${ALL_PRESETS[@]}"; do
            run_seeds s3_ablation "abl_${dataset}_${preset}" "$task" "$dataset" h-sfp "$cfg" "$preset"
        done
    done
}

# ── s4: non-IID robustness (classification only) ─────────────────────────────
s4_noniid() {
    log_stage "s4_noniid — Dirichlet alpha 0.7 (mild) / 0.3 (severe), classification only"
    log_warn "segmentation has no Dirichlet partitioner — ISIC-2018 non-IID cells cannot be produced"
    local entry task dataset cfg alpha tag
    for alpha in 0.7 0.3; do
        tag="a$(echo "$alpha" | tr '.' 'p')"
        for entry in "${HSFP_CFG[@]}"; do
            IFS='|' read -r task dataset cfg <<< "$entry"
            [[ "$task" == "segmentation" ]] && continue
            run_seeds s4_noniid "noniid_${tag}_${dataset}_hsfp"  "$task" "$dataset" h-sfp "$cfg" \
                baseline_hsfp --set partition=dirichlet --set "dirichlet_alpha=${alpha}"
            run_seeds s4_noniid "noniid_${tag}_${dataset}_ehsfp" "$task" "$dataset" h-sfp "$cfg" \
                full_e_hsfp   --set partition=dirichlet --set "dirichlet_alpha=${alpha}"
        done
        # optional: the same alphas for the baselines, for a full non-IID table
        if [[ "${NONIID_BASELINES:-0}" == "1" ]]; then
            local m mcfg
            for entry in "${MAIN_MATRIX[@]}"; do
                IFS='|' read -r task dataset m mcfg <<< "$entry"
                [[ "$task" == "segmentation" ]] && continue
                run_seeds s4_noniid "noniid_${tag}_${dataset}_${m}" "$task" "$dataset" "$m" "$mcfg" "" \
                    --set partition=dirichlet --set "dirichlet_alpha=${alpha}"
            done
        fi
    done
}

# ── s5: prototype-dropout sweep ──────────────────────────────────────────────
# Values per JOURNAL_SIMULATION_RESULTS.md Section 21 item 2 (the brief's set,
# {0,0.1,0.2,0.3,0.5}, NOT the old script's {0,0.1,0.3,0.5,0.7}).
s5_dropout() {
    log_stage "s5_dropout — prototype_dropout_rate in {0.0,0.1,0.2,0.3,0.5}"
    local rate tag
    for rate in 0.0 0.1 0.2 0.3 0.5; do
        tag="dr$(echo "$rate" | tr '.' 'p')"
        run_seeds s5_dropout "dropout_cls_${tag}" classification cifar100 h-sfp "$CLS_CFG" \
            full_e_hsfp --set "prototype_dropout_rate=${rate}"
        run_seeds s5_dropout "dropout_seg_${tag}" segmentation isic2018 h-sfp "$SEG_CFG" \
            full_e_hsfp --set "prototype_dropout_rate=${rate}"
    done
}

# ── s6: staleness sweep ──────────────────────────────────────────────────────
s6_staleness() {
    log_stage "s6_staleness — max_prototype_age in {0,1,3,5,10}"
    local tau
    for tau in 0 1 3 5 10; do
        run_seeds s6_staleness "staleness_cls_tau${tau}" classification cifar100 h-sfp "$CLS_CFG" \
            full_e_hsfp --set "max_prototype_age=${tau}"
        run_seeds s6_staleness "staleness_seg_tau${tau}" segmentation isic2018 h-sfp "$SEG_CFG" \
            full_e_hsfp --set "max_prototype_age=${tau}"
    done
}

# ── s7: serverless stress ────────────────────────────────────────────────────
# One factor at a time around the frozen defaults (cold_start 0.15, timeout 0.05).
s7_serverless() {
    log_stage "s7_serverless — cold-start and function-timeout stress"
    local cs to tag
    for cs in 0.05 0.15 0.30; do
        tag="cs$(echo "$cs" | tr '.' 'p')_to0p05"
        run_seeds s7_serverless "stress_cls_${tag}" classification cifar100 h-sfp "$CLS_CFG" \
            full_e_hsfp --set "cold_start_probability=${cs}" --set "function_timeout_probability=0.05"
    done
    for to in 0.02 0.15; do
        tag="cs0p15_to$(echo "$to" | tr '.' 'p')"
        run_seeds s7_serverless "stress_cls_${tag}" classification cifar100 h-sfp "$CLS_CFG" \
            full_e_hsfp --set "cold_start_probability=0.15" --set "function_timeout_probability=${to}"
    done
}

# ── s8: memory-capacity ablation ─────────────────────────────────────────────
s8_memory() {
    log_stage "s8_memory — memory_size in {100,250,500,1000,2000}"
    local m
    for m in 100 250 500 1000 2000; do
        run_seeds s8_memory "memcap_cls_m${m}" classification cifar100 h-sfp "$CLS_CFG" \
            full_e_hsfp --set "memory_size=${m}"
    done
}

# ── s9: aggregation-rule comparison ──────────────────────────────────────────
# NO --ablation here: a preset would overwrite aggregation_mode (trap 3).
s9_aggregation() {
    log_stage "s9_aggregation — average vs sample_count_weighted vs learnable_reliability"
    local mode
    for mode in average sample_count_weighted learnable_reliability; do
        run_seeds s9_aggregation "aggr_cls_${mode}" classification cifar100 h-sfp "$CLS_CFG" "" \
            --set use_episodic_memory=true --set "aggregation_mode=${mode}"
        run_seeds s9_aggregation "aggr_seg_${mode}" segmentation isic2018 h-sfp "$SEG_CFG" "" \
            --set use_episodic_memory=true --set "aggregation_mode=${mode}"
    done
}

# ── s10: aggregation-interval sweep ──────────────────────────────────────────
s10_intervals() {
    log_stage "s10_intervals — (Ic,Ie) = (5,10), (10,20), (25,50)"
    local iv cfg
    for iv in 5_10 10_20 25_50; do
        for cfg in "configs/classification/h-sfp/cifar_our_resnet50_${iv}.yaml"; do
            run_seeds s10_intervals "intv_cifar100_${iv}_hsfp"  classification cifar100 h-sfp "$cfg" baseline_hsfp
            run_seeds s10_intervals "intv_cifar100_${iv}_ehsfp" classification cifar100 h-sfp "$cfg" full_e_hsfp
        done
        for cfg in "configs/classification/h-sfp/ham10000_our_resnet50_${iv}.yaml"; do
            run_seeds s10_intervals "intv_ham10000_${iv}_hsfp"  classification ham10000 h-sfp "$cfg" baseline_hsfp
            run_seeds s10_intervals "intv_ham10000_${iv}_ehsfp" classification ham10000 h-sfp "$cfg" full_e_hsfp
        done
        for cfg in "configs/segmentation/h-sfp/isic_our_resnet50_${iv}.yaml"; do
            run_seeds s10_intervals "intv_isic2018_${iv}_hsfp"  segmentation isic2018 h-sfp "$cfg" baseline_hsfp
            run_seeds s10_intervals "intv_isic2018_${iv}_ehsfp" segmentation isic2018 h-sfp "$cfg" full_e_hsfp
        done
    done
}

# ── s11: scalability ─────────────────────────────────────────────────────────
# mid_server[0] and num_edges are read independently (trap 5) — always both.
s11_scalability() {
    log_stage "s11_scalability — client count {50,100,200,400}, edge count {2,5,10}"
    local n e tmp
    for n in 50 100 200 400; do
        run_seeds s11_scalability "scale_clients_n${n}_hsfp"  classification cifar100 h-sfp "$CLS_CFG" \
            baseline_hsfp --set "num_users=${n}" --set num_edges=5
        run_seeds s11_scalability "scale_clients_n${n}_ehsfp" classification cifar100 h-sfp "$CLS_CFG" \
            full_e_hsfp   --set "num_users=${n}" --set num_edges=5
    done
    for e in 2 5 10; do
        if [[ "$PLAN_ONLY" == "1" || "$DRY_RUN" == "1" ]]; then
            tmp="$CLS_CFG"
        else
            tmp="$(write_child_cfg "$CLS_CFG" "edges${e}" "mid_server: [${e}]" "num_edges: ${e}")"
        fi
        run_seeds s11_scalability "scale_edges_e${e}_hsfp"  classification cifar100 h-sfp "$tmp" \
            baseline_hsfp --set num_users=200
        run_seeds s11_scalability "scale_edges_e${e}_ehsfp" classification cifar100 h-sfp "$tmp" \
            full_e_hsfp   --set num_users=200
        [[ "$tmp" != "$CLS_CFG" ]] && rm -f "$tmp"
    done
}

# ── s12: recover the ECCV campaign's 2 OOM failures ──────────────────────────
# The ONLY stage that does not use the frozen protocol. These two cells belong
# to the ECCV partial-participation grid (docs/ECCV_REBUTTAL_SIMULATIONS.md §4),
# where 43 of 45 completed and these two died in Adam.step() with all 200
# clients active under an IID split. To stay comparable with the 43 siblings
# they must rerun at the ECCV protocol, not the journal one — hence the explicit
# epochs=30 and friends, which override this script's --set epochs=$ROUNDS
# (main.py folds --set into a dict, so the later duplicate key wins).
# local_bs is halved 16 -> 8 as the memory fix; nothing else changes.
ECCV_PROTO=(--set dataset=cifar100 --set frac=1.0 --set iid=true
            --set epochs=30 --set ssl_epochs_client=3 --set ssl_epochs_edge=3
            --set syn_epochs_cloud=3 --set t1=3 --set t2=6 --set local_ep=3
            --set local_bs=8)

s12_eccv_recover() {
    log_stage "s12_eccv_recover — the 2 ECCV partial-participation runs that OOM'd"
    [[ "$PLAN_ONLY" != "1" ]] && log_warn "ECCV protocol (30 rounds, seed 0, local_bs=8) — NOT a frozen-protocol result"
    run_exp s12_eccv_recover eccv_recover_partial_federated_frac1p0_iid \
        classification cifar100 federated \
        configs/classification/federated/cifar_fedavg_resnet50.yaml 0 "" "${ECCV_PROTO[@]}"
    run_exp s12_eccv_recover eccv_recover_partial_hierfl_frac1p0_iid \
        classification cifar100 hierfl \
        configs/classification/hierfl/cifar_hierfl_resnet50.yaml 0 "" "${ECCV_PROTO[@]}"
    [[ "$PLAN_ONLY" == "1" ]] && return 0
    cat <<'EOF'

  On success, fold both into the ECCV record:
    python scripts/camera_ready/collect_camera_ready.py
    python scripts/camera_ready/make_tables.py --only partial
  then replace the two FAILED cells in docs/ECCV_REBUTTAL_SIMULATIONS.md §4.
EOF
}

# ── s13: prototype-synthesis ablation — GATED OFF ────────────────────────────
s13_synthesis() {
    log_stage "s13_synthesis — residual generator + dropout-consistency loss"
    log_error "GATED: use_residual_generator is a no-op."
    log_error "  classification/H-SFP/hierarchy.py:471  self.residual_generator = None"
    log_error "  segmentation/H-SFP/hierarchy.py:328    self.residual_generator = None"
    log_error "Running this stage now would produce a table whose two arms are"
    log_error "byte-identical and label them as a synthesis ablation. Wire the"
    log_error "generator + dropout_consistency_loss in first, then remove this gate."
    [[ "$FORCE" == "1" ]] || return 1
    log_warn "FORCE=1 — proceeding anyway. The result is NOT a synthesis ablation."
    local g
    for g in 0.0 0.1 0.3; do
        run_seeds s13_synthesis "synth_cls_g$(echo "$g" | tr '.' 'p')" classification cifar100 h-sfp \
            "$CLS_CFG" full_e_hsfp --set "generator_scale=${g}"
    done
}

# ── s99: collect + validate ──────────────────────────────────────────────────
s99_collect() {
    log_stage "s99_collect — parse logs, revalidate metrics, rebuild the ledger"
    [[ "$PLAN_ONLY" == "1" ]] && { echo "  minutes, CPU only"; return 0; }
    [[ "$DRY_RUN" == "1" ]] && return 0

    if [[ -f scripts/journal_experiments/collect_results.py ]]; then
        python scripts/journal_experiments/collect_results.py || log_warn "collect_results.py failed"
    fi
    # Independently recompute accuracy / macro-F1 from the saved raw predictions
    # rather than trusting the training loop's own printed numbers.
    if [[ -f scripts/journal_experiments/validate_metrics.py ]]; then
        local npz
        while IFS= read -r npz; do
            echo "--- $(dirname "$npz")"
            python scripts/journal_experiments/validate_metrics.py "$npz" 2>&1 | tail -8
        done < <(find "$OUT_ROOT" -name 'best_val_predictions.npz' 2>/dev/null | sort)
    fi
    if [[ -f docs/journal_campaign/build_matrix.py ]]; then
        python docs/journal_campaign/build_matrix.py  && \
        python docs/journal_campaign/build_ledger.py  && \
        python docs/journal_campaign/assemble_md.py   || log_warn "ledger rebuild failed"
    fi
    log_info "manifest: ${MANIFEST}"
}


# =============================================================================
# COMPLETION CAMPAIGN STAGES (2026-10-02) — one per manuscript table axis.
# Mapped from the manuscript (Tables I-XIII). Every stress/ablation grid runs
# on CIFAR-100 (the conference ablation dataset, Table VII) with the frozen
# protocol: 60 rounds, seeds 0-4, IID unless stated. Cells identical to an
# already-completed run are NOT re-run; the collector reuses them:
#   p=0.2 (III), tau=0 (IV), K=200 (XII), baseline/full (VIII) -> main_cifar100_{hsfp,ehsfp}
# Ordering is seed-major inside each stage so partial results stay paired.
# =============================================================================
CIFAR100_HSFP="configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"

c_seed_major() {  # c_seed_major <callback>  — calls "<callback> <seed>" for each seed
    local seed; for seed in $SEEDS; do "$1" "$seed"; done
}

# Table I — FedProx / FedNova matched rows (FedAvg/HierFL/SplitFed/HeteroSFL/HSFL done in s2)
c_fedvariants() {
    log_stage "c_fedvariants — Table I FedProx/FedNova, CIFAR-10/100 + HAM10000"
    local v seed
    for seed in $SEEDS; do for v in fedprox fednova; do
        run_exp c_fedvariants "fed_cifar10_${v}_s${seed}"  classification cifar10  federated "configs/classification/federated/cifar_${v}_alexnet.yaml"   "$seed" ""
        run_exp c_fedvariants "fed_cifar100_${v}_s${seed}" classification cifar100 federated "configs/classification/federated/cifar_${v}_resnet50.yaml"  "$seed" ""
        run_exp c_fedvariants "fed_ham10000_${v}_s${seed}" classification ham10000 federated "configs/classification/federated/ham10000_${v}_resnet50.yaml" "$seed" ""
    done; done
}

# Table VIII — E-HSFP module ablation (residual-generator row is BLOCKED: not wired)
c_ablation() {
    log_stage "c_ablation — Table VIII presets on CIFAR-100"
    local seed preset
    for seed in $SEEDS; do
        for preset in hsfp_memory hsfp_memory_dropout hsfp_memory_reliability hsfp_memory_reliability_prc; do
            run_exp c_ablation "abl_cifar100_${preset}_s${seed}" classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" "$preset"
        done
    done
}

# Table III / Fig. 5 — packet-loss (prototype dropout) rate p; p=0.2 is main E-HSFP
c_dropout() {
    log_stage "c_dropout — Table III p in {0,0.1,0.3,0.5} (+0.2 reused)"
    local seed rate
    for seed in $SEEDS; do for rate in 0.0 0.1 0.3 0.5; do
        run_exp c_dropout "dropout_cifar100_p$(tr . p <<< "$rate")_s${seed}" classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" full_e_hsfp \
            --set "prototype_dropout_rate=${rate}"
    done; done
}

# Table IV / Fig. 6 — bounded staleness tau (real delayed packets); tau=0 is main E-HSFP
c_staleness() {
    log_stage "c_staleness — Table IV tau in {1,3,5,10} (+0 reused)"
    local seed tau
    for seed in $SEEDS; do for tau in 1 3 5 10; do
        run_exp c_staleness "stale_cifar100_tau${tau}_s${seed}" classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" full_e_hsfp \
            --set "staleness_tau=${tau}"
    done; done
}

# Table V — serverless-style events, one factor at a time + combined
c_serverless() {
    log_stage "c_serverless — Table V event stress"
    local seed
    local none=(--set client_timeout_probability=0.0 --set edge_timeout_probability=0.0 --set cold_start_probability=0.0)
    for seed in $SEEDS; do
        run_exp c_serverless "sv_cifar100_none_s${seed}"      classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" full_e_hsfp "${none[@]}"
        run_exp c_serverless "sv_cifar100_timeout_s${seed}"   classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" full_e_hsfp \
            --set client_timeout_probability=0.10 --set edge_timeout_probability=0.0 --set cold_start_probability=0.0
        run_exp c_serverless "sv_cifar100_coldstart_s${seed}" classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" full_e_hsfp \
            --set client_timeout_probability=0.0 --set edge_timeout_probability=0.0 --set cold_start_probability=0.30 --set cold_start_defers_packet=true
        run_exp c_serverless "sv_cifar100_partial_s${seed}"   classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" full_e_hsfp \
            "${none[@]}" --set partial_edge_probability=0.30 --set partial_edge_fraction=0.5
        run_exp c_serverless "sv_cifar100_missedge_s${seed}"  classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" full_e_hsfp \
            --set client_timeout_probability=0.0 --set edge_timeout_probability=0.10 --set cold_start_probability=0.0
        run_exp c_serverless "sv_cifar100_combined_s${seed}"  classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" full_e_hsfp \
            --set client_timeout_probability=0.10 --set edge_timeout_probability=0.10 --set cold_start_probability=0.30 \
            --set cold_start_defers_packet=true --set partial_edge_probability=0.30 --set partial_edge_fraction=0.5
    done
}

# Table VI / Fig. 2 — Dirichlet non-IID (IID cells reuse s2 main runs)
c_noniid() {
    log_stage "c_noniid — Table VI dir(0.7)/dir(0.3), H-SFP + E-HSFP"
    local seed alpha tag entry task dataset cfg
    for seed in $SEEDS; do for alpha in 0.7 0.3; do
        tag="a$(tr . p <<< "$alpha")"
        for entry in "${HSFP_CFG[@]}"; do
            IFS='|' read -r task dataset cfg <<< "$entry"
            [[ "$task" == "segmentation" ]] && continue
            run_exp c_noniid "noniid_${tag}_${dataset}_hsfp_s${seed}"  "$task" "$dataset" h-sfp "$cfg" "$seed" baseline_hsfp \
                --set partition=dirichlet --set "dirichlet_alpha=${alpha}"
            run_exp c_noniid "noniid_${tag}_${dataset}_ehsfp_s${seed}" "$task" "$dataset" h-sfp "$cfg" "$seed" full_e_hsfp \
                --set partition=dirichlet --set "dirichlet_alpha=${alpha}"
        done
    done; done
}

# Table IX / Fig. 9 — per-class FIFO memory capacity M (eq. 7) under dropout (p=0.2)
# + staleness (tau=3); the global record cap is lifted so M is the binding limit.
c_memory() {
    log_stage "c_memory — Table IX M in {0,1,5,10,20}, p=0.2, tau=3"
    local seed m
    for seed in $SEEDS; do for m in 0 1 5 10 20; do
        run_exp c_memory "memM_cifar100_M${m}_s${seed}" classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" full_e_hsfp \
            --set "memory_per_class=${m}" --set memory_size=1000000 --set staleness_tau=3
    done; done
}

# Table X — aggregation rule x {clean, dropout p=0.3, staleness tau=5}. No preset
# (a preset would overwrite aggregation_mode); module flags set explicitly.
c_aggregation() {
    log_stage "c_aggregation — Table X 4 rules x 3 conditions"
    local seed rule mode mem cond cextra
    for seed in $SEEDS; do
      for rule in average sample_count_weighted learnable_reliability learnable_reliability_memory; do
        mode="${rule%_memory}"; mem=false; [[ "$rule" == *_memory ]] && mem=true
        for cond in clean drop stale; do
          case "$cond" in
            clean) cextra=(--set use_prototype_dropout=false) ;;
            drop)  cextra=(--set use_prototype_dropout=true --set prototype_dropout_rate=0.3) ;;
            stale) cextra=(--set use_prototype_dropout=false --set staleness_tau=5) ;;
          esac
          run_exp c_aggregation "aggr_cifar100_${rule}_${cond}_s${seed}" classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" "" \
              --set "aggregation_mode=${mode}" --set "use_episodic_memory=${mem}" --set use_prc_loss=false \
              --set use_serverless_simulation=false --set use_residual_generator=false "${cextra[@]}"
        done
      done
    done
}

# Table XII — client population K (fixed dataset size); K=200 is main H-SFP
c_scalability() {
    log_stage "c_scalability — Table XII H-SFP(A) K in {20,50,100,500}"
    local seed k
    for seed in $SEEDS; do for k in 20 50 100 500; do
        run_exp c_scalability "scale_cifar100_K${k}_hsfp_s${seed}" classification cifar100 h-sfp "$CIFAR100_HSFP" "$seed" baseline_hsfp \
            --set "num_users=${k}"
    done; done
}

# Table XIII — literal 200-round communication total for E-HSFP (H-SFP(A) interval).
# Separate protocol (200 rounds); one seed; never pooled with 60-round results.
c_comm200() {
    log_stage "c_comm200 — Table XIII E-HSFP 200-round communication (seed 0)"
    run_exp c_comm200 "comm200_cifar100_ehsfp_s0" classification cifar100 h-sfp "$CIFAR100_HSFP" 0 full_e_hsfp --set epochs=200
}

# =============================================================================
# DRIVER
# =============================================================================
STAGES=(s0_preflight s1_parity s2_main s2b_fedvariants s3_ablation s4_noniid
        s5_dropout s6_staleness s7_serverless s8_memory s9_aggregation
        s10_intervals s11_scalability s12_eccv_recover s99_collect
        c_fedvariants c_ablation c_dropout c_staleness c_serverless c_noniid
        c_memory c_aggregation c_scalability c_comm200)

# The defensible journal core: the parity gate that decides whether any of it
# is trustworthy, the headline matched comparison, and the E-HSFP component
# story. The sweeps (s5-s11) are supporting figures, not the argument.
CORE_STAGES=(s1_parity s2_main s3_ablation)

cmd_runnable() {
    cat <<EOF

============================================================================
  WHAT CAN AND CANNOT RUN
  Audited 2026-08-02 against the code, not inferred from docs.
============================================================================

  RUNNABLE — 13 stages
    s0_preflight s1_parity s2_main s2b_fedvariants s3_ablation
    s5_dropout s6_staleness s7_serverless s8_memory s9_aggregation
    s10_intervals s11_scalability s12_eccv_recover s99_collect

    Evidence: the 2026-07-24 pre-flight gate ran all 6 method families x 4
    datasets, 24/24 clean. CIFAR-10, CIFAR-100, HAM10000 and ISIC-2018 are
    all present on disk.

  RESTRICTED — 1 stage, partially
    s4_noniid   classification only. CIFAR-10/100 + HAM10000 run; ISIC-2018
                cannot. partition_from_config is imported by all 6
                classification methods and by NO segmentation method, so
                there is no Dirichlet partitioner on that side. Producing
                ISIC non-IID cells needs new code, not GPU time.

  BLOCKED — cannot run
    s13_synthesis   residual generator hard-coded to None in both
                    hierarchies; the two arms would be byte-identical.
    ImageNet-1K     dataset absent, and out of scope by user decision
                    2026-07-23.

  RUNNABLE BUT NOT YET TRUSTWORTHY
    Every H-SFP / E-HSFP stage is downstream of the CIFAR-100 parity gap
    (~11% here vs the 55.10% published reference). The ECCV camera-ready
    runs show the same collapse at 3.5-6.5%, so this is not a recent
    regression - it has been present since June across two independent
    campaigns. s1_parity is 4 runs / ~17h and decides whether the other
    ~130 days are worth spending.

  SCOPE LEVERS (measured, not guessed)
    everything, 5 seeds          696 runs   ~130 days
    SEEDS="0 1 2"                420 runs   ~78 days
    SEEDS="0 1 2" SKIP_SEG=1     314 runs   ~57 days
    ./journal_run.sh core        parity + main table + ablation
    Departing from the frozen [0,1,2,3,4] means n must be stated honestly in
    the paper; SEEDS is recorded per row in the manifest either way.
============================================================================
EOF
}

cmd_coverage() {
    local cr="results/camera_ready"
    local done_n=0 fail_n=0
    if [[ -d "${cr}/.markers" ]]; then
        done_n=$(find "${cr}/.markers" -name '*.done'   2>/dev/null | wc -l)
        fail_n=$(find "${cr}/.markers" -name '*.failed' 2>/dev/null | wc -l)
    fi
    cat <<EOF

============================================================================
  PRIOR-WORK COVERAGE
  Full record: docs/ECCV_REBUTTAL_SIMULATIONS.md
============================================================================

  ECCV 2026 rebuttal / camera-ready campaign (2026-06-22..26)
    training runs completed : ${done_n}
    training runs failed    : ${fail_n}   (CUDA OOM, frac=1.0 IID)
    executed protocol       : CIFAR-100, 30 rounds, seed 0, 200 clients/5 edges,
                              ssl_epochs=3, local_ep=3, t1=3 t2=6

  What that covers for THIS campaign:

  REUSED — done, not scheduled here
    covariance ablation        results/camera_ready/covariance/
    Lstat sensitivity          results/camera_ready/lstat/
    phase profiling            results/camera_ready/profiling/
    feature inversion          results/camera_ready/inversion/
    fairness manifest          results/camera_ready/fairness/
    partition diagnostics      results/camera_ready/hetero/partition_diagnostics/
    partial participation      results/camera_ready/partial/   (grid the journal
                                 brief does not request; no stage here)

  NOT REUSABLE — 0 runs skipped on their account
    hetero + partial training runs (63). Wrong round budget (30 vs ${ROUNDS}),
    1 seed not 5, reduced SSL/local epochs, pre-fix code, and a 'test_f1'
    column that actually holds top-1 accuracy for h-sfp/hsfl/splitfl.
    H-SFP is collapsed at 3.5-6.5% in every one of them.

  STILL OWED BY ECCV
    2 OOM runs  ->  ./journal_run.sh s12_eccv_recover

  ALPHAS DO NOT OVERLAP
    ECCV ran two-level alpha in {1.0,0.1} and flat alpha in {0.1,0.05}.
    s4_noniid runs the frozen 0.7 / 0.3, never executed in this repo.
============================================================================
EOF
}

cmd_status() {
    echo "results root : ${RESULTS_ROOT}"
    if [[ ! -d "$MARKER_DIR" ]]; then echo "no runs yet."; return 0; fi
    local d f
    d=$(find "$MARKER_DIR" -name '*.done'   2>/dev/null | wc -l)
    f=$(find "$MARKER_DIR" -name '*.failed' 2>/dev/null | wc -l)
    echo "done         : ${d}"
    echo "failed       : ${f}"
    if [[ "$f" -gt 0 ]]; then
        echo "failed runs:"
        find "$MARKER_DIR" -name '*.failed' -printf '  - %f\n' 2>/dev/null | sed 's/\.failed$//'
        echo "  (./journal_run.sh retry_failed clears these so they re-run)"
    fi
    [[ -f "$MANIFEST" ]] && echo "manifest     : ${MANIFEST} ($(($(wc -l < "$MANIFEST") - 1)) rows)"
}

cmd_retry_failed() {
    local n; n=$(find "$MARKER_DIR" -name '*.failed' 2>/dev/null | wc -l)
    find "$MARKER_DIR" -name '*.failed' -delete 2>/dev/null
    log_info "cleared ${n} failed marker(s); re-run the stage to retry them"
}

cmd_plan() {
    PLAN_ONLY=1
    echo
    echo "============================================================================"
    echo "  E-HSFP JOURNAL CAMPAIGN PLAN"
    echo "  seeds=[${SEEDS}]  rounds=${ROUNDS}  gpu=${GPU_ID}  skip_seg=${SKIP_SEG}"
    echo "  Estimates are extrapolated from measured 60-round runtimes in"
    echo "  results/journal/logs/ — rough, single-GPU, sequential, +-30%."
    echo "  Runs already marked done are excluded."
    echo "============================================================================"
    printf "  %-18s %8s   %s\n" STAGE RUNS "EST. WALL TIME"
    local st total_runs=0 total_s=0
    for st in "${STAGES[@]}"; do
        _PLAN_RUNS=0; _PLAN_SECONDS=0
        "$st" >/dev/null 2>&1
        printf "  %-18s %8d   %s\n" "$st" "$_PLAN_RUNS" "$(hms "$_PLAN_SECONDS")"
        total_runs=$((total_runs + _PLAN_RUNS)); total_s=$((total_s + _PLAN_SECONDS))
    done
    echo "  --------------------------------------------------------------------------"
    printf "  %-18s %8d   %s  (~%d days non-stop)\n" TOTAL "$total_runs" "$(hms "$total_s")" $((total_s / 86400))
    echo "============================================================================"
    cat <<EOF

  Suggested order — do NOT jump straight to s2_main:
    1. ./journal_run.sh s0_preflight     tests + 2-round crash gate
    2. ./journal_run.sh s1_parity        BLOCKING: explain the CIFAR-100
                                         ~11% vs 55.10% gap before anything else
    3. ./journal_run.sh s2_main          the headline table
    4. ./journal_run.sh s3_ablation      the E-HSFP component story
    5. everything else, as time allows
    6. ./journal_run.sh s99_collect      after each batch

  The defensible core, if the full matrix is out of reach:
    ./journal_run.sh core            # = s1_parity + s2_main + s3_ablation
    SEEDS="0 1 2" ./journal_run.sh core
  That is the headline comparison plus the E-HSFP component story — the two
  tables a reviewer actually checks. The sweeps are supporting figures.

  Cheap first pass (proxy budget, 1 seed) to shake out failures:
    SEEDS=0 ROUNDS=10 ./journal_run.sh s2_main
  Proxy numbers are NOT reportable — see ALL_SIMULATION_RESULTS.md Section 2.

  What can and cannot run, and why:
    ./journal_run.sh runnable

  s13_synthesis is gated off: the residual generator is a no-op in both
  hierarchies. Wire it in before claiming a prototype-synthesis ablation.

  Already done elsewhere and NOT scheduled above — the ECCV rebuttal's 7
  analysis experiments and its partial-participation grid:
    ./journal_run.sh coverage       (full record: docs/ECCV_REBUTTAL_SIMULATIONS.md)
EOF
    PLAN_ONLY=0
}

main() {
    if [[ $# -eq 0 || "${1:-}" == "plan" ]]; then cmd_plan; exit 0; fi
    case "${1:-}" in
        status)       cmd_status; exit 0 ;;
        coverage)     cmd_coverage; exit 0 ;;
        runnable)     cmd_runnable; exit 0 ;;
        retry_failed) mkdir -p "$MARKER_DIR"; cmd_retry_failed; exit 0 ;;
    esac

    check_env
    local requested=("$@")
    [[ "${1}" == "all"  ]] && requested=("${STAGES[@]}")
    [[ "${1}" == "core" ]] && requested=("${CORE_STAGES[@]}")

    local st
    for st in "${requested[@]}"; do
        if ! declare -F "$st" >/dev/null; then
            log_error "unknown stage '${st}'. Available: ${STAGES[*]}"
            exit 1
        fi
    done

    local campaign_t0; campaign_t0="$(date +%s)"
    for st in "${requested[@]}"; do
        "$st"
    done

    echo
    echo "============================================================"
    echo "  CAMPAIGN SUMMARY  (stages: ${requested[*]})"
    echo "============================================================"
    echo "  Attempted: ${_TOTAL}"
    echo "  Passed:    ${_PASSED}"
    echo "  Skipped:   ${_SKIPPED}"
    echo "  Failed:    ${_FAILED}"
    echo "  Elapsed:   $(hms $(( $(date +%s) - campaign_t0 )))"
    echo "  Logs:      ${LOG_DIR}/"
    echo "  Manifest:  ${MANIFEST}"
    if [[ ${#_FAILED_LIST[@]} -gt 0 ]]; then
        echo
        echo "  Failed:"
        printf '    - %s\n' "${_FAILED_LIST[@]}"
        echo "  ./journal_run.sh retry_failed   # then re-run the stage"
    fi
    echo "============================================================"
    [[ $_FAILED -gt 0 ]] && exit 1
    exit 0
}

# Only drive the campaign when executed. Sourcing exposes the helpers
# (write_child_cfg, run_exp, est_s) for testing without launching anything.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
