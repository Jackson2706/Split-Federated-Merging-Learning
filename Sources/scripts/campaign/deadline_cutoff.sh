#!/usr/bin/env bash
# Deadline cutoff (user instruction 2026-10-19: "if we can not finish before 1 PM, stop and get the results and
# put these to latex file"). Run at 13:00 KST:
#   1 for every listed run that is neither .done nor past round 60, record its last completed round's
#     validation accuracy / macro-F1 (the same metric the tables report at round 60) in
#     results/journal_v1/partial_runs.json -- fill_manuscript.py prints these with a "round k of 60" marker;
#   2 SIGTERM the run's training process. The lane marks it .failed (never retried), and when no run is left
#     it runs the final collection and autostop.sh as usual.
# A run that already finished all 60 rounds (final evaluation in progress) is left to complete.
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RES="$SRC/results/journal_v1"; MK="$RES/.markers"; CAMP="$SRC/results/campaign"
OUT="${OUT_ROOT:-/workspace/campaign/outputs_journal_v1}"
RUNS=(j3_memM_cifar100_M20_s0 j3_noniid_a0p3_cifar100_ehsfp_s0 j3_noniid_a0p7_cifar100_ehsfp_s0)
log() { echo "$(TZ=Asia/Seoul date -Is) deadline_cutoff: $*" | tee -a "$CAMP/events.log"; }
kill_list=()
for e in "${RUNS[@]}"; do
  [[ -f "$MK/$e.done" ]] && { log "$e already done"; continue; }
  m=$(ls -t "$OUT/$e"/*/*/metrics_unrounded.jsonl 2>/dev/null | head -1)
  [[ -z "$m" ]] && { log "$e: no per-round metrics; not stopped"; continue; }
  python3 - "$e" "$m" "$RES/partial_runs.json" <<'EOF' || { log "$e: could not record partial result; not stopped"; continue; }
import json, os, sys
eid, mpath, out = sys.argv[1:]
last = [json.loads(l) for l in open(mpath) if l.strip()][-1]
if int(last["epoch"]) >= 60:
    sys.exit(3)
d = json.load(open(out)) if os.path.exists(out) else {}
d[eid] = {"round": int(last["epoch"]), "rounds_planned": 60, "final_acc": float(last["validation_accuracy"]),
          "final_f1": float(last["validation_f1"]), "source": mpath,
          "reason": "stopped at the 2026-10-19 13:00 KST submission cutoff (user instruction)"}
tmp = out + ".tmp"; json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, out)
print(f"{eid}: round {last['epoch']} acc {last['validation_accuracy']:.4f} f1 {last['validation_f1']:.4f}")
EOF
  rc=$?
  [[ $rc -eq 3 ]] && { log "$e: 60 rounds done, final evaluation running; left to finish"; continue; }
  lp=$(cat "$MK/$e.lock/pid" 2>/dev/null)
  py=$(pgrep -P "${lp:-0}" -f 'python main.py' | head -1)
  [[ -n "$py" ]] && kill_list+=("$e:$py") || log "$e: training process not found"
done
for x in "${kill_list[@]}"; do
  log "stopping ${x%%:*} (pid ${x##*:}); partial result recorded in partial_runs.json"
  kill -TERM "${x##*:}"
done
log "cutoff done (${#kill_list[@]} run(s) stopped)"
