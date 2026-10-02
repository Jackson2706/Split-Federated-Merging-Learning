#!/usr/bin/env bash
# Idempotently (re)start campaign GPU lanes in the detached tmux session
# "ehsfp-lanes". One lane per results/journal_v1/queue/<lane>.queue listed in
# results/journal_v1/queue/LANES. A lane already alive is left alone, so this
# is safe to run at any time (and from cron @reboot after a host restart).
#   scripts/journal_experiments/start_lanes.sh          # start missing lanes
#   scripts/journal_experiments/start_lanes.sh status   # show lane state
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QDIR="$ROOT/results/journal_v1/queue"
SESSION=ehsfp-lanes
cd "$ROOT" || exit 1
[[ -f "$QDIR/LANES" ]] || { echo "no $QDIR/LANES"; exit 1; }
# Artifacts go to the data disk (nofail mount): never start lanes without it.
# shellcheck source=/dev/null
source "$QDIR/lane_env.sh"
[[ -d "$(dirname "$OUT_ROOT")" ]] || { echo "$(date -Is) OUT_ROOT disk not mounted: $OUT_ROOT"; exit 1; }
[[ -f "$ROOT/results/journal_v1/.markers/STOP_ALL" && "${1:-}" != "status" ]] && { echo "STOP_ALL present; not starting lanes"; exit 0; }
alive() { local f="$QDIR/$1.pid"; [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; }
if [[ "${1:-}" == "status" ]]; then
  while read -r lane; do [[ -z "$lane" || "$lane" == \#* ]] && continue
    if alive "$lane"; then echo "$lane: running pid $(cat "$QDIR/$lane.pid")"; else echo "$lane: not running"; fi
  done < "$QDIR/LANES"; exit 0
fi
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n ctl "bash"
while read -r lane; do
  [[ -z "$lane" || "$lane" == \#* ]] && continue
  if alive "$lane"; then echo "$lane already running"; continue; fi
  tmux new-window -d -t "$SESSION" -n "$lane" \
    "source '$QDIR/lane_env.sh'; echo \$\$ > '$QDIR/$lane.pid'; exec '$ROOT/scripts/journal_experiments/queue_lane.sh' '$QDIR/$lane.queue' '$lane' >> '$QDIR/$lane.out' 2>&1"
  echo "started $lane"
done < "$QDIR/LANES"
