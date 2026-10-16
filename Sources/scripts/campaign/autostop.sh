#!/usr/bin/env bash
# Stop this Vast instance (halts GPU billing; disk kept, storage still billed) once the campaign is finished.
# User decision 2026-10-15: auto-stop after the selected queue completes.
# Preconditions (all required): no live run lock, no runnable item left in master.queue, the final
# collection succeeded (rc 0) AFTER the last run finished, and AUTOSTOP_DISABLED is absent.
# Failed runs do not block the stop (they are reported in status.json / RUN_STATE.md for later review).
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RES="$SRC/results/journal_v1"; CAMP="$SRC/results/campaign"
log() { echo "$(TZ=Asia/Seoul date -Is) autostop: $*" | tee -a "$CAMP/events.log"; }
[[ -f "$CAMP/AUTOSTOP_DISABLED" ]] && { log "disabled by $CAMP/AUTOSTOP_DISABLED"; exit 0; }
for lk in "$RES"/.markers/*.lock; do
  [[ -d "$lk" ]] || continue
  o="$(cat "$lk/pid" 2>/dev/null)"; [[ -n "$o" ]] && kill -0 "$o" 2>/dev/null && { log "run still active ($lk)"; exit 0; }
done
left=$(grep -v '^#' "$RES/queue/master.queue" | while IFS='|' read -r _ eid _; do
  [[ -n "$eid" && ! -f "$RES/.markers/$eid.done" && ! -f "$RES/.markers/$eid.failed" ]] && echo "$eid"; done | wc -l)
[[ "$left" -gt 0 ]] && { log "$left runnable queue items left"; exit 0; }
rc=$(python3 -c "import json;print(json.load(open('$CAMP/status.json')).get('last_collection_rc'))")
[[ "$rc" != "0" ]] && { log "last collection rc=$rc; not stopping"; exit 0; }
set -a; . /etc/environment; set +a
log "campaign finished and collected; stopping instance $CONTAINER_ID"
vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> "$CAMP/events.log" 2>&1
