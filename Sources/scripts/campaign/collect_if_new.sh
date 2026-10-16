#!/usr/bin/env bash
# Event-driven collection (user request 2026-10-16: "update progress and LaTeX every 2 hours or on new results").
# Called by start_lanes.sh (cron, every 10 min). If any run finished (.done/.failed marker) after the last
# collection started, run collect_all.sh in the background. collect_all.sh's own flock prevents overlap; if
# it is busy, the next 10-min tick retries because the markers are still newer than the last collection log.
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAMP="$SRC/results/campaign"; MK="$SRC/results/journal_v1/.markers"
ref="$(readlink -f "$CAMP/collect_logs/latest.log" 2>/dev/null)"
[[ -f "$ref" ]] || exit 0
new="$(find "$MK" -maxdepth 1 \( -name '*.done' -o -name '*.failed' \) -newer "$ref" -printf '%f ' 2>/dev/null)"
[[ -z "$new" ]] && exit 0
echo "$(TZ=Asia/Seoul date -Is) new results ($new) -> collect"
nohup /bin/bash "$SRC/scripts/campaign/collect_all.sh" new_results >> "$CAMP/cron_collect.log" 2>&1 < /dev/null &
