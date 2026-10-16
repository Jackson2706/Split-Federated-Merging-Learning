#!/usr/bin/env bash
# Unattended Claude review, at most once per cron tick (every 2 h, after the collector).
# Runs ONLY if something needs attention: a new .failed run, a collector error, a dead lane while work is
# queued, LaTeX errors, or resource pressure. Bounded: flock (no overlap), 25 min timeout, 40 turns,
# compact input (status summary + failing log tails only). Output: results/campaign/reviews/.
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(dirname "$SRC")"
CAMP="$SRC/results/campaign"; mkdir -p "$CAMP/reviews"
exec 8>"$CAMP/review.lock"
flock -n 8 || { echo "review already running"; exit 0; }
[[ -f "$CAMP/REVIEW_DISABLED" ]] && { echo "review disabled ($CAMP/REVIEW_DISABLED)"; exit 0; }
export PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin

SUMMARY="$(python3 - "$CAMP" "$SRC" <<'EOF'
import json, os, sys, glob
camp, src = sys.argv[1:]
st = json.load(open(os.path.join(camp, "status.json")))
seen_p = os.path.join(camp, "reviews", "seen_failures.json")
seen = set(json.load(open(seen_p))) if os.path.exists(seen_p) else set()
new_fail = [f for f in st.get("failed", []) if f not in seen]
queued = sum(v.get("queued", 0) for v in st.get("counts", {}).values())
dead = [l["lane"] for l in st.get("lanes", []) if not l["alive"]]
b = st.get("builds", {})
latex_bad = {k: v for k, v in b.items() if v.get("latex_errors") or not v.get("pdf")}
reasons = []
if new_fail: reasons.append(f"new failed runs: {new_fail}")
if str(st.get("last_collection_rc")) not in ("0", "None"): reasons.append(f"collector rc={st.get('last_collection_rc')}")
if dead and queued: reasons.append(f"dead lanes {dead} with {queued} queued")
if latex_bad: reasons.append(f"latex problems: {list(latex_bad)}")
if st.get("resources", {}).get("pressure"): reasons.append(f"resource pressure {st['resources']['pressure']}")
if not st.get("fits_deadline", True): reasons.append(f"ETA {st.get('eta_all_kst')} beyond deadline")
if not reasons:
    print("NONE"); sys.exit()
out = ["REASONS: " + "; ".join(reasons), "STATUS: " + json.dumps({k: st.get(k) for k in
       ("updated_kst", "counts", "running", "lanes", "failed", "resources", "eta_all_kst", "eta_by_tier_kst",
        "measured_hours_per_run", "builds", "last_collection_rc")}, default=str)[:6000]]
for f in new_fail[:3]:
    lp = os.path.join(src, "results", "journal_v1", "logs", f"{f}.log")
    tail = open(lp, errors="replace").read().splitlines()[-40:] if os.path.exists(lp) else ["(no log)"]
    out.append(f"--- tail {f}.log\n" + "\n".join(tail)[-4000:])
lc = os.path.join(camp, "collect_logs", "latest.log")
if os.path.exists(lc):
    out.append("--- collector log tail\n" + "\n".join(open(lc, errors="replace").read().splitlines()[-30:])[-3000:])
json.dump(sorted(seen | set(new_fail)), open(seen_p, "w"))
print("\n".join(out))
EOF
)"
TS="$(TZ=Asia/Seoul date +%Y%m%dT%H%M)"
if [[ "$SUMMARY" == "NONE" ]]; then
  echo "$(TZ=Asia/Seoul date -Is) review skipped: nothing needs attention" >> "$CAMP/reviews/review_runs.log"; exit 0
fi
OUT="$CAMP/reviews/review_${TS}.md"
SYS="You are the unattended reviewer of a running ML experiment campaign in $REPO. Read RUN_STATE.md first.
Rules: never push, publish, purchase, or delete results/logs/markers of completed runs; never fabricate or
edit measured values; keep scientific settings unchanged (rounds, clients, seeds, model, data). Investigate
each failure from its log; fix only clear code/infrastructure bugs, then re-admit the run by removing its
.failed marker (at most once per run - check events.log for earlier re-admissions). Do not restart a run
that failed deterministically without a fix. Keep the scheduler running (scripts/journal_experiments/start_lanes.sh).
Record every action as one line appended to Sources/results/campaign/events.log (prefix 'review:') and add a
short dated note to RUN_STATE.md under '## Review notes'. Be brief; stop as soon as issues are handled."
cd "$REPO" && timeout 1500 claude -p "Campaign needs attention. Compact status follows; handle the issues, then exit.

$SUMMARY" --append-system-prompt "$SYS" --max-turns 40 --output-format text \
  --allowedTools "Read,Grep,Glob,Edit,Write,Bash" \
  --disallowedTools "Bash(git push:*)" "Bash(git reset:*)" "Bash(rm -rf:*)" "Bash(git clean:*)" > "$OUT" 2>&1
rc=$?
echo "$(TZ=Asia/Seoul date -Is) review rc=$rc out=${OUT#$SRC/}" >> "$CAMP/reviews/review_runs.log"
echo "$(TZ=Asia/Seoul date -Is) review rc=$rc out=${OUT#$SRC/}" >> "$CAMP/events.log"
