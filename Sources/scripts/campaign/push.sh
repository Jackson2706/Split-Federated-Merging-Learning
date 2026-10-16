#!/usr/bin/env bash
# Commit the campaign's code + collected results and push them to origin/<branch> (user request 2026-10-15:
# "upload the changes whenever you wake up to collect result"). Called after every successful collection
# (cron tick and the final collection), before autostop. Never force-pushes; never rewrites history.
# If the remote branch has moved (non-fast-forward) the push is skipped and recorded in push_status.json,
# which makes review.sh flag it. Disable with results/campaign/PUSH_DISABLED.
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(dirname "$SRC")"
CAMP="$SRC/results/campaign"; RES="$SRC/results/journal_v1"
BRANCH="${PUSH_BRANCH:-DungTT}"
log() { echo "$(TZ=Asia/Seoul date -Is) push: $*" >> "$CAMP/events.log"; echo "push: $*"; }
status() { python3 - "$CAMP/push_status.json" "$1" "$2" <<'EOF'
import json, sys, datetime, zoneinfo
p, ok, msg = sys.argv[1:]
json.dump({"updated_kst": datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
           "ok": ok == "1", "message": msg}, open(p, "w"), indent=1)
EOF
}
[[ -f "$CAMP/PUSH_DISABLED" ]] && { log "disabled by $CAMP/PUSH_DISABLED"; exit 0; }
exec 7>"$CAMP/push.lock"; flock -w 600 7 || { log "lock busy"; exit 0; }
cd "$REPO" || exit 1
export GIT_TERMINAL_PROMPT=0
cur="$(git rev-parse --abbrev-ref HEAD)"
[[ "$cur" == "$BRANCH" ]] || { log "on branch $cur, expected $BRANCH; skipped"; status 0 "wrong branch $cur"; exit 0; }

# --- stage: tracked changes (incl. deletions) in the campaign areas + explicit campaign artifacts -------------
NOAUX=(':(exclude,glob)Submissions/**/main_autofilled.aux' ':(exclude,glob)Submissions/**/main_autofilled.bbl'
       ':(exclude,glob)Submissions/**/main_autofilled.blg' ':(exclude,glob)Submissions/**/main_autofilled.log'
       ':(exclude,glob)Submissions/**/main_autofilled.out')
git add -A -- RUN_STATE.md Sources/scripts Sources/artifacts Sources/journal_run.sh Sources/manifests Sources/reports \
  "Submissions/E-HSFP Journal" "Submissions/MiSFP Journal" "${NOAUX[@]}"
git add -u -- Sources/results
# gitignored-by-default (/results/**, *.md) campaign records, force-added the same way earlier campaigns were
F=()
for f in "$CAMP"/{status.json,events.log,EXPERIMENT_MATRIX.csv,EXPERIMENT_MATRIX.md,descoped.txt,push_status.json} \
         "$CAMP"/reviews/*.md "$CAMP"/reviews/review_runs.log \
         "$RES"/legacy/*.csv "$RES"/provenance/*.json "$RES"/queue/{master.queue,LANES,lane_env.sh} \
         "$SRC"/docs/DATASETS.md "$SRC"/data/DATASETS_STATUS.json; do
  [[ -f "$f" ]] && F+=("$f")
done
# markers and logs of finished runs only (running logs change every tick)
for m in "$RES"/.markers/*.done "$RES"/.markers/*.failed; do
  [[ -f "$m" ]] || continue
  F+=("$m"); e="$(basename "${m%.*}")"
  [[ -f "$RES/logs/$e.log" ]] && F+=("$RES/logs/$e.log")
done
[[ ${#F[@]} -gt 0 ]] && git add -f -- "${F[@]}"

# --- secret guard: refuse to commit anything that looks like a credential -----------------------------------
if git diff --cached -U0 | grep -E '^\+' | grep -qE 'ghp_[A-Za-z0-9]{20,}|github_pat_|CONTAINER_API_KEY=|sk-ant-|AKIA[0-9A-Z]{16}'; then
  git reset -q; log "possible credential in staged diff; nothing committed"; status 0 "secret guard tripped"; exit 1
fi

if git diff --cached --quiet; then
  log "nothing new to commit"
else
  n=$(git diff --cached --name-only | wc -l)
  sum=$(python3 -c "import json;s=json.load(open('$CAMP/status.json'));c=s.get('counts',{});print(', '.join(f\"{k}: {v.get('complete',0)} done/{v.get('running',0)} running/{v.get('queued',0)} queued/{v.get('failed',0)} failed\" for k,v in c.items()))" 2>/dev/null)
  git commit -q -m "campaign: results collection $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M KST')

Automated commit after the 2-hourly collector (scripts/campaign/push.sh); $n files.
${sum}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XBJgH9DSReRxXrJNGvuZus" || { log "commit failed"; status 0 "commit failed"; exit 1; }
  log "committed $(git rev-parse --short HEAD) ($n files)"
fi

# --- push (fast-forward only). origin's URL holds the token as a bare username, which GitHub rejects without a
# password, so push to the same repo with the token as the password; the token never reaches logs. ---------
tok="$(git remote get-url origin | sed -nE 's#^https://(ghp_[^@:]+)@.*#\1#p')"
url="$(git remote get-url origin | sed -E 's#^https://[^@]*@#https://#')"
[[ -n "$tok" ]] && url="${url/https:\/\//https://x-access-token:$tok@}"
[[ "$(git rev-parse HEAD)" == "$(git rev-parse "refs/remotes/origin/$BRANCH" 2>/dev/null)" ]] && { status 1 "up to date"; exit 0; }
out="$(timeout 300 git push "$url" "HEAD:refs/heads/$BRANCH" 2>&1)"; rc=$?
out="$(echo "$out" | sed -E 's#(ghp_|github_pat_)[A-Za-z0-9_]+#***#g' | tail -3 | tr '\n' ' ')"
if [[ $rc -eq 0 ]]; then
  git update-ref "refs/remotes/origin/$BRANCH" HEAD
  log "pushed $(git rev-parse --short HEAD) to origin/$BRANCH"; status 1 "pushed $(git rev-parse --short HEAD)"
else
  log "push failed rc=$rc: $out"; status 0 "push failed rc=$rc: $out"; exit 1
fi
