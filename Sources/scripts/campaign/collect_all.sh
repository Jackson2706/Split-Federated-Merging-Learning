#!/usr/bin/env bash
# Deterministic results collector for BOTH journal submissions (no LLM involved).
#   scripts/campaign/collect_all.sh [reason]       reason: cron | manual | final | ...
# Cron runs it every 2 h at even Asia/Seoul hours; queue_lane.sh calls it with reason=final when the
# last lane drains the queue. A flock prevents overlapping runs. Steps:
#   1 rebuild master.queue (drops runs that became .done)   2 E-HSFP collect -> tables -> figures
#   3 MiSFP collect -> measurements.csv -> validate -> table rows/figures   4 build both PDFs as
#   main_autofilled.pdf (never touches main.pdf)   5 status.json / events.log / RUN_STATE.md
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(dirname "$SRC")"
CAMP="$SRC/results/campaign"; mkdir -p "$CAMP/collect_logs"
exec 9>"$CAMP/collector.lock"
flock -n 9 || { echo "$(TZ=Asia/Seoul date -Is) collector already running; skip"; exit 0; }
# shellcheck source=/dev/null
source "$SRC/results/journal_v1/queue/lane_env.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl-collector
REASON="${1:-manual}"
TS="$(TZ=Asia/Seoul date +%Y%m%dT%H%M%S)"
LOG="$CAMP/collect_logs/collect_${TS}_${REASON}.log"
# next even KST hour
NEXT="$(python3 -c "
import datetime as d, zoneinfo as z
n=d.datetime.now(z.ZoneInfo('Asia/Seoul')).replace(minute=0,second=0,microsecond=0)
n+=d.timedelta(hours=1 if n.hour%2 else 2); print(n.isoformat())")"
rc=0
{
echo "=== $(TZ=Asia/Seoul date -Is) collect reason=$REASON"
cd "$SRC" || exit 1
python scripts/campaign/build_queue.py || rc=1
echo "--- E-HSFP collect"
python scripts/journal_experiments/collect_journal_v1.py 2>&1 | grep -v UserWarning | tail -3 || rc=1
python scripts/journal_experiments/fill_manuscript.py 2>&1 | grep -v "^  suspect" | tail -3 || rc=1
( cd "$REPO/Submissions/E-HSFP Journal" && python figures/make_figures.py 2>&1 | tail -30 ) || rc=1
echo "--- MiSFP collect"
python scripts/campaign/collect_misfp.py || rc=1
( cd "$REPO/Submissions/MiSFP Journal" && python scripts/generate_results.py | head -8 ) || { echo "MiSFP validation reported errors"; rc=1; }
echo "--- LaTeX builds"
BUILDS="$CAMP/builds.json"; echo "{}" > "$BUILDS"
for paper in "E-HSFP Journal" "MiSFP Journal"; do
  d="$REPO/Submissions/$paper"
  ( cd "$d" && pdflatex -interaction=nonstopmode -jobname=main_autofilled main.tex >/dev/null 2>&1
    bibtex main_autofilled >/dev/null 2>&1
    pdflatex -interaction=nonstopmode -jobname=main_autofilled main.tex >/dev/null 2>&1
    pdflatex -interaction=nonstopmode -jobname=main_autofilled main.tex >/dev/null 2>&1 )
  python3 - "$d" "$paper" "$BUILDS" <<'EOF'
import json, re, sys, os
d, paper, out = sys.argv[1:]
log = open(os.path.join(d, "main_autofilled.log"), errors="replace").read() if os.path.exists(os.path.join(d, "main_autofilled.log")) else ""
blg = open(os.path.join(d, "main_autofilled.blg"), errors="replace").read() if os.path.exists(os.path.join(d, "main_autofilled.blg")) else ""
m = re.search(r"Output written on main_autofilled.pdf \((\d+) pages", log)
r = {"pdf": f"{d}/main_autofilled.pdf" if m else None, "pages": int(m.group(1)) if m else 0,
     "latex_errors": len(re.findall(r"^!", log, re.M)),
     "undefined_citations": sorted(set(re.findall(r"Citation `([^']+)' on page \d+ undefined", log))),
     "undefined_references": sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log))),
     "bibtex_errors": len(re.findall(r"error message", blg)) and int((re.findall(r"There were (\d+) error", blg) or ["1"])[0])}
j = json.load(open(out)); j[paper] = r; json.dump(j, open(out, "w"), indent=2)
print(paper, {k: (v if not isinstance(v, list) else len(v)) for k, v in r.items()})
EOF
done
} > "$LOG" 2>&1 || rc=1
python scripts/campaign/campaign_status.py --builds "$CAMP/builds.json" --collector "$rc" --next "$NEXT" \
  --event "collector reason=$REASON rc=$rc log=${LOG#$SRC/}" >> "$LOG" 2>&1
ln -sf "$LOG" "$CAMP/collect_logs/latest.log"
python scripts/campaign/update_report.py "$REASON" >> "$LOG" 2>&1   # what-changed report (LATEST_UPDATE.md, UPDATES.md)
echo "$(TZ=Asia/Seoul date -Is) collector done rc=$rc log=$LOG"
exit $rc
