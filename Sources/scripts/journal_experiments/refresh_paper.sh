#!/usr/bin/env bash
# Hourly (cron): collect finished runs -> refill manuscript tables -> regenerate figures.
# Only recorded results are used; unfinished cells stay \pending.
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAPER="$SRC/../Submissions/E-HSFP Journal"
source "$SRC/results/journal_v1/queue/lane_env.sh"
cd "$SRC" || exit 1
echo "=== $(date -Is) refresh"
python scripts/journal_experiments/collect_journal_v1.py 2>&1 | tail -2
python scripts/journal_experiments/fill_manuscript.py 2>&1 | grep -v "^  suspect" | tail -2
cd "$PAPER" && python figures/make_figures.py 2>&1 | grep -c "^wrote" | sed 's/^/figures written: /'
# Build a PDF with the current results as main_autofilled.pdf (never touches main.pdf).
cd "$PAPER" && { pdflatex -interaction=nonstopmode -jobname=main_autofilled main.tex >/dev/null 2>&1; \
  bibtex main_autofilled >/dev/null 2>&1; \
  pdflatex -interaction=nonstopmode -jobname=main_autofilled main.tex >/dev/null 2>&1; \
  pdflatex -interaction=nonstopmode -jobname=main_autofilled main.tex > /tmp/main_autofilled_build.log 2>&1; \
  echo "pdf: $(grep -cE '^!' /tmp/main_autofilled_build.log) LaTeX errors -> $PAPER/main_autofilled.pdf"; }
