#!/bin/bash
# GPU memory per process, queue log tail, latest round / validation accuracy of each gate run.
cd "$(dirname "$0")"
echo "== GPU =="; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "== queue_status.log (tail) =="; tail -n 12 results/queue_status.log 2>/dev/null
echo "== gate runs =="
for d in results/G1_* results/G0_*; do
  [ -d "$d" ] || continue
  m="$d/metrics.csv"
  if [ -f "$m" ]; then
    last=$(tail -n 1 "$m" | cut -d, -f1,2,8)
    done_flag=$([ -f "$d/DONE" ] && echo DONE || echo running)
    acc=$(awk -F, 'NR>1 && $2!="" {a=$2; r=$1} END {print r","a}' "$m")
    echo "$(basename $d): latest row round,acc,time=$last | last val round,acc=$acc | $done_flag"
  else
    echo "$(basename $d): no metrics yet"
  fi
done
