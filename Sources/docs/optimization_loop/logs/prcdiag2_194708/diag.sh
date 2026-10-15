cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"; LOG="$RUNDIR/prc.log"
HSFP_FREEZE_BACKBONE=1 CUDA_VISIBLE_DEVICES=0 python -X faulthandler -u main.py --task classification --method h-sfp \
  --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation hsfp_memory_reliability_prc --seed 20260714 \
  --set epochs=2 --set prototype_space=centered_cosine --set num_workers=0 \
  --set output_dir=outputs/proxy/prcdiag2 > "$LOG" 2>&1 &
TPID=$!
for i in $(seq 1 36); do
  sleep 10
  BIG=$(for p in $(pgrep -f prcdiag2 2>/dev/null); do r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); echo "${r:-0} $p"; done | sort -rn | head -1)
  BPID=$(echo "$BIG" | awk '{print $2}'); BRSS=$(echo "$BIG" | awk '{print $1}')
  BW=$(cat /proc/$BPID/wchan 2>/dev/null)
  echo "t=$((i*10))s pid=$BPID rss=${BRSS}kB wchan=$BW" >> "$RUNDIR/watch.log"
  if [ "$BW" = "futex_wait_queue" ] && [ "${BRSS:-0}" -gt 1500000 ]; then
    echo ">>> HANG at $((i*10))s: SIGABRT $BPID" >> "$RUNDIR/watch.log"
    kill -ABRT "$BPID"; sleep 8; break
  fi
  if ! kill -0 "$TPID" 2>/dev/null && [ $i -gt 2 ]; then echo "trainer exited (completed?)" >> "$RUNDIR/watch.log"; break; fi
done
pkill -9 -f prcdiag2 2>/dev/null
echo "DONE $(date -Is)" > "$RUNDIR/DONE.marker"
