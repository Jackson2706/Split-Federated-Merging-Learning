set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
P30="docs/optimization_loop/logs/plan30_isicfixed_143359"; P31="docs/optimization_loop/logs/plan31_hetero_181446"; RUNDIR="docs/optimization_loop/logs/plan32_finalize_184531"
# wait for both chains to finish (max ~5h)
for i in $(seq 1 600); do
  [ -f "$P30/DONE.marker" ] && [ -f "$P31/DONE.marker" ] && break
  sleep 30
done
echo "both prior chains done $(date -Is)" > "$RUNDIR/PROGRESS.txt"

declare -A CFG=( [federated]=federated [hierfl]=hierfl [heterosfl]=heterosfl [hsfp]=hsfp )
declare -A METH=( [federated]=federated [hierfl]=hierfl [heterosfl]=hetero-sfl [hsfp]=h-sfp )

check_ok () { # method seed -> prints "OK" or "MISSING"
  local M="$1" S="$2"
  for L in "$P30/isic_${M}_${S}.log" "$P31/isic_${M}_${S}.log"; do
    [ -f "$L" ] || continue
    grep -qE "EXIT isic_${M}_${S} code=0" "$L" && { echo OK; return; }
  done
  echo MISSING
}

for M in federated hierfl heterosfl hsfp; do
  for S in 20260714 20260715 20260716; do
    STATUS=$(check_ok "$M" "$S")
    echo "$M $S: $STATUS" >> "$RUNDIR/PROGRESS.txt"
    if [ "$STATUS" = "MISSING" ]; then
      EX=""; [ "$M" = "hsfp" ] && EX="--ablation baseline_hsfp"
      echo "START retry_${M}_$S $(date -Is)" > "$RUNDIR/retry_${M}_${S}.log"; T=$(date +%s)
      timeout 6000 env CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --method ${METH[$M]}         --cfg configs/proxy/plan_isic2018_${CFG[$M]}.yaml --seed $S $EX --set output_dir="outputs/isic2018/${CFG[$M]}_seed${S}_final" >> "$RUNDIR/retry_${M}_${S}.log" 2>&1
      echo "EXIT retry_${M}_$S code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/retry_${M}_${S}.log"
      echo "retry_${M}_$S done" >> "$RUNDIR/PROGRESS.txt"
    fi
  done
done

# final aggregate: prefer _final (retry) dirs, fall back to _v2/_v3 successes
python3 tools/fair_compare/aggregate.py --dataset isic2018 \
  outputs/isic2018/federated_seed*_v2 outputs/isic2018/hierfl_seed*_v2 \
  outputs/isic2018/heterosfl_seed*_v3 outputs/isic2018/hsfp_seed*_v2 \
  outputs/isic2018/*_final \
  --output results/fair_comparison_isic2018.csv >> "$RUNDIR/aggregate.log" 2>&1
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
