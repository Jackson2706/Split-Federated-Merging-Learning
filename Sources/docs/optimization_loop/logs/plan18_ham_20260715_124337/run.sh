set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
# (diag) PRC fix re-test at the exact bisection point
echo "START prc_retest $(date -Is)" > "$RUNDIR/prc_retest.log"; T=$(date +%s)
HSFP_FREEZE_BACKBONE=1 timeout 600 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
  --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation hsfp_memory_reliability_prc --seed 20260714 \
  --set epochs=2 --set prototype_space=centered_cosine --set num_workers=0 \
  --set proxy_id=plan18_prc_retest --set output_dir=outputs/proxy/plan18_prc_retest >> "$RUNDIR/prc_retest.log" 2>&1
echo "EXIT prc_retest code=$? runtime_s=$(( $(date +%s)-T )) (124=HANG)" >> "$RUNDIR/prc_retest.log"; echo "prc_retest $(tail -1 $RUNDIR/prc_retest.log)" >> "$RUNDIR/PROGRESS.txt"
# HAM10000 grid: 6 methods x 3 seeds
ham () { # name method extra
  local NAME="$1" METHOD="$2"; shift 2
  for S in 20260714 20260715 20260716; do
    local OUT="outputs/proxy/plan_ham10000_${NAME}_seed${S}"
    echo "START ham_${NAME}_$S $(date -Is)" > "$RUNDIR/ham_${NAME}_${S}.log"; local T0=$(date +%s)
    timeout 3600 env CUDA_VISIBLE_DEVICES=0 "$@" python main.py --task classification --method "$METHOD" \
      --cfg configs/proxy/plan_ham10000_${NAME}.yaml --seed $S --set output_dir="$OUT" >> "$RUNDIR/ham_${NAME}_${S}.log" 2>&1
    echo "EXIT ham_${NAME}_$S code=$? runtime_s=$(( $(date +%s)-T0 ))" >> "$RUNDIR/ham_${NAME}_${S}.log"
    echo "ham_${NAME}_$S done" >> "$RUNDIR/PROGRESS.txt"
  done
}
# h-sfp needs freeze + ablation + centered + nw0
for S in 20260714 20260715 20260716; do
  OUT="outputs/proxy/plan_ham10000_hsfp_seed${S}"
  echo "START ham_hsfp_$S $(date -Is)" > "$RUNDIR/ham_hsfp_${S}.log"; T0=$(date +%s)
  HSFP_FREEZE_BACKBONE=1 timeout 3600 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
    --cfg configs/proxy/plan_ham10000_hsfp.yaml --seed $S --ablation baseline_hsfp \
    --set prototype_space=centered_cosine --set num_workers=0 --set output_dir="$OUT" >> "$RUNDIR/ham_hsfp_${S}.log" 2>&1
  echo "EXIT ham_hsfp_$S code=$? runtime_s=$(( $(date +%s)-T0 ))" >> "$RUNDIR/ham_hsfp_${S}.log"; echo "ham_hsfp_$S done" >> "$RUNDIR/PROGRESS.txt"
done
ham splitfl splitfl
ham federated federated
ham hierfl hierfl
ham hsfl hsfl
ham heterosfl hetero-sfl
python3 tools/fair_compare/aggregate.py --dataset ham10000 outputs/proxy/plan_ham10000_* >> "$RUNDIR/aggregate.log" 2>&1
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
