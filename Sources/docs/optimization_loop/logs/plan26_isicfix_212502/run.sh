set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="docs/optimization_loop/logs/plan26_isicfix_212502"
# (0) Dice-fix verify: federated ISIC, 2 rounds
echo "START verify $(date -Is)" > "$RUNDIR/verify.log"; T=$(date +%s)
timeout 1800 env CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --method federated   --cfg configs/proxy/plan_isic2018_federated.yaml --seed 20260714 --set epochs=2   --set output_dir=outputs/isic2018_verify/federated >> "$RUNDIR/verify.log" 2>&1
VC=$?; NAN=$(grep -ciE "NaN/Inf" "$RUNDIR/verify.log")
echo "EXIT verify code=$VC nan_lines=$NAN runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/verify.log"
echo "verify code=$VC nan=$NAN" >> "$RUNDIR/PROGRESS.txt"
if [ $VC -eq 0 ] && [ "$NAN" -eq 0 ]; then
  # (1) full ISIC grid: 4 methods x 3 seeds
  for M in federated hierfl hetero-sfl h-sfp; do
    CFG=$M; [ "$M" = "hetero-sfl" ] && CFG=heterosfl
    for S in 20260714 20260715 20260716; do
      EX=""; [ "$M" = "h-sfp" ] && EX="--ablation baseline_hsfp"
      echo "START isic_${CFG}_$S $(date -Is)" > "$RUNDIR/isic_${CFG}_${S}.log"; T=$(date +%s)
      timeout 7200 env CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --method $M         --cfg configs/proxy/plan_isic2018_${CFG}.yaml --seed $S $EX --set output_dir="outputs/isic2018/${CFG}_seed${S}" >> "$RUNDIR/isic_${CFG}_${S}.log" 2>&1
      echo "EXIT isic_${CFG}_$S code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/isic_${CFG}_${S}.log"; echo "isic_${CFG}_$S done" >> "$RUNDIR/PROGRESS.txt"
    done
  done
  python3 tools/fair_compare/aggregate.py --dataset isic2018 outputs/isic2018/* >> "$RUNDIR/aggregate.log" 2>&1
  # (2) full-60 CIFAR grid
  for S in 20260714 20260715 20260716; do
    for spec in "splitfl|splitfl|configs/proxy/plan2_cifar100_splitfl.yaml||" "federated|federated|configs/proxy/plan2_cifar100_federated.yaml||" "hierfl|hierfl|configs/proxy/plan15_cifar100_hierfl.yaml||" "hsfl|hsfl|configs/proxy/plan15_cifar100_hsfl.yaml||" "heterosfl|hetero-sfl|configs/proxy/plan15_cifar100_heterosfl.yaml||" "hsfp|h-sfp|configs/proxy/plan2_cifar100_hsfp_baseline.yaml|HSFP_FREEZE_BACKBONE=1|--ablation baseline_hsfp --set prototype_space=centered_cosine --set num_workers=0"; do
      NAME=$(echo "$spec"|cut -d'|' -f1); METH=$(echo "$spec"|cut -d'|' -f2); CFG=$(echo "$spec"|cut -d'|' -f3); ENVX=$(echo "$spec"|cut -d'|' -f4); ARGS=$(echo "$spec"|cut -d'|' -f5)
      echo "START f60_${NAME}_$S $(date -Is)" > "$RUNDIR/f60_${NAME}_${S}.log"; T=$(date +%s)
      $ENVX timeout 36000 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method $METH         --cfg $CFG --seed $S --set epochs=60 $ARGS --set output_dir="outputs/full60/${NAME}_seed${S}" >> "$RUNDIR/f60_${NAME}_${S}.log" 2>&1
      echo "EXIT f60_${NAME}_$S code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/f60_${NAME}_${S}.log"; echo "f60_${NAME}_$S done" >> "$RUNDIR/PROGRESS.txt"
    done
  done
  python3 tools/fair_compare/aggregate.py outputs/full60/* --output results/fair_comparison_cifar100_full60.csv >> "$RUNDIR/aggregate.log" 2>&1
else
  echo "DICE-FIX VERIFY FAILED (code=$VC nan=$NAN) — NOT launching ISIC/full-60 grid" >> "$RUNDIR/PROGRESS.txt"
fi
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
