set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
run () {  # name method cfg seed
  local NAME="$1" METHOD="$2" CFG="$3" SEED="$4"
  local OUT="outputs/proxy/plan15_${NAME}_seed${SEED}"
  echo "START $NAME s$SEED $(date -Is)" > "$RUNDIR/${NAME}_${SEED}.log"; local T0=$(date +%s)
  timeout 3600 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method "$METHOD" \
    --cfg "$CFG" --seed "$SEED" --set output_dir="$OUT" >> "$RUNDIR/${NAME}_${SEED}.log" 2>&1
  echo "EXIT $NAME s$SEED code=$? runtime_s=$(( $(date +%s)-T0 ))" >> "$RUNDIR/${NAME}_${SEED}.log"
  echo "$NAME s$SEED done" >> "$RUNDIR/PROGRESS.txt"
}
for S in 20260715 20260716; do
  run splitfl   splitfl    configs/proxy/plan2_cifar100_splitfl.yaml   $S
  run federated federated  configs/proxy/plan2_cifar100_federated.yaml $S
done
for S in 20260714 20260715 20260716; do
  run hierfl    hierfl     configs/proxy/plan15_cifar100_hierfl.yaml    $S
  run hsfl      hsfl       configs/proxy/plan15_cifar100_hsfl.yaml      $S
  run heterosfl hetero-sfl configs/proxy/plan15_cifar100_heterosfl.yaml $S
done
# aggregate everything
python3 tools/fair_compare/aggregate.py \
  outputs/proxy/plan15_* \
  outputs/proxy/plan2_cifar100_splitfl_seed20260714 \
  outputs/proxy/plan2_cifar100_federated_seed20260714 \
  outputs/proxy/plan11_freeze_centered_seed20260714 \
  outputs/proxy/plan14_freeze_centered_s20260715 \
  outputs/proxy/plan14_freeze_centered_s20260716 \
  --output results/fair_comparison_cifar100.csv >> "$RUNDIR/aggregate.log" 2>&1
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
