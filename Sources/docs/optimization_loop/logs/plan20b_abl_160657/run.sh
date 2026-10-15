set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
HS="--method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --seed 20260714 --set prototype_space=centered_cosine --set num_workers=0 --set epochs=10"
for ABL in baseline_hsfp hsfp_memory hsfp_memory_dropout hsfp_memory_reliability; do
  echo "START $ABL $(date -Is)" > "$RUNDIR/${ABL}.log"; T=$(date +%s)
  HSFP_FREEZE_BACKBONE=1 timeout 5400 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification $HS --ablation $ABL \
    --set output_dir=outputs/proxy/plan20b_${ABL} >> "$RUNDIR/${ABL}.log" 2>&1
  echo "EXIT $ABL code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/${ABL}.log"; echo "$ABL done" >> "$RUNDIR/PROGRESS.txt"
done
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
