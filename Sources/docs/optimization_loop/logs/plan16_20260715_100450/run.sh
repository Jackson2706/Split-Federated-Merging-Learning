set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
# (1) hierfl s716 re-run (fixed) — 10 rounds
echo "START hierfl_s716 $(date -Is)" > "$RUNDIR/hierfl_s716.log"; T0=$(date +%s)
timeout 3600 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method hierfl \
  --cfg configs/proxy/plan15_cifar100_hierfl.yaml --seed 20260716 \
  --set output_dir=outputs/proxy/plan15_hierfl_seed20260716 >> "$RUNDIR/hierfl_s716.log" 2>&1
echo "EXIT hierfl_s716 code=$? runtime_s=$(( $(date +%s)-T0 ))" >> "$RUNDIR/hierfl_s716.log"
echo "hierfl_s716 done" >> "$RUNDIR/PROGRESS.txt"
# (2) E-HSFP deadlock bisection: ablation ladder, 2 rounds, 600s timeout each
for ABL in hsfp_memory hsfp_memory_dropout hsfp_memory_reliability hsfp_memory_reliability_prc; do
  echo "START bisect_$ABL $(date -Is)" > "$RUNDIR/bisect_${ABL}.log"; T1=$(date +%s)
  HSFP_FREEZE_BACKBONE=1 timeout 600 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
    --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation $ABL --seed 20260714 \
    --set epochs=2 --set prototype_space=centered_cosine --set num_workers=0 \
    --set proxy_id=bisect_$ABL --set output_dir=outputs/proxy/plan16_bisect_$ABL >> "$RUNDIR/bisect_${ABL}.log" 2>&1
  RC=$?
  echo "EXIT bisect_$ABL code=$RC runtime_s=$(( $(date +%s)-T1 )) (124=TIMEOUT/HANG)" >> "$RUNDIR/bisect_${ABL}.log"
  echo "bisect_$ABL code=$RC" >> "$RUNDIR/PROGRESS.txt"
done
# re-aggregate CIFAR-100
python3 tools/fair_compare/aggregate.py outputs/proxy/plan15_* outputs/proxy/plan2_cifar100_splitfl_seed20260714 outputs/proxy/plan2_cifar100_federated_seed20260714 outputs/proxy/plan11_freeze_centered_seed20260714 outputs/proxy/plan14_freeze_centered_s20260715 outputs/proxy/plan14_freeze_centered_s20260716 --output results/fair_comparison_cifar100.csv >> "$RUNDIR/aggregate.log" 2>&1
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
