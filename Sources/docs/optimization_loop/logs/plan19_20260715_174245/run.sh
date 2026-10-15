set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
# (1) HAM h-sfp x3 with fixed logger
for S in 20260714 20260715 20260716; do
  OUT="outputs/proxy/plan19_ham_hsfp_seed${S}"
  echo "START ham_hsfp_$S $(date -Is)" > "$RUNDIR/ham_hsfp_${S}.log"; T=$(date +%s)
  HSFP_FREEZE_BACKBONE=1 timeout 3600 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
    --cfg configs/proxy/plan_ham10000_hsfp.yaml --seed $S --ablation baseline_hsfp \
    --set prototype_space=centered_cosine --set num_workers=0 --set output_dir="$OUT" >> "$RUNDIR/ham_hsfp_${S}.log" 2>&1
  echo "EXIT ham_hsfp_$S code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/ham_hsfp_${S}.log"; echo "ham_hsfp_$S done" >> "$RUNDIR/PROGRESS.txt"
done
# (2) E-HSFP deadlock-fix verify (prc ablation, 2 rounds)
echo "START prc_verify $(date -Is)" > "$RUNDIR/prc_verify.log"; T=$(date +%s)
HSFP_FREEZE_BACKBONE=1 timeout 600 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
  --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation hsfp_memory_reliability_prc --seed 20260714 \
  --set epochs=2 --set prototype_space=centered_cosine --set num_workers=0 \
  --set output_dir=outputs/proxy/plan19_prc_verify >> "$RUNDIR/prc_verify.log" 2>&1
PRC=$?; echo "EXIT prc_verify code=$PRC runtime_s=$(( $(date +%s)-T )) (124=HANG)" >> "$RUNDIR/prc_verify.log"; echo "prc_verify code=$PRC" >> "$RUNDIR/PROGRESS.txt"
if [ $PRC -eq 0 ]; then
  for MODE in iid noniid; do
    EXTRA=""; [ "$MODE" = "noniid" ] && EXTRA="--set partition=two_level_dirichlet --set alpha_edge=0.1"
    echo "START fulle_$MODE $(date -Is)" > "$RUNDIR/fulle_${MODE}.log"; T=$(date +%s)
    HSFP_FREEZE_BACKBONE=1 timeout 5400 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
      --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation full_e_hsfp --seed 20260714 \
      --set epochs=10 --set prototype_space=centered_cosine --set num_workers=0 $EXTRA \
      --set output_dir=outputs/proxy/plan19_fulle_${MODE} >> "$RUNDIR/fulle_${MODE}.log" 2>&1
    echo "EXIT fulle_$MODE code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/fulle_${MODE}.log"; echo "fulle_$MODE done" >> "$RUNDIR/PROGRESS.txt"
  done
else
  echo "PRC STILL HANGS after Codex fix — need deeper debug" >> "$RUNDIR/PROGRESS.txt"
fi
python3 tools/fair_compare/aggregate.py --dataset ham10000 outputs/proxy/plan_ham10000_* outputs/proxy/plan19_ham_hsfp_* >> "$RUNDIR/aggregate.log" 2>&1
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
