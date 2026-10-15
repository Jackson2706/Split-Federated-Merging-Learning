set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
HS="--method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --seed 20260714 --set prototype_space=centered_cosine --set num_workers=0"
echo "START prc_verify $(date -Is)" > "$RUNDIR/prc_verify.log"; T=$(date +%s)
HSFP_FREEZE_BACKBONE=1 timeout 600 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification $HS --ablation hsfp_memory_reliability_prc \
  --set epochs=2 --set output_dir=outputs/proxy/plan20_prc_verify >> "$RUNDIR/prc_verify.log" 2>&1
PRC=$?; echo "EXIT prc_verify code=$PRC runtime_s=$(( $(date +%s)-T )) (124=HANG)" >> "$RUNDIR/prc_verify.log"; echo "prc_verify code=$PRC" >> "$RUNDIR/PROGRESS.txt"
if [ $PRC -eq 0 ]; then
  for MODE in iid noniid; do
    EXTRA=""; [ "$MODE" = "noniid" ] && EXTRA="--set partition=two_level_dirichlet --set alpha_edge=0.1"
    echo "START fulle_$MODE $(date -Is)" > "$RUNDIR/fulle_${MODE}.log"; T=$(date +%s)
    HSFP_FREEZE_BACKBONE=1 timeout 5400 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification $HS --ablation full_e_hsfp \
      --set epochs=10 $EXTRA --set output_dir=outputs/proxy/plan20_fulle_${MODE} >> "$RUNDIR/fulle_${MODE}.log" 2>&1
    echo "EXIT fulle_$MODE code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/fulle_${MODE}.log"; echo "fulle_$MODE done" >> "$RUNDIR/PROGRESS.txt"
  done
  # also run the intermediate ablations (10-round) for the E-HSFP ablation table
  for ABL in hsfp_memory hsfp_memory_dropout hsfp_memory_reliability hsfp_memory_reliability_prc; do
    echo "START abl_$ABL $(date -Is)" > "$RUNDIR/abl_${ABL}.log"; T=$(date +%s)
    HSFP_FREEZE_BACKBONE=1 timeout 5400 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification $HS --ablation $ABL \
      --set epochs=10 --set output_dir=outputs/proxy/plan20_abl_${ABL} >> "$RUNDIR/abl_${ABL}.log" 2>&1
    echo "EXIT abl_$ABL code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/abl_${ABL}.log"; echo "abl_$ABL done" >> "$RUNDIR/PROGRESS.txt"
  done
fi
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
