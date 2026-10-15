set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="docs/optimization_loop/logs/plan22_prcfull_165603"; PREV="docs/optimization_loop/logs/plan20b_abl_160657"
# wait for the working-ablations batch to finish (GPU free)
for i in $(seq 1 120); do [ -f "$PREV/DONE.marker" ] && break; sleep 30; done
HS="--method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --seed 20260714 --set prototype_space=centered_cosine --set num_workers=0"
# verify Codex PRC-backward fix
echo "START prc_verify $(date -Is)" > "$RUNDIR/prc_verify.log"; T=$(date +%s)
HSFP_FREEZE_BACKBONE=1 timeout 900 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification $HS --ablation hsfp_memory_reliability_prc   --set epochs=2 --set output_dir=outputs/proxy/plan22_prc_verify >> "$RUNDIR/prc_verify.log" 2>&1
PRC=$?; echo "EXIT prc_verify code=$PRC runtime_s=$(( $(date +%s)-T )) (124=HANG)" >> "$RUNDIR/prc_verify.log"; echo "prc_verify code=$PRC" >> "$RUNDIR/PROGRESS.txt"
if [ $PRC -eq 0 ]; then
  echo "START abl_prc10 $(date -Is)" > "$RUNDIR/abl_prc10.log"; T=$(date +%s)
  HSFP_FREEZE_BACKBONE=1 timeout 5400 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification $HS --ablation hsfp_memory_reliability_prc     --set epochs=10 --set output_dir=outputs/proxy/plan22_abl_prc10 >> "$RUNDIR/abl_prc10.log" 2>&1
  echo "EXIT abl_prc10 code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/abl_prc10.log"; echo "abl_prc10 done" >> "$RUNDIR/PROGRESS.txt"
  for MODE in iid noniid; do
    EXTRA=""; [ "$MODE" = "noniid" ] && EXTRA="--set partition=two_level_dirichlet --set alpha_edge=0.1"
    echo "START fulle_$MODE $(date -Is)" > "$RUNDIR/fulle_${MODE}.log"; T=$(date +%s)
    HSFP_FREEZE_BACKBONE=1 timeout 5400 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification $HS --ablation full_e_hsfp       --set epochs=10 $EXTRA --set output_dir=outputs/proxy/plan22_fulle_${MODE} >> "$RUNDIR/fulle_${MODE}.log" 2>&1
    echo "EXIT fulle_$MODE code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/fulle_${MODE}.log"; echo "fulle_$MODE done" >> "$RUNDIR/PROGRESS.txt"
  done
else echo "PRC STILL HANGS after fix5" >> "$RUNDIR/PROGRESS.txt"; fi
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
