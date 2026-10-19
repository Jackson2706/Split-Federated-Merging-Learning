set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
HS="--cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation full_e_hsfp --seed 20260714 --set prototype_space=centered_cosine --set num_workers=0"
# (0) hierfl s716 fresh dir
echo "START hierfl_s716b $(date -Is)" > "$RUNDIR/hierfl_s716b.log"; T=$(date +%s)
timeout 3600 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method hierfl \
  --cfg configs/proxy/plan15_cifar100_hierfl.yaml --seed 20260716 --set output_dir=outputs/proxy/plan17_hierfl_seed20260716 >> "$RUNDIR/hierfl_s716b.log" 2>&1
echo "EXIT hierfl_s716b code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/hierfl_s716b.log"; echo "hierfl_s716b done" >> "$RUNDIR/PROGRESS.txt"
# (1) full_e 2-round PRC-fix verify
echo "START verify $(date -Is)" > "$RUNDIR/verify.log"; T=$(date +%s)
HSFP_FREEZE_BACKBONE=1 timeout 600 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp $HS \
  --set epochs=2 --set proxy_id=plan17_verify --set output_dir=outputs/proxy/plan17_fulle_verify >> "$RUNDIR/verify.log" 2>&1
VRC=$?; echo "EXIT verify code=$VRC runtime_s=$(( $(date +%s)-T )) (124=HANG)" >> "$RUNDIR/verify.log"; echo "verify code=$VRC" >> "$RUNDIR/PROGRESS.txt"
if [ $VRC -eq 0 ]; then
  # (2) full_e 10-round IID
  echo "START fulle_iid $(date -Is)" > "$RUNDIR/fulle_iid.log"; T=$(date +%s)
  HSFP_FREEZE_BACKBONE=1 timeout 5400 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp $HS \
    --set epochs=10 --set proxy_id=plan17_fulle_iid --set output_dir=outputs/proxy/plan17_fulle_iid >> "$RUNDIR/fulle_iid.log" 2>&1
  echo "EXIT fulle_iid code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/fulle_iid.log"; echo "fulle_iid done" >> "$RUNDIR/PROGRESS.txt"
  # (3) full_e 10-round NON-IID (dirichlet)
  echo "START fulle_noniid $(date -Is)" > "$RUNDIR/fulle_noniid.log"; T=$(date +%s)
  HSFP_FREEZE_BACKBONE=1 timeout 5400 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp $HS \
    --set epochs=10 --set partition=two_level_dirichlet --set alpha_edge=0.1 \
    --set proxy_id=plan17_fulle_noniid --set output_dir=outputs/proxy/plan17_fulle_noniid >> "$RUNDIR/fulle_noniid.log" 2>&1
  echo "EXIT fulle_noniid code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/fulle_noniid.log"; echo "fulle_noniid done" >> "$RUNDIR/PROGRESS.txt"
else
  echo "VERIFY HUNG — full_e still deadlocks after PRC fix; skipping 10-round runs" >> "$RUNDIR/PROGRESS.txt"
fi
python3 tools/fair_compare/aggregate.py outputs/proxy/plan15_* outputs/proxy/plan17_hierfl_seed20260716 outputs/proxy/plan2_cifar100_splitfl_seed20260714 outputs/proxy/plan2_cifar100_federated_seed20260714 outputs/proxy/plan11_freeze_centered_seed20260714 outputs/proxy/plan14_freeze_centered_s20260715 outputs/proxy/plan14_freeze_centered_s20260716 --output results/fair_comparison_cifar100.csv >> "$RUNDIR/aggregate.log" 2>&1
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
