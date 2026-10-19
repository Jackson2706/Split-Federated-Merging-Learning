set -o pipefail
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
# A: freeze-backbone + centered_cosine
echo "START freeze $(date -Is)" | tee "$RUNDIR/freeze_centered.log"; T0=$(date +%s)
HSFP_FREEZE_BACKBONE=1 CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
  --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 \
  --set epochs=10 --set prototype_space=centered_cosine --set proxy_id=plan11_freeze_centered \
  --set output_dir=outputs/proxy/plan11_freeze_centered_seed20260714 >> "$RUNDIR/freeze_centered.log" 2>&1
echo "EXIT freeze code=$? runtime_s=$(( $(date +%s)-T0 ))" | tee -a "$RUNDIR/freeze_centered.log"
# B: full_e_hsfp + centered_cosine (E-HSFP contribution on corrected eval)
echo "START full_e $(date -Is)" | tee "$RUNDIR/full_e_centered.log"; T1=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
  --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation full_e_hsfp --seed 20260714 \
  --set epochs=10 --set prototype_space=centered_cosine --set proxy_id=plan11_full_e_centered \
  --set output_dir=outputs/proxy/plan11_full_e_centered_seed20260714 >> "$RUNDIR/full_e_centered.log" 2>&1
echo "EXIT full_e code=$? runtime_s=$(( $(date +%s)-T1 ))" | tee -a "$RUNDIR/full_e_centered.log"
echo "ALL_DONE $(date -Is)" | tee "$RUNDIR/DONE.marker"
