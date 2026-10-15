set -o pipefail
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
ATTEMPT="plan1c_hostgpu_v1"
for PRESET in baseline_hsfp full_e_hsfp; do
  echo "===== START $PRESET $(date -Is) =====" | tee "$RUNDIR/${PRESET}.log"
  CUDA_VISIBLE_DEVICES=0 python main.py \
    --task classification --method h-sfp \
    --cfg configs/smoke/plan1_cifar100_ehsfp_gate.yaml \
    --ablation "$PRESET" --seed 20260713 \
    --set smoke_attempt_id="$ATTEMPT" \
    >> "$RUNDIR/${PRESET}.log" 2>&1
  echo "===== EXIT $PRESET code=$? $(date -Is) =====" | tee -a "$RUNDIR/${PRESET}.log"
done
echo "ALL_DONE $(date -Is)" | tee "$RUNDIR/DONE.marker"
