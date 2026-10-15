set -o pipefail
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"; NAME=whitened_cosine
echo "===== START $NAME $(date -Is) =====" | tee "$RUNDIR/${NAME}.log"
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
  --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 \
  --set epochs=10 --set prototype_space=whitened_cosine \
  --set proxy_id=plan7_whitened_seed20260714 \
  --set output_dir=outputs/proxy/plan7_whitened_seed20260714 \
  >> "$RUNDIR/${NAME}.log" 2>&1
CODE=$?; echo "===== EXIT $NAME code=$CODE runtime_s=$(( $(date +%s)-T0 )) $(date -Is) =====" | tee -a "$RUNDIR/${NAME}.log"
echo "$NAME code=$CODE" > "$RUNDIR/SUMMARY.txt"
echo "ALL_DONE $(date -Is)" | tee "$RUNDIR/DONE.marker"
