set -o pipefail
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
for SEED in 20260715 20260716; do
  NAME="freeze_centered_s${SEED}"; OUT="outputs/proxy/plan14_${NAME}"
  echo "START $NAME $(date -Is)" | tee "$RUNDIR/${NAME}.log"; T0=$(date +%s)
  HSFP_FREEZE_BACKBONE=1 CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
    --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed $SEED \
    --set epochs=10 --set prototype_space=centered_cosine --set num_workers=0 \
    --set proxy_id=$NAME --set output_dir=$OUT >> "$RUNDIR/${NAME}.log" 2>&1
  echo "EXIT $NAME code=$? runtime_s=$(( $(date +%s)-T0 ))" | tee -a "$RUNDIR/${NAME}.log"
done
echo "ALL_DONE $(date -Is)" | tee "$RUNDIR/DONE.marker"
