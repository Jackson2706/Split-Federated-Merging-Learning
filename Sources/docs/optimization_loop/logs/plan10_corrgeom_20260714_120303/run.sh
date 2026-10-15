set -o pipefail
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
run_one () {
  local NAME="$1"; local SPACE="$2"; local OUT="outputs/proxy/${NAME}_seed20260714"
  echo "START $NAME $(date -Is)" | tee "$RUNDIR/${NAME}.log"; local T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
    --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 \
    --set epochs=10 --set prototype_space=$SPACE --set proxy_id=$NAME --set output_dir=$OUT \
    >> "$RUNDIR/${NAME}.log" 2>&1
  echo "EXIT $NAME code=$? runtime_s=$(( $(date +%s)-T0 ))" | tee -a "$RUNDIR/${NAME}.log"
}
run_one plan10_centered_corr centered_cosine
run_one plan10_whitened_corr whitened_cosine
echo "ALL_DONE $(date -Is)" | tee "$RUNDIR/DONE.marker"
