set -o pipefail
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
run_proxy () {
  local NAME="$1"; local OUT="$2"; shift 2
  echo "===== START $NAME $(date -Is) =====" | tee "$RUNDIR/${NAME}.log"
  local T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
    --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 \
    --set epochs=10 --set prototype_space=raw --set proxy_id="$NAME" --set output_dir="$OUT" "$@" \
    >> "$RUNDIR/${NAME}.log" 2>&1
  echo "===== EXIT $NAME code=$? runtime_s=$(( $(date +%s)-T0 )) =====" | tee -a "$RUNDIR/${NAME}.log"
  local CKPT=$(find "$OUT" -name checkpoint.pt | head -1)
  echo "$NAME checkpoint=$CKPT" >> "$RUNDIR/paths.txt"
  if [ -n "$CKPT" ]; then
    CUDA_VISIBLE_DEVICES=0 python tools/diag/encoder_linear_probe.py \
      --config configs/proxy/plan2_cifar100_hsfp_baseline.yaml --checkpoint "$CKPT" \
      --output-dir "outputs/diag/${NAME}_probe" --seed 20260714 --device cuda --linear-epochs 10 \
      >> "$RUNDIR/${NAME}_probe.log" 2>&1
  fi
}
run_proxy plan8a_cloud2 outputs/proxy/plan8a_cloud2_seed20260714 --set syn_epochs_cloud=2
run_proxy plan8b_cloud2_ssllr3 outputs/proxy/plan8b_cloud2_ssllr3_seed20260714 --set syn_epochs_cloud=2 --set ssl_lr=1e-3
echo "ALL_DONE $(date -Is)" | tee "$RUNDIR/DONE.marker"
