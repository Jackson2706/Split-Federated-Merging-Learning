set -o pipefail
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"; NAME=raw_evalfix; OUT=outputs/proxy/plan9_raw_evalfix_seed20260714
echo "START $(date -Is)" | tee "$RUNDIR/${NAME}.log"; T0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
  --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 \
  --set epochs=10 --set prototype_space=raw --set proxy_id=plan9_raw_evalfix --set output_dir=$OUT \
  >> "$RUNDIR/${NAME}.log" 2>&1
echo "EXIT code=$? runtime_s=$(( $(date +%s)-T0 ))" | tee -a "$RUNDIR/${NAME}.log"
CKPT=$(find $OUT -name checkpoint.pt | head -1)
CUDA_VISIBLE_DEVICES=0 python tools/diag/encoder_linear_probe.py --config configs/proxy/plan2_cifar100_hsfp_baseline.yaml \
  --checkpoint "$CKPT" --output-dir outputs/diag/plan9_raw_evalfix_probe --seed 20260714 --device cuda --linear-epochs 10 \
  >> "$RUNDIR/${NAME}_probe.log" 2>&1
echo "ALL_DONE $(date -Is)" | tee "$RUNDIR/DONE.marker"
