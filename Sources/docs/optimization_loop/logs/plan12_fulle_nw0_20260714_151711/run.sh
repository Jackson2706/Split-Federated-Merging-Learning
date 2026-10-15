set -o pipefail
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"; NAME=full_e_nw0; OUT=outputs/proxy/plan12_full_e_nw0_seed20260714
echo "START $(date -Is)" | tee "$RUNDIR/${NAME}.log"; T0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
  --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation full_e_hsfp --seed 20260714 \
  --set epochs=10 --set prototype_space=centered_cosine --set num_workers=0 \
  --set proxy_id=plan12_full_e_nw0 --set output_dir=$OUT >> "$RUNDIR/${NAME}.log" 2>&1
echo "EXIT code=$? runtime_s=$(( $(date +%s)-T0 ))" | tee -a "$RUNDIR/${NAME}.log"
echo "ALL_DONE $(date -Is)" | tee "$RUNDIR/DONE.marker"
