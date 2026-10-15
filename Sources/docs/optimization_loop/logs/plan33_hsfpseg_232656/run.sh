set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
# wait for plan32 to fully finish (frees GPU)
for i in $(seq 1 400); do [ -f "docs/optimization_loop/logs/plan32_finalize_184531/DONE.marker" ] && break; sleep 30; done
echo "plan32 done, corrected h-sfp seg re-run start $(date -Is)" > "docs/optimization_loop/logs/plan33_hsfpseg_232656/PROGRESS.txt"
for S in 20260714 20260715 20260716; do
  echo "START hsfp_$S $(date -Is)" > "docs/optimization_loop/logs/plan33_hsfpseg_232656/hsfp_${S}.log"; T=$(date +%s)
  timeout 6000 env CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --method h-sfp     --cfg configs/proxy/plan_isic2018_hsfp.yaml --seed $S --ablation baseline_hsfp     --set output_dir="outputs/isic2018/hsfp_seed${S}_fixed" >> "docs/optimization_loop/logs/plan33_hsfpseg_232656/hsfp_${S}.log" 2>&1
  echo "EXIT hsfp_$S code=$? runtime_s=$(( $(date +%s)-T ))" >> "docs/optimization_loop/logs/plan33_hsfpseg_232656/hsfp_${S}.log"; echo "hsfp_$S done" >> "docs/optimization_loop/logs/plan33_hsfpseg_232656/PROGRESS.txt"
done
# final combined aggregate across all 4 methods (best-known dirs)
python3 tools/fair_compare/aggregate.py --dataset isic2018 \
  outputs/isic2018/federated_seed*_v2 outputs/isic2018/hierfl_seed*_v2 \
  outputs/isic2018/heterosfl_seed*_v3 outputs/isic2018/hsfp_seed*_fixed \
  --output results/fair_comparison_isic2018.csv >> "docs/optimization_loop/logs/plan33_hsfpseg_232656/aggregate.log" 2>&1
echo "ALL_DONE $(date -Is)" > "docs/optimization_loop/logs/plan33_hsfpseg_232656/DONE.marker"
