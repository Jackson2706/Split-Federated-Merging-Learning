set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
# wait for h-sfp (plan30) to finish
for i in $(seq 1 400); do [ -f "docs/optimization_loop/logs/plan30_isicfixed_143359/DONE.marker" ] && break; pgrep -f "method h-sfp.*isic" >/dev/null || break; sleep 30; done
echo "starting heterosfl re-run $(date -Is)" > "docs/optimization_loop/logs/plan31_hetero_181446/PROGRESS.txt"
for S in 20260714 20260715 20260716; do
  echo "START isic_heterosfl_$S $(date -Is)" > "docs/optimization_loop/logs/plan31_hetero_181446/isic_heterosfl_${S}.log"; T=$(date +%s)
  timeout 3000 env CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --method hetero-sfl     --cfg configs/proxy/plan_isic2018_heterosfl.yaml --seed $S --set output_dir="outputs/isic2018/heterosfl_seed${S}_v3" >> "docs/optimization_loop/logs/plan31_hetero_181446/isic_heterosfl_${S}.log" 2>&1
  echo "EXIT isic_heterosfl_$S code=$? runtime_s=$(( $(date +%s)-T ))" >> "docs/optimization_loop/logs/plan31_hetero_181446/isic_heterosfl_${S}.log"; echo "isic_heterosfl_$S done" >> "docs/optimization_loop/logs/plan31_hetero_181446/PROGRESS.txt"
done
echo "ALL_DONE $(date -Is)" > "docs/optimization_loop/logs/plan31_hetero_181446/DONE.marker"
