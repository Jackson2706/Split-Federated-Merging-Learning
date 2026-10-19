set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
for i in $(seq 1 2000); do [ -f "docs/optimization_loop/logs/plan26_isicfix_212502/DONE.marker" ] && break; sleep 30; done
echo "prev chain done, H-SFP full-60 start $(date -Is)" > "docs/optimization_loop/logs/plan27_hsfp60_152010/PROGRESS.txt"
for S in 20260714 20260715 20260716; do
  echo "START hsfp60_$S $(date -Is)" > "docs/optimization_loop/logs/plan27_hsfp60_152010/hsfp60_${S}.log"; T=$(date +%s)
  env HSFP_FREEZE_BACKBONE=1 CUDA_VISIBLE_DEVICES=0 timeout 36000 python main.py --task classification --method h-sfp     --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --seed $S --ablation baseline_hsfp     --set epochs=60 --set prototype_space=centered_cosine --set num_workers=0     --set output_dir=outputs/full60/hsfp_seed${S} >> "docs/optimization_loop/logs/plan27_hsfp60_152010/hsfp60_${S}.log" 2>&1
  echo "EXIT hsfp60_$S code=$? runtime_s=$(( $(date +%s)-T ))" >> "docs/optimization_loop/logs/plan27_hsfp60_152010/hsfp60_${S}.log"; echo "hsfp60_$S done" >> "docs/optimization_loop/logs/plan27_hsfp60_152010/PROGRESS.txt"
done
echo "ALL_DONE $(date -Is)" > "docs/optimization_loop/logs/plan27_hsfp60_152010/DONE.marker"
