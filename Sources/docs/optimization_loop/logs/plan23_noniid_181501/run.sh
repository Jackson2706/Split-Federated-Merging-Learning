set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
for i in $(seq 1 120); do [ -f "docs/optimization_loop/logs/plan22_prcfull_165603/DONE.marker" ] && break; sleep 30; done
HS="--method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --seed 20260714 --set prototype_space=centered_cosine --set num_workers=0 --set epochs=10 --set partition=two_level_dirichlet --set alpha_edge=0.1"
for ABL in baseline_hsfp hsfp_memory_dropout hsfp_memory_reliability; do
  echo "START noniid_$ABL $(date -Is)" > "docs/optimization_loop/logs/plan23_noniid_181501/noniid_${ABL}.log"; T=$(date +%s)
  HSFP_FREEZE_BACKBONE=1 timeout 5400 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification $HS --ablation $ABL     --set output_dir=outputs/proxy/plan23_noniid_${ABL} >> "docs/optimization_loop/logs/plan23_noniid_181501/noniid_${ABL}.log" 2>&1
  echo "EXIT noniid_$ABL code=$? runtime_s=$(( $(date +%s)-T ))" >> "docs/optimization_loop/logs/plan23_noniid_181501/noniid_${ABL}.log"; echo "noniid_$ABL done" >> "docs/optimization_loop/logs/plan23_noniid_181501/PROGRESS.txt"
done
echo "ALL_DONE $(date -Is)" > "docs/optimization_loop/logs/plan23_noniid_181501/DONE.marker"
