set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
for i in $(seq 1 200); do [ -f "docs/optimization_loop/logs/plan23_noniid_181501/DONE.marker" ] && break; sleep 30; done
echo "plan23 done, starting ISIC $(date -Is)" > "docs/optimization_loop/logs/plan24_isic_184200/PROGRESS.txt"
# run ISIC seg grid: 4 methods x 3 seeds, per-run timeout, resilient
for M in federated hierfl hetero-sfl h-sfp; do
  CFGNAME=$M; [ "$M" = "hetero-sfl" ] && CFGNAME=heterosfl
  for S in 20260714 20260715 20260716; do
    EXTRA=""; [ "$M" = "h-sfp" ] && EXTRA="--ablation baseline_hsfp"
    OUT="outputs/isic2018/${CFGNAME}_seed${S}"
    echo "START isic_${CFGNAME}_$S $(date -Is)" > "docs/optimization_loop/logs/plan24_isic_184200/isic_${CFGNAME}_${S}.log"; T=$(date +%s)
    timeout 5400 env CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --method $M       --cfg configs/proxy/plan_isic2018_${CFGNAME}.yaml --seed $S $EXTRA --set output_dir="$OUT" >> "docs/optimization_loop/logs/plan24_isic_184200/isic_${CFGNAME}_${S}.log" 2>&1
    echo "EXIT isic_${CFGNAME}_$S code=$? runtime_s=$(( $(date +%s)-T ))" >> "docs/optimization_loop/logs/plan24_isic_184200/isic_${CFGNAME}_${S}.log"
    echo "isic_${CFGNAME}_$S done" >> "docs/optimization_loop/logs/plan24_isic_184200/PROGRESS.txt"
  done
done
python3 tools/fair_compare/aggregate.py --dataset isic2018 outputs/isic2018/* >> "docs/optimization_loop/logs/plan24_isic_184200/aggregate.log" 2>&1
echo "ALL_DONE $(date -Is)" > "docs/optimization_loop/logs/plan24_isic_184200/DONE.marker"
