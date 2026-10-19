set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
# wait for ISIC grid to finish (up to ~30h)
for i in $(seq 1 3600); do [ -f "docs/optimization_loop/logs/plan24_isic_184200/DONE.marker" ] && break; sleep 30; done
echo "ISIC done, starting full-60 CIFAR $(date -Is)" > "docs/optimization_loop/logs/plan25_full60_200435/PROGRESS.txt"
run () { # name method cfg seed extra_env extra_args
  local NAME="$1" METHOD="$2" CFG="$3" SEED="$4" ENVX="$5"; shift 5
  local OUT="outputs/full60/${NAME}_seed${SEED}"
  echo "START ${NAME}_$SEED $(date -Is)" > "docs/optimization_loop/logs/plan25_full60_200435/${NAME}_${SEED}.log"; local T=$(date +%s)
  $ENVX timeout 36000 env CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method "$METHOD"     --cfg "$CFG" --seed "$SEED" --set epochs=60 "$@" --set output_dir="$OUT" >> "docs/optimization_loop/logs/plan25_full60_200435/${NAME}_${SEED}.log" 2>&1
  echo "EXIT ${NAME}_$SEED code=$? runtime_s=$(( $(date +%s)-T ))" >> "docs/optimization_loop/logs/plan25_full60_200435/${NAME}_${SEED}.log"
  echo "${NAME}_$SEED done" >> "docs/optimization_loop/logs/plan25_full60_200435/PROGRESS.txt"
}
for S in 20260714 20260715 20260716; do
  run splitfl   splitfl    configs/proxy/plan2_cifar100_splitfl.yaml   $S ""
  run federated federated  configs/proxy/plan2_cifar100_federated.yaml $S ""
  run hierfl    hierfl     configs/proxy/plan15_cifar100_hierfl.yaml    $S ""
  run hsfl      hsfl       configs/proxy/plan15_cifar100_hsfl.yaml      $S ""
  run heterosfl hetero-sfl configs/proxy/plan15_cifar100_heterosfl.yaml $S ""
  run hsfp      h-sfp      configs/proxy/plan2_cifar100_hsfp_baseline.yaml $S "HSFP_FREEZE_BACKBONE=1" --ablation baseline_hsfp --set prototype_space=centered_cosine --set num_workers=0
done
python3 tools/fair_compare/aggregate.py outputs/full60/* --output results/fair_comparison_cifar100_full60.csv >> "docs/optimization_loop/logs/plan25_full60_200435/aggregate.log" 2>&1
echo "ALL_DONE $(date -Is)" > "docs/optimization_loop/logs/plan25_full60_200435/DONE.marker"
