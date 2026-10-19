set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
# wait for H-SFP full-60 (plan27) to finish
for i in $(seq 1 3600); do [ -f "docs/optimization_loop/logs/plan27_hsfp60_152010/DONE.marker" ] && break; sleep 30; done
echo "plan27 done, ISIC verify $(date -Is)" > "docs/optimization_loop/logs/plan28_isicre_160903/PROGRESS.txt"
# verify gate: federated ISIC 3 rounds, require IoU>0 (meaningful metric)
timeout 2400 env CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --method federated   --cfg configs/proxy/plan_isic2018_federated.yaml --seed 20260714 --set epochs=3   --set output_dir=outputs/isic2018_verify2/federated > "docs/optimization_loop/logs/plan28_isicre_160903/verify.log" 2>&1
IOU=$(grep -hoiE "IoU[=: ]+[0-9.]+" "docs/optimization_loop/logs/plan28_isicre_160903/verify.log" | grep -oE "[0-9.]+" | tail -1)
NAN=$(grep -ciE "NaN/Inf" "docs/optimization_loop/logs/plan28_isicre_160903/verify.log")
echo "verify IoU=${IOU:-none} nan=$NAN" >> "docs/optimization_loop/logs/plan28_isicre_160903/PROGRESS.txt"
GO=$(python3 -c "print(1 if (${IOU:-0}>0.01 and $NAN==0) else 0)" 2>/dev/null)
if [ "$GO" = "1" ]; then
  for M in federated hierfl hetero-sfl h-sfp; do
    CFG=$M; [ "$M" = "hetero-sfl" ] && CFG=heterosfl; [ "$M" = "h-sfp" ] && CFG=hsfp
    for S in 20260714 20260715 20260716; do
      EX=""; [ "$M" = "h-sfp" ] && EX="--ablation baseline_hsfp"
      echo "START isic_${CFG}_$S $(date -Is)" > "docs/optimization_loop/logs/plan28_isicre_160903/isic_${CFG}_${S}.log"; T=$(date +%s)
      timeout 9000 env CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --method $M         --cfg configs/proxy/plan_isic2018_${CFG}.yaml --seed $S $EX --set output_dir="outputs/isic2018/${CFG}_seed${S}" >> "docs/optimization_loop/logs/plan28_isicre_160903/isic_${CFG}_${S}.log" 2>&1
      echo "EXIT isic_${CFG}_$S code=$? runtime_s=$(( $(date +%s)-T ))" >> "docs/optimization_loop/logs/plan28_isicre_160903/isic_${CFG}_${S}.log"; echo "isic_${CFG}_$S done" >> "docs/optimization_loop/logs/plan28_isicre_160903/PROGRESS.txt"
    done
  done
  python3 tools/fair_compare/aggregate.py --dataset isic2018 outputs/isic2018/* >> "docs/optimization_loop/logs/plan28_isicre_160903/aggregate.log" 2>&1
else echo "ISIC VERIFY FAILED (IoU=${IOU:-none} nan=$NAN) - grid NOT launched" >> "docs/optimization_loop/logs/plan28_isicre_160903/PROGRESS.txt"; fi
echo "ALL_DONE $(date -Is)" > "docs/optimization_loop/logs/plan28_isicre_160903/DONE.marker"
