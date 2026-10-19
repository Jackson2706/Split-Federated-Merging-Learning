set +e
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="docs/optimization_loop/logs/plan30_isicfixed_143359"
# verify gate: federated ISIC, 3 rounds (the one that collapsed before) - require IoU>0, no NaN, no shape errors
echo "START verify $(date -Is)" > "$RUNDIR/verify.log"; T=$(date +%s)
timeout 1800 env CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --method federated   --cfg configs/proxy/plan_isic2018_federated.yaml --seed 20260714 --set epochs=3   --set output_dir=outputs/isic2018_verify3/federated >> "$RUNDIR/verify.log" 2>&1
VC=$?
IOU=$(grep -hoiE "IoU=[0-9.]+" "$RUNDIR/verify.log" | tail -1 | grep -oE "[0-9.]+")
NAN=$(grep -ciE "NaN/Inf|shape mismatch" "$RUNDIR/verify.log")
echo "EXIT verify code=$VC IoU=${IOU:-none} nan_or_shape_err=$NAN" >> "$RUNDIR/verify.log"
echo "verify code=$VC IoU=${IOU:-none} nan=$NAN" >> "$RUNDIR/PROGRESS.txt"
GO=$(python3 -c "print(1 if ($VC==0 and ${IOU:-0}>1.0 and $NAN==0) else 0)" 2>/dev/null)
if [ "$GO" = "1" ]; then
  for M in federated hierfl hetero-sfl h-sfp; do
    CFG=$M; [ "$M" = "hetero-sfl" ] && CFG=heterosfl; [ "$M" = "h-sfp" ] && CFG=hsfp
    for S in 20260714 20260715 20260716; do
      EX=""; [ "$M" = "h-sfp" ] && EX="--ablation baseline_hsfp"
      echo "START isic_${CFG}_$S $(date -Is)" > "$RUNDIR/isic_${CFG}_${S}.log"; T=$(date +%s)
      timeout 3000 env CUDA_VISIBLE_DEVICES=0 python main.py --task segmentation --method $M         --cfg configs/proxy/plan_isic2018_${CFG}.yaml --seed $S $EX --set output_dir="outputs/isic2018/${CFG}_seed${S}_v2" >> "$RUNDIR/isic_${CFG}_${S}.log" 2>&1
      echo "EXIT isic_${CFG}_$S code=$? runtime_s=$(( $(date +%s)-T ))" >> "$RUNDIR/isic_${CFG}_${S}.log"; echo "isic_${CFG}_$S done" >> "$RUNDIR/PROGRESS.txt"
    done
  done
  python3 tools/fair_compare/aggregate.py --dataset isic2018 outputs/isic2018/*_v2 --output results/fair_comparison_isic2018.csv >> "$RUNDIR/aggregate.log" 2>&1
else
  echo "VERIFY GATE FAILED (code=$VC IoU=${IOU:-none} nan=$NAN) - grid NOT launched" >> "$RUNDIR/PROGRESS.txt"
fi
echo "ALL_DONE $(date -Is)" > "$RUNDIR/DONE.marker"
