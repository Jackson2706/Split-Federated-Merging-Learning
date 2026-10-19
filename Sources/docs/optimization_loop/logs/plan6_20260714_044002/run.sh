set -o pipefail
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
CKPT_SSL="outputs/proxy/plan2_cifar100_hsfp_baseline_seed20260714/e4b67d7d5c12/817f3265d5bec70c72df50cb009c9de5770c8cc660e00418c49d30ca77a7555b/checkpoint.pt"

echo "===== [1/3] supervised proxy START $(date -Is) =====" | tee "$RUNDIR/supervised.log"
T0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp \
  --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 \
  --set epochs=10 --set client_objective=supervised --set client_objective_weight=0.1 \
  --set proxy_id=plan6_cifar100_hsfp_supervised_seed20260714 \
  --set output_dir=outputs/proxy/plan6_cifar100_hsfp_supervised_seed20260714 \
  >> "$RUNDIR/supervised.log" 2>&1
echo "===== supervised EXIT code=$? runtime_s=$(( $(date +%s)-T0 )) $(date -Is) =====" | tee -a "$RUNDIR/supervised.log"

SUP_CKPT=$(find outputs/proxy/plan6_cifar100_hsfp_supervised_seed20260714 -name checkpoint.pt | head -1)
echo "SUP_CKPT=$SUP_CKPT" | tee "$RUNDIR/paths.txt"

echo "===== [2/3] supervised linear-probe $(date -Is) =====" | tee "$RUNDIR/probe_supervised.log"
CUDA_VISIBLE_DEVICES=0 python tools/diag/encoder_linear_probe.py \
  --config configs/proxy/plan2_cifar100_hsfp_baseline.yaml --checkpoint "$SUP_CKPT" \
  --output-dir outputs/diag/plan6_supervised_encoder_probe --seed 20260714 --device cuda --linear-epochs 10 \
  >> "$RUNDIR/probe_supervised.log" 2>&1
echo "probe_supervised EXIT=$?" | tee -a "$RUNDIR/probe_supervised.log"

echo "===== [3/3] ssl-control linear-probe (PLAN-2 ckpt) $(date -Is) =====" | tee "$RUNDIR/probe_ssl.log"
CUDA_VISIBLE_DEVICES=0 python tools/diag/encoder_linear_probe.py \
  --config configs/proxy/plan2_cifar100_hsfp_baseline.yaml --checkpoint "$CKPT_SSL" \
  --output-dir outputs/diag/plan6_ssl_encoder_probe --seed 20260714 --device cuda --linear-epochs 10 \
  >> "$RUNDIR/probe_ssl.log" 2>&1
echo "probe_ssl EXIT=$?" | tee -a "$RUNDIR/probe_ssl.log"

echo "ALL_DONE $(date -Is)" | tee "$RUNDIR/DONE.marker"
