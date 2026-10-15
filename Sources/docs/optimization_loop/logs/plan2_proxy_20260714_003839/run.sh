set -o pipefail
cd /home/jackson/Desktop/Split-Federated-Merging-Learning
RUNDIR="$1"
run_one () {
  local NAME="$1"; shift
  echo "===== START $NAME $(date -Is) =====" | tee "$RUNDIR/${NAME}.log"
  local T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 "$@" >> "$RUNDIR/${NAME}.log" 2>&1
  local CODE=$?
  local T1=$(date +%s)
  echo "===== EXIT $NAME code=$CODE runtime_s=$((T1-T0)) $(date -Is) =====" | tee -a "$RUNDIR/${NAME}.log"
  echo "$NAME code=$CODE runtime_s=$((T1-T0))" >> "$RUNDIR/SUMMARY.txt"
}
run_one splitfl   python main.py --task classification --method splitfl   --cfg configs/proxy/plan2_cifar100_splitfl.yaml   --seed 20260714
run_one federated python main.py --task classification --method federated --cfg configs/proxy/plan2_cifar100_federated.yaml --seed 20260714
run_one hsfp_baseline python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714
run_one hsfp_full_e   python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_full_e.yaml   --ablation full_e_hsfp   --seed 20260714
echo "ALL_DONE $(date -Is)" | tee "$RUNDIR/DONE.marker"
