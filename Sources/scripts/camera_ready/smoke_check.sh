#!/usr/bin/env bash
# ============================================================================
# Pre-grid smoke check: run ONE short end-to-end run per method through
# main.py to confirm the bug-fixed code trains without crashing.
# Very small: 2 epochs, frac=0.05, ssl/local_ep=1 -> a couple minutes each.
# Writes per-method logs + a SMOKE_STATUS summary. Not for results, only sanity.
# ============================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUT="results/camera_ready/smoke_logs"
mkdir -p "$OUT"
STATUS="$OUT/SMOKE_STATUS.txt"
: > "$STATUS"

OVR=(--set dataset=cifar100 --set epochs=2 --set ssl_epochs_client=1
     --set ssl_epochs_edge=1 --set syn_epochs_cloud=1 --set t1=1 --set t2=2
     --set local_ep=1 --set frac=0.05)

declare -A CFG=(
  [h-sfp]="configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"
  [federated]="configs/classification/federated/cifar_fedavg_resnet50.yaml"
  [hierfl]="configs/classification/hierfl/cifar_hierfl_resnet50.yaml"
  [splitfl]="configs/classification/splitfl/cifar_splitfed_resnet50.yaml"
  [hsfl]="configs/classification/hsfl/cifar_our_resnet50_5_10.yaml"
)

# Order: our method + the two we rewrote/most-risky first.
for m in h-sfp hsfl federated hierfl splitfl; do
    log="$OUT/smoke_${m}.log"
    echo "[smoke] $(date +%H:%M:%S) START $m -> $log"
    if python main.py --task classification --method "$m" --cfg "${CFG[$m]}" \
        --seed 0 "${OVR[@]}" > "$log" 2>&1; then
        f1=$(grep -ioE "Test F1[^0-9]*[0-9.]+%?|best_f1[^0-9]*[0-9.]+" "$log" | tail -1)
        echo "PASS  $m   ${f1:-(no F1 line found)}" | tee -a "$STATUS"
    else
        echo "FAIL  $m   (see $log)  last line: $(tail -1 "$log")" | tee -a "$STATUS"
    fi
done

# Extra: validate the two_level_dirichlet partition path (used by the hetero grid
# and exercised by the HSFP/HSFL `_client_to_edge` branch) on our method + HSFL.
for m in h-sfp hsfl; do
    log="$OUT/smoke_${m}_hetero.log"
    echo "[smoke] $(date +%H:%M:%S) START ${m}-hetero -> $log"
    if python main.py --task classification --method "$m" \
        --cfg "${CFG[$m]}" --seed 0 "${OVR[@]}" \
        --set partition=two_level_dirichlet --set num_edges=5 \
        --set alpha_edge=0.1 --set alpha_client=0.1 --set iid=false \
        > "$log" 2>&1; then
        echo "PASS  ${m}-hetero" | tee -a "$STATUS"
    else
        echo "FAIL  ${m}-hetero  (see $log)  last line: $(tail -1 "$log")" | tee -a "$STATUS"
    fi
done

echo "[smoke] ALL_SMOKE_DONE $(date)" | tee -a "$STATUS"
