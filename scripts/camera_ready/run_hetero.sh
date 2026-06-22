#!/usr/bin/env bash
# ============================================================================
# Experiment 1: Hierarchical heterogeneity (two-level Dirichlet).
# Sweeps (alpha_edge, alpha_client) in {1.0,0.1}x{1.0,0.1} for each method/seed.
# Addresses reviewer concern: hierarchical heterogeneity (inter- vs intra-edge).
#
# Smoke:  SMOKE=1 ./scripts/camera_ready/run_hetero.sh
# Full :  ./scripts/camera_ready/run_hetero.sh
# Subset: METHODS="h-sfp federated" ./scripts/camera_ready/run_hetero.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

METHODS="${METHODS:-h-sfp hsfl splitfl federated hierfl}"
DATASET="${DATASET:-cifar100}"
NUM_EDGES="${NUM_EDGES:-5}"
# (alpha_edge, alpha_client) pairs
SETTINGS=("1.0 1.0" "1.0 0.1" "0.1 1.0" "0.1 0.1")

[[ "$SMOKE" == "1" ]] && NUM_EDGES=2

for s in "${SETTINGS[@]}"; do
    read -r AE AC <<< "$s"
    for method in $METHODS; do
        for seed in $SEEDS; do
            exp_id="hetero_${method}_ae${AE}_ac${AC}_s${seed}"
            run_main "$exp_id" "$method" "$seed" \
                --set "dataset=${DATASET}" \
                --set "partition=two_level_dirichlet" \
                --set "num_edges=${NUM_EDGES}" \
                --set "alpha_edge=${AE}" \
                --set "alpha_client=${AC}" \
                --set "iid=false"
        done
    done
done

log_info "Hetero experiment dispatch complete."
log_info "Aggregate: python scripts/camera_ready/collect_camera_ready.py"
log_info "Tables   : python scripts/camera_ready/make_tables.py"
