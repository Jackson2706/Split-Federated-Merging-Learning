#!/usr/bin/env bash
# ============================================================================
# Experiment 2: Partial participation.
# Active-client ratio (frac) in {0.1, 0.5, 1.0} crossed with data distribution
# in {IID, Dirichlet a=0.05, Dirichlet a=0.1}. 200 clients (full).
# Headline settings requested: IID@10%, IID@50%, Dir(0.05), Dir(0.1).
# Addresses reviewer concern: partial participation.
#
# Smoke:  SMOKE=1 ./scripts/camera_ready/run_partial.sh
# Full :  ./scripts/camera_ready/run_partial.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

METHODS="${METHODS:-h-sfp federated hierfl splitfl hsfl}"
DATASET="${DATASET:-cifar100}"
FRACS="${FRACS:-0.1 0.5 1.0}"
# distribution specs: "iid" | "dir:<alpha>"
DISTS="${DISTS:-iid dir:0.05 dir:0.1}"

for dist in $DISTS; do
    for frac in $FRACS; do
        for method in $METHODS; do
            for seed in $SEEDS; do
                if [[ "$dist" == "iid" ]]; then
                    dist_tag="iid"
                    dist_args=(--set "iid=true")
                else
                    alpha="${dist#dir:}"
                    dist_tag="dir${alpha}"
                    dist_args=(--set "iid=false" --set "partition=dirichlet" --set "dirichlet_alpha=${alpha}")
                fi
                exp_id="partial_${method}_frac${frac}_${dist_tag}_s${seed}"
                run_main "$exp_id" "$method" "$seed" \
                    --set "dataset=${DATASET}" \
                    --set "frac=${frac}" \
                    "${dist_args[@]}"
            done
        done
    done
done

log_info "Partial-participation dispatch complete."
log_info "Aggregate: python scripts/camera_ready/collect_camera_ready.py"
