#!/usr/bin/env bash
# ============================================================================
# Camera-ready training grids on a constrained single-GPU budget.
# Runs (sequentially, with resume markers): hierarchical heterogeneity,
# partial participation, runtime profiling -> then collect + LaTeX tables.
#
# Defaults encode the "Fast" submission profile; override via env, e.g.:
#   SEEDS="0 1 2" METHODS="h-sfp federated splitfl hsfl hierfl" \
#   EXTRA_SET="epochs=60 ssl_epochs_client=5 ssl_epochs_edge=5 syn_epochs_cloud=5 t1=5 t2=10 local_ep=5" \
#   ./scripts/camera_ready/run_submission_grids.sh
#
# Re-running skips completed runs (results/camera_ready/.markers/); RESET=1 forces.
# ============================================================================
set -uo pipefail   # NOT -e: a single failed run must not abort the batch
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEEDS="${SEEDS:-0}"
export METHODS="${METHODS:-h-sfp federated splitfl}"
export EXTRA_SET="${EXTRA_SET:-epochs=30 ssl_epochs_client=3 ssl_epochs_edge=3 syn_epochs_cloud=3 t1=3 t2=6 local_ep=3}"

echo "[submission] $(date) SEEDS='$SEEDS' METHODS='$METHODS'"
echo "[submission] EXTRA_SET='$EXTRA_SET'"

echo "[submission] === hierarchical heterogeneity ==="
bash "$HERE/run_hetero.sh"

echo "[submission] === partial participation ==="
bash "$HERE/run_partial.sh"

echo "[submission] === runtime profiling ==="
bash "$HERE/run_profiling.sh"

echo "[submission] === aggregate + tables ==="
python "$HERE/collect_camera_ready.py" || true
python "$HERE/make_tables.py" || true

echo "[submission] ALL_GRIDS_DONE $(date)"
