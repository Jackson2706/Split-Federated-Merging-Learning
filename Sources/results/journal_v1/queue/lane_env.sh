# Environment shared by every campaign lane (sourced by start_lanes.sh and re-sourced between runs).
# New machine (2026-10-15): 1x RTX 6000 Ada 48 GB, 128 cores, 503 GB RAM, 32 GB container disk (no volume).
# Previous machine's values: /workspace/campaign/lane_env.sh.oldmachine
export OUT_ROOT=/workspace/campaign/outputs_journal_v1
export ROUNDS=60
export MIN_FREE_GB=3            # disk is 32 GB total; runs keep checkpoints (~100 MB each)
export MIN_REPO_FREE_GB=1
export GPU_ID=0
export MIN_GPU_FREE_MB=4000
export PYTHONUNBUFFERED=1
export PATH=/workspace/venvs/sfml/bin:$PATH
export DIAG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/Split-Federated-Merging-Learning${PYTHONPATH:+:$PYTHONPATH}   # code imports Sources.*
# CPU thread caps per run (set from the concurrency pilot; avoids 128-way oversubscription per process)
export OMP_NUM_THREADS=${LANE_THREADS:-6}
export MKL_NUM_THREADS=${LANE_THREADS:-6}
export MAX_TRANSIENT_RETRIES=2
