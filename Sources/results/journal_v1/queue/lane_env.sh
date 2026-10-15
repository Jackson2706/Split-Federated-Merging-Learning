# Environment shared by every campaign lane (sourced by start_lanes.sh)
export OUT_ROOT=/media/jackson/Data/ehsfp_campaign/outputs_journal_v1
export ROUNDS=60
export MIN_FREE_GB=20
export MIN_REPO_FREE_GB=2
export GPU_ID=0
export PYTHONUNBUFFERED=1
export PATH=/home/jackson/miniconda3/envs/AAAI-2026/bin:$PATH
export DIAG=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/jackson/Desktop/Split-Federated-Merging-Learning${PYTHONPATH:+:$PYTHONPATH}   # code imports Sources.* since the 2026-10-02 restructure
