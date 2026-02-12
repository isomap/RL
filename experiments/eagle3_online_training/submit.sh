#!/bin/bash
# Eagle3 online training submission script for DFW cluster
set -euo pipefail

CONTAINER=/lustre/fsw/portfolios/coreai/users/hiso/images/nvcr_nemo-rl_v0.5.0.sqsh
REPO_ROOT=$(git rev-parse --show-toplevel)

COMMAND="uv run ./examples/run_grpo.py --config examples/configs/grpo_math_qwen3_1.7b_eagle3.yaml" \
CONTAINER=$CONTAINER \
MOUNTS="$REPO_ROOT:$REPO_ROOT,/lustre:/lustre" \
sbatch \
    --nodes=1 \
    --account=coreai_horizon_dilations \
    --job-name=eagle3-online-training \
    --partition=batch \
    --time=4:00:00 \
    --exclusive \
    --mem=0 \
    ray.sub