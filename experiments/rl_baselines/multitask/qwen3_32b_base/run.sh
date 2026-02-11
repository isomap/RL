#!/bin/bash
# Launch GRPO multitask training: Qwen3-32B-Base on HSG (8 nodes, 4 GPUs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

COMMAND="uv run python examples/run_grpo.py --config experiments/rl_baselines/multitask/qwen3_32b_base/config.yaml" \
CONTAINER=${CONTAINER:?Set CONTAINER env var} \
MOUNTS=${MOUNTS:-"$REPO_DIR:$REPO_DIR"} \
GPUS_PER_NODE=4 \
sbatch \
    --nodes=8 \
    --account=${ACCOUNT:?Set ACCOUNT} \
    --partition=${PARTITION:-batch_long} \
    --time=${TIME:-7-00:00:00} \
    --job-name=grpo-multitask-qwen3-32b-base \
    "$REPO_DIR/ray.sub"
