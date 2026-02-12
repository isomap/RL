#!/bin/bash
# Launch GRPO multitask training: Qwen3-8B on DFW (1 node, 8 GPUs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

COMMAND="uv run python examples/run_grpo.py --config experiments/rl_baselines/multitask/qwen3_8b/config.yaml" \
CONTAINER=${CONTAINER:?Set CONTAINER env var} \
MOUNTS=${MOUNTS:-"$REPO_DIR:$REPO_DIR"} \
GPUS_PER_NODE=8 \
sbatch \
    --nodes=4 \
    --account=${ACCOUNT:?Set ACCOUNT} \
    --partition=${PARTITION:-batch_long} \
    --time=${TIME:-8:00:00} \
    --job-name=grpo-multitask-qwen3-8b \
    "$REPO_DIR/ray.sub"
