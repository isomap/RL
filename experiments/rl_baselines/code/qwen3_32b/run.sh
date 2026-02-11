#!/bin/bash
# Launch GRPO code training: Qwen3-32B on HSG (8 nodes, 4 GPUs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

COMMAND="uv run python examples/run_grpo.py --config experiments/rl_baselines/code/qwen3_32b/config.yaml" \
CONTAINER=${CONTAINER:?Set CONTAINER env var} \
MOUNTS=${MOUNTS:-"$REPO_DIR:$REPO_DIR"} \
GPUS_PER_NODE=4 \
sbatch \
    --nodes=8 \
    --account=${ACCOUNT:?Set ACCOUNT} \
    --partition=${PARTITION:-batch_long} \
    --time=${TIME:-7-00:00:00} \
    --job-name=grpo-code-qwen3-32b \
    "$REPO_DIR/ray.sub"
