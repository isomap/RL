#!/bin/bash
# Launch GRPO multitask training: GPT-OSS 20B on DFW (8 nodes, 8 GPUs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

COMMAND="uv run python examples/run_grpo.py --config experiments/rl_baselines/multitask/gpt_oss_20b/config.yaml" \
CONTAINER=${CONTAINER:?Set CONTAINER env var} \
MOUNTS=${MOUNTS:-"$REPO_DIR:$REPO_DIR"} \
GPUS_PER_NODE=8 \
sbatch \
    --nodes=8 \
    --account=${ACCOUNT:?Set ACCOUNT} \
    --partition=${PARTITION:-batch_large_long} \
    --time=${TIME:-8:00:00} \
    --job-name=grpo-multitask-gptoss-20b \
    "$REPO_DIR/ray.sub"
