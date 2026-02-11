#!/bin/bash
# Launch GRPO multitask training: GPT-OSS 120B on HSG (32 nodes, 4 GPUs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

COMMAND="uv run python examples/run_grpo.py --config experiments/rl_baselines/multitask/gpt_oss_120b/config.yaml" \
CONTAINER=${CONTAINER:?Set CONTAINER env var} \
MOUNTS=${MOUNTS:-"$REPO_DIR:$REPO_DIR"} \
GPUS_PER_NODE=4 \
sbatch \
    --nodes=32 \
    --account=${ACCOUNT:?Set ACCOUNT} \
    --partition=${PARTITION:-batch_long} \
    --time=${TIME:-7-00:00:00} \
    --job-name=grpo-multitask-gptoss-120b \
    "$REPO_DIR/ray.sub"
