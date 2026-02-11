#!/bin/bash
# Launch GRPO math training: Qwen3-8B-Base on DFW (1 node, 8 GPUs)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

COMMAND="uv run python examples/run_grpo.py --config experiments/rl_baselines/math/qwen3_8b_base/config.yaml" \
CONTAINER=${CONTAINER:?Set CONTAINER env var} \
MOUNTS=${MOUNTS:-"$REPO_DIR:$REPO_DIR"} \
GPUS_PER_NODE=8 \
sbatch \
    --nodes=1 \
    --account=${ACCOUNT:?Set ACCOUNT} \
    --partition=${PARTITION:-batch_long} \
    --time=${TIME:-8:00:00} \
    --job-name=grpo-math-qwen3-8b-base \
    "$REPO_DIR/ray.sub"
