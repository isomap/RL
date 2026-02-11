# RL Baselines

Baseline RL training results across models, data settings, and scales. These baselines will be compared against speculative decoding variants.

## Model Matrix

| ID | Model | Variant | HF Path | Arch | Params (Active) | Cluster | Nodes | GPUs/Node |
|----|-------|---------|---------|------|-----------------|---------|-------|-----------|
| M1 | Qwen3-8B | Base | `Qwen/Qwen3-8B-Base` | Dense | 8B | DFW | 1 | 8 |
| M2 | Qwen3-8B | Post-trained | `Qwen/Qwen3-8B` | Dense | 8B | DFW | 1 | 8 |
| M3 | Qwen3-30B-A3B | Base | `Qwen/Qwen3-30B-A3B-Base` | MoE | 30B (3B) | DFW | 4 | 8 |
| M4 | Qwen3-30B-A3B | Post-trained | `Qwen/Qwen3-30B-A3B` | MoE | 30B (3B) | DFW | 4 | 8 |
| M5 | Qwen3-32B | Base | `Qwen/Qwen3-32B-Base` | Dense | 32B | HSG | 8 | 4 |
| M6 | Qwen3-32B | Post-trained | `Qwen/Qwen3-32B` | Dense | 32B | HSG | 8 | 4 |
| M7 | GPT-OSS 20B | - | TBD (internal) | MoE | 20B | DFW | 8 | 8 |
| M8 | GPT-OSS 120B | - | TBD (internal) | MoE | 120B | HSG | 32 | 4 |
| M9 | Nemotron Nano | Post-trained | TBD (NVIDIA) | MoE | 30B (3B) | DFW | 4 | 8 |

## Data Settings

| Setting | Source | Environment | Reward |
|---------|--------|-------------|--------|
| Math | `nvidia/Nemotron-Post-Training-Dataset-v1` (math split) | `math` (hf_math_verify) | Exact match (0/1) |
| Code | `nvidia/Nemotron-Post-Training-Dataset-v1` (code split) | `code` (sandboxed exec) | Pass/fail on tests |
| Multi-task | NeMo Gym blend | NeMo Gym (math, code, IF, MCQA, structured) | Per-task rewards |

## Training

GRPO algorithm, Megatron backend, vLLM generation. Sequence lengths set to each model's native `max_position_embeddings` (32K-40K).

## Evaluations

### Math

| Benchmark | Dataset | Size | Metric |
|-----------|---------|------|--------|
| AIME 2024 | `aime2024` | 30 | pass@1 (avg 8) |
| AIME 2025 | `aime2025` | 30 | pass@1 (avg 8) |
| MATH-500 | `math500` | 500 | pass@1 |
| MATH (full) | `math` | 12,500 | pass@1 |
| GSM8K | `gsm8k` | 1,000 | pass@1 |

### Code (post-hoc)

| Benchmark | Size | Metric |
|-----------|------|--------|
| LiveCodeBench | ~1,055 | pass@1 |
| HumanEval | 164 | pass@1 |
| MBPP | 427 | pass@1 |

### General

| Benchmark | Dataset | Metric |
|-----------|---------|--------|
| MMLU | `mmlu` | accuracy |
| MMLU-Pro | `mmlu_pro` | accuracy |
| GPQA | `gpqa` | accuracy |
| IFEval | TBD | strict accuracy |

## Status

| Model | Math | Code | Multi-task |
|-------|------|------|------------|
| Qwen3-8B-Base | - | - | - |
| Qwen3-8B | - | - | - |
| Qwen3-30B-A3B-Base | - | - | - |
| Qwen3-30B-A3B | - | - | - |
| Qwen3-32B-Base | - | - | - |
| Qwen3-32B | - | - | - |
| GPT-OSS 20B | - | - | - |
| GPT-OSS 120B | - | - | - |
| Nemotron Nano | - | - | - |

## Directory Structure

```
shared/           # Shared data prep, eval configs, SLURM template
math/             # GRPO on math data (9 models)
code/             # GRPO on code data (9 models)
multitask/        # GRPO on NeMo Gym multi-task (9 models)
specdec/          # Phase 2: speculative decoding comparisons
```