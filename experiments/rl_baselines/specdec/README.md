# Speculative Decoding Comparisons (Phase 2)

Planned: for each baseline model in math/, code/, and multitask/, create a matching config with speculative decoding enabled during GRPO rollout generation.

## Comparisons

For each model + data combination:

| Metric | Baseline | + Spec Dec | Delta |
|--------|----------|------------|-------|
| Throughput (tokens/sec) | - | - | - |
| Wall-clock time per step | - | - | - |
| Final eval (AIME/MATH/etc.) | - | - | - |
| Training loss curve | - | - | - |

## Ablations

- With/without online eagle head training
- Draft model size vs speedup tradeoff
- Effect on RL training dynamics (reward hacking, mode collapse)

## Blocked on

- Speculative decoding feature landing in `specdec-online-training-dev` branch
- Draft model selection for each target model
