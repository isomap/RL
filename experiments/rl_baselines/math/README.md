# Math Baselines

GRPO training on math data across 9 model configurations.

| Model | Arch | Seq Len | Cluster | Nodes |
|-------|------|---------|---------|-------|
| Qwen3-8B-Base | Dense | 32K | DFW 1n8g | 1 |
| Qwen3-8B | Dense | 40K | DFW 1n8g | 1 |
| Qwen3-30B-A3B-Base | MoE | 40K | DFW 4n8g | 4 |
| Qwen3-30B-A3B | MoE | 40K | DFW 4n8g | 4 |
| Qwen3-32B-Base | Dense | 32K | HSG 8n4g | 8 |
| Qwen3-32B | Dense | 40K | HSG 8n4g | 8 |
| GPT-OSS 20B | MoE | TBD | DFW 8n8g | 8 |
| GPT-OSS 120B | MoE | TBD | HSG 32n4g | 32 |
| Nemotron Nano | MoE | 40K | DFW 4n8g | 4 |

Data: `nvidia/Nemotron-Post-Training-Dataset-v1` math split. Environment: `math` with `hf_math_verify`.