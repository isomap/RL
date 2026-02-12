## Eagle3 Online Training

GRPO + Eagle3 speculative decoding joint training. Policy model (Qwen3-1.7B) is frozen for the Eagle loss; the Eagle3 draft head is trained with forward KL against the policy's next-token distribution.

Config: `examples/configs/grpo_math_qwen3_1.7b_eagle3.yaml`

### Run

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/ghq/github.com/isomap/RL
bash experiments/eagle3_online_training/submit.sh
```