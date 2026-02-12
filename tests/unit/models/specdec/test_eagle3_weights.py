"""Unit tests for Eagle3 Megatron <-> HuggingFace weight conversion round-trips.

The first two tests (QKV and gate/up split-fuse) are pure torch and run anywhere.
The last two tests import from nemo_rl.models.specdec.llama_eagle3 which depends
on megatron-core, so they are skipped when that package is unavailable.
"""

import pytest
import torch

# Typical Qwen3-1.7B / Llama-style dimensions (GQA: num_heads != num_kv_heads)
HIDDEN_SIZE = 2048
NUM_ATTENTION_HEADS = 16
NUM_KEY_VALUE_HEADS = 8
KV_CHANNELS = HIDDEN_SIZE // NUM_ATTENTION_HEADS  # 128
FFN_HIDDEN_SIZE = 6144
VOCAB_SIZE = 151936

Q_DIM = NUM_ATTENTION_HEADS * KV_CHANNELS  # 2048
KV_DIM = NUM_KEY_VALUE_HEADS * KV_CHANNELS  # 1024
QKV_DIM = Q_DIM + 2 * KV_DIM  # 4096

try:
    from nemo_rl.models.specdec.llama_eagle3 import (
        load_hf_weights_to_eagle,
        save_eagle_weights_to_hf,
    )

    HAS_MEGATRON = True
except ImportError:
    HAS_MEGATRON = False

requires_megatron = pytest.mark.skipif(
    not HAS_MEGATRON, reason="megatron-core not available"
)


def _make_hf_state_dict() -> dict[str, torch.Tensor]:
    """Build a minimal HF Eagle3 state dict with deterministic random weights."""
    g = torch.Generator().manual_seed(42)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    return {
        "fc.weight": rand(HIDDEN_SIZE, HIDDEN_SIZE * 3),
        "norm.weight": rand(HIDDEN_SIZE),
        "lm_head.weight": rand(VOCAB_SIZE, HIDDEN_SIZE),
        "midlayer.input_layernorm.weight": rand(HIDDEN_SIZE),
        "midlayer.hidden_norm.weight": rand(HIDDEN_SIZE),
        "midlayer.post_attention_layernorm.weight": rand(HIDDEN_SIZE),
        "midlayer.self_attn.q_proj.weight": rand(Q_DIM, 2 * HIDDEN_SIZE),
        "midlayer.self_attn.k_proj.weight": rand(KV_DIM, 2 * HIDDEN_SIZE),
        "midlayer.self_attn.v_proj.weight": rand(KV_DIM, 2 * HIDDEN_SIZE),
        "midlayer.self_attn.o_proj.weight": rand(HIDDEN_SIZE, Q_DIM),
        "midlayer.mlp.gate_proj.weight": rand(FFN_HIDDEN_SIZE, HIDDEN_SIZE),
        "midlayer.mlp.up_proj.weight": rand(FFN_HIDDEN_SIZE, HIDDEN_SIZE),
        "midlayer.mlp.down_proj.weight": rand(HIDDEN_SIZE, FFN_HIDDEN_SIZE),
    }


class TestQKVSplitFuseRoundTrip:
    """Fuse Q+K+V into a single tensor, then split back. Verify exact match."""

    def test_round_trip(self):
        g = torch.Generator().manual_seed(0)
        q = torch.randn(Q_DIM, 2 * HIDDEN_SIZE, generator=g)
        k = torch.randn(KV_DIM, 2 * HIDDEN_SIZE, generator=g)
        v = torch.randn(KV_DIM, 2 * HIDDEN_SIZE, generator=g)

        # Fuse (as load_hf_weights_to_eagle does)
        fused = torch.cat([q, k, v], dim=0)
        assert fused.shape == (QKV_DIM, 2 * HIDDEN_SIZE)

        # Split (as save_eagle_weights_to_hf does)
        q2, k2, v2 = fused.split([Q_DIM, KV_DIM, KV_DIM], dim=0)

        assert torch.equal(q, q2)
        assert torch.equal(k, k2)
        assert torch.equal(v, v2)


class TestGateUpSplitFuseRoundTrip:
    """Fuse gate+up into a single tensor, then split back. Verify exact match."""

    def test_round_trip(self):
        g = torch.Generator().manual_seed(1)
        gate = torch.randn(FFN_HIDDEN_SIZE, HIDDEN_SIZE, generator=g)
        up = torch.randn(FFN_HIDDEN_SIZE, HIDDEN_SIZE, generator=g)

        # Fuse
        fused = torch.cat([gate, up], dim=0)
        assert fused.shape == (2 * FFN_HIDDEN_SIZE, HIDDEN_SIZE)

        # Split
        gate2, up2 = fused.split([FFN_HIDDEN_SIZE, FFN_HIDDEN_SIZE], dim=0)

        assert torch.equal(gate, gate2)
        assert torch.equal(up, up2)


EXPECTED_HF_KEYS = {
    "fc.weight",
    "norm.weight",
    "lm_head.weight",
    "midlayer.input_layernorm.weight",
    "midlayer.hidden_norm.weight",
    "midlayer.post_attention_layernorm.weight",
    "midlayer.self_attn.q_proj.weight",
    "midlayer.self_attn.k_proj.weight",
    "midlayer.self_attn.v_proj.weight",
    "midlayer.self_attn.o_proj.weight",
    "midlayer.mlp.gate_proj.weight",
    "midlayer.mlp.up_proj.weight",
    "midlayer.mlp.down_proj.weight",
}


class _Cfg:
    """Minimal config namespace matching the test dimensions."""

    num_attention_heads = NUM_ATTENTION_HEADS
    num_query_groups = NUM_KEY_VALUE_HEADS
    kv_channels = KV_CHANNELS
    ffn_hidden_size = FFN_HIDDEN_SIZE
    hidden_size = HIDDEN_SIZE


@requires_megatron
class TestWeightNameMapping:
    """Verify that save_eagle_weights_to_hf produces all expected HF keys."""

    def test_all_keys_present(self):
        g = torch.Generator().manual_seed(7)

        def rand(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=g)

        megatron_state = {
            "model.fc.weight": rand(HIDDEN_SIZE, HIDDEN_SIZE * 3),
            "model.norm.weight": rand(HIDDEN_SIZE),
            "lm_head.weight": rand(VOCAB_SIZE, HIDDEN_SIZE),
            "model.layer.input_layernorm.weight": rand(HIDDEN_SIZE),
            "model.layer.hidden_norm.weight": rand(HIDDEN_SIZE),
            "model.layer.pre_mlp_layernorm.weight": rand(HIDDEN_SIZE),
            "model.layer.self_attention.linear_qkv.weight": rand(
                QKV_DIM, 2 * HIDDEN_SIZE
            ),
            "model.layer.self_attention.linear_proj.weight": rand(HIDDEN_SIZE, Q_DIM),
            "model.layer.mlp.linear_fc1.weight": rand(2 * FFN_HIDDEN_SIZE, HIDDEN_SIZE),
            "model.layer.mlp.linear_fc2.weight": rand(HIDDEN_SIZE, FFN_HIDDEN_SIZE),
        }

        class _MockModel:
            def state_dict(self):
                return megatron_state

        class _MockWrapped:
            module = _MockModel()

        hf_state = save_eagle_weights_to_hf(_MockWrapped(), _Cfg())
        assert set(hf_state.keys()) == EXPECTED_HF_KEYS


@requires_megatron
class TestFullStateDictRoundTrip:
    """Load HF -> Megatron -> export back to HF, verify all tensors match."""

    @pytest.fixture
    def hf_state(self):
        return _make_hf_state_dict()

    def test_round_trip(self, hf_state):
        stored_state: dict[str, torch.Tensor] = {}

        class _MockModel:
            """Mimics Eagle3ForCausalLM.

            load_hf_weights_to_eagle calls model.load_state_dict() directly.
            save_eagle_weights_to_hf unwraps model.module before state_dict().
            """

            def load_state_dict(self, state_dict, strict=True):
                stored_state.update(state_dict)
                return [], []

            def state_dict(self):
                return dict(stored_state)

        class _MockWrapped:
            """Mimics Float16Module / DDP wrapper."""

            def __init__(self):
                self.module = _MockModel()

            def load_state_dict(self, state_dict, strict=True):
                stored_state.update(state_dict)
                return [], []

        mock_model = _MockWrapped()

        # Step 1: HF -> Megatron
        load_hf_weights_to_eagle(mock_model, hf_state, _Cfg())

        # Step 2: Megatron -> HF
        exported = save_eagle_weights_to_hf(mock_model, _Cfg())

        # Step 3: Verify all HF keys match (lm_head skipped on load)
        hf_keys_without_lm_head = {k for k in hf_state if k != "lm_head.weight"}
        exported_keys_without_lm_head = {k for k in exported if k != "lm_head.weight"}
        assert hf_keys_without_lm_head == exported_keys_without_lm_head

        for key in hf_keys_without_lm_head:
            assert torch.allclose(hf_state[key], exported[key], atol=0, rtol=0), (
                f"Mismatch at {key}: max diff = {(hf_state[key] - exported[key]).abs().max()}"
            )
