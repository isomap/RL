# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Eagle3 speculative decoding model — v2 (reuses standard Megatron modules).

Single-layer Eagle head that reuses the target model's LM head.

Key design choices:
  - ``EagleSelfAttention`` is a thin subclass of the standard ``SelfAttention``
    that only overrides the ``linear_qkv`` input dimension (``2*h``).
  - ``EagleLayer`` subclasses ``TransformerLayer`` to inject the
    embed-concatenation logic in ``_forward_attention``.
  - Gets ``TEDotProductAttention`` (flash attention) through the standard TE
    spec, fixing the ``scaled_masked_softmax_cuda`` import error.
  - The LM head is **not** owned by Eagle — it is borrowed from the target
    model at forward time.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, Optional, Tuple, Union

import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from torch import Tensor

try:
    from megatron.core.extensions.transformer_engine import TENorm  # noqa: F401

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


# =============================================================================
# EagleSelfAttention — only overrides linear_qkv input size
# =============================================================================


class EagleSelfAttention(SelfAttention):
    """``SelfAttention`` with a configurable QKV input dimension.

    The **only** change relative to the parent class is that the first
    positional argument to ``build_module(submodules.linear_qkv, ...)`` is
    ``qkv_input_size`` instead of ``config.hidden_size``.  Everything else —
    QKV split, rotary embeddings, core attention, output projection — is
    inherited as-is.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        qkv_input_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            **kwargs,
        )

        # Re-create linear_qkv with custom input size when needed.
        if qkv_input_size is not None and qkv_input_size != config.hidden_size:
            self.linear_qkv = build_module(
                submodules.linear_qkv,
                qkv_input_size,
                self.linear_qkv_out_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="qkv",
                tp_group=self.pg_collection.tp,
            )


# =============================================================================
# EagleLayer — TransformerLayer with embed concatenation
# =============================================================================


class EagleLayer(TransformerLayer):
    """Single Eagle decoder layer.

    Before self-attention the layer normalises both the token embeddings and
    the auxiliary hidden states, then concatenates them along the last
    dimension to produce a ``[s, b, 2*h]`` input for the QKV projection.

    All other behaviour (MLP, residual connections, BDA, …) is inherited
    from ``TransformerLayer``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            **kwargs,
        )
        # Extra norm for the auxiliary hidden states branch.
        self.hidden_norm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    # ------------------------------------------------------------------
    # Override _forward_attention to inject the concatenation logic.
    # ------------------------------------------------------------------
    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        # Eagle passes embeds through this kwarg:
        embeds: Optional[Tensor] = None,
        **kwargs,
    ):
        """Pre-norm → concat embeds and hidden → self-attn → BDA → residual."""
        assert embeds is not None, "EagleLayer requires `embeds` input"

        residual = hidden_states

        # Norm each branch independently.
        embeds_norm = self.input_layernorm(embeds)
        hidden_norm = self.hidden_norm(hidden_states)

        # Concat: [s, b, h] + [s, b, h] → [s, b, 2*h]
        attn_input = torch.cat([embeds_norm, hidden_norm], dim=-1)

        # Self attention  (returns (output, bias) tuple).
        attention_output_with_bias = self.self_attention(
            attn_input,
            attention_mask=attention_mask,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ("context", "context_mask", "embeds")
            },
        )

        # Bias-dropout-add with residual.
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        return hidden_states, None  # context=None (no cross-attn)


# =============================================================================
# Helpers to build the TE / local layer spec
# =============================================================================


def _get_base_layer_spec(config: TransformerConfig) -> ModuleSpec:
    """Return a standard GPT TransformerLayer spec (TE when available)."""
    if HAVE_TE:
        return get_gpt_layer_with_transformer_engine_spec(
            qk_layernorm=getattr(config, "qk_layernorm", False),
        )
    else:
        return get_gpt_layer_local_spec(
            qk_layernorm=getattr(config, "qk_layernorm", False),
            normalization=config.normalization,
        )


def _make_eagle_layer_spec(
    base_spec: ModuleSpec,
    qkv_input_size: int,
) -> ModuleSpec:
    """Derive an Eagle layer spec from a standard GPT *base_spec*.

    Changes:
      * ``module`` → ``EagleLayer``
      * ``self_attention.module`` → ``EagleSelfAttention``
      * Adds ``qkv_input_size`` as a param to the self-attention spec.
    """
    spec = copy.deepcopy(base_spec)
    spec.module = EagleLayer

    attn_spec = spec.submodules.self_attention
    attn_spec.module = EagleSelfAttention
    if attn_spec.params is None:
        attn_spec.params = {}
    attn_spec.params["qkv_input_size"] = qkv_input_size

    return spec


# =============================================================================
# EagleModel
# =============================================================================


class EagleModel(MegatronModule):
    """Single-layer Eagle model reusing standard Megatron modules.

    Architecture:
      1. FC linear ``3*h → h`` to reduce auxiliary hidden states.
      2. ``EagleLayer`` (embeds + hidden concat → 2*h QKV → attn → MLP).
      3. Final RMS norm.

    The LM head is **not** included — it is borrowed from the target model.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config = config
        self.hidden_size = config.hidden_size

        # FC projection: 3*h → h  (gather output so every rank sees full h)
        self.fc = ColumnParallelLinear(
            self.hidden_size * 3,
            self.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=True,
            skip_bias_add=True,
        )

        # Rotary embeddings.
        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=getattr(config, "rotary_percent", 1.0),
            rotary_interleaved=getattr(config, "rotary_interleaved", False),
            seq_len_interpolation_factor=getattr(
                config, "seq_len_interpolation_factor", None
            ),
            rotary_base=getattr(config, "rotary_base", 10000),
        )

        # Single Eagle decoder layer.
        base_spec = _get_base_layer_spec(config)
        eagle_spec = _make_eagle_layer_spec(
            base_spec, qkv_input_size=2 * self.hidden_size
        )
        self.layer = build_module(eagle_spec, config=config, layer_number=1)

        # Final norm.
        if HAVE_TE:
            self.norm = TENorm(
                config=config,
                hidden_size=self.hidden_size,
                eps=config.layernorm_epsilon,
            )
        else:
            from megatron.core.transformer.torch_norm import WrappedTorchNorm

            self.norm = WrappedTorchNorm(
                config=config,
                hidden_size=self.hidden_size,
                eps=config.layernorm_epsilon,
            )

    def forward(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            hidden_states: Auxiliary hidden states ``[s, b, 3*h]``.
            input_embeds: Token embeddings ``[s, b, h]``.
            attention_mask: Optional attention mask.

        Returns:
            Normed output ``[s, b, h]``.
        """
        # FC projection: [s, b, 3*h] → [s, b, h]
        hidden_states, _ = self.fc(hidden_states)

        # Rotary embeddings (tuple of (q_emb, k_emb))
        seq_length = hidden_states.shape[0]
        seq_length *= self.config.context_parallel_size
        rotary_pos_emb = self.rotary_pos_emb(seq_length)
        rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)

        # Scatter back to SP so it matches input_embeds [s/tp, b, H]
        if self.config.sequence_parallel:
            from megatron.core.tensor_parallel import (
                scatter_to_sequence_parallel_region,
            )

            hidden_states = scatter_to_sequence_parallel_region(hidden_states)

        # Single EagleLayer
        hidden_states, _ = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            embeds=input_embeds,
        )

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states


# =============================================================================
# Eagle3ForCausalLM
# =============================================================================


class Eagle3ForCausalLM(MegatronModule):
    """Eagle3 causal LM wrapping ``EagleModel``.

    The LM head is **not** owned by this module — it is reused from the target
    model and passed to :meth:`forward` via the ``lm_head`` argument.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config = config
        self.model = EagleModel(config=config)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        input_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        compute_logits: bool = True,
    ) -> Union[Tensor, Tensor]:
        """Forward pass.

        Args:
            hidden_states: Auxiliary hidden states ``[s, b, 3*h]``.
            input_embeds: Token embeddings ``[s, b, h]``.
            attention_mask: Optional attention mask.
            compute_logits: Whether to compute logits (default ``True``).

        Returns:
            Logits ``[s, b, vocab]`` when ``compute_logits=True``, else
            hidden states ``[s, b, h]``.
        """
        hidden_states = self.model(
            hidden_states=hidden_states,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
        )

        if not compute_logits:
            return hidden_states

        logits, _ = self.lm_head(hidden_states)
        return logits.transpose(0, 1).contiguous()

    def combine_hidden_states(self, hidden_states: Tensor) -> Tensor:
        """FC-reduce auxiliary hidden states (vLLM interface compat)."""
        output, _ = self.model.fc(hidden_states)
        return output


# =============================================================================
# Weight Loading Utilities
# =============================================================================


def create_config_from_hf(
    hf_config,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
) -> TransformerConfig:
    """Create ``TransformerConfig`` from a HuggingFace model config."""
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads

    rotary_base = getattr(hf_config, "rope_theta", 10000)

    config = TransformerConfig(
        num_layers=getattr(hf_config, "num_hidden_layers", 1),
        hidden_size=hf_config.hidden_size,
        ffn_hidden_size=hf_config.intermediate_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        ),
        kv_channels=head_dim,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=getattr(hf_config, "rms_norm_eps", 1e-6),
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        normalization="RMSNorm",
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
    )

    config.vocab_size = hf_config.vocab_size
    config.rotary_base = rotary_base
    config.rotary_interleaved = False
    config.rotary_percent = 1.0

    return config


def load_hf_weights_to_eagle(
    model: Eagle3ForCausalLM,
    hf_state_dict: Dict[str, Tensor],
    config: TransformerConfig,
) -> Tuple[list, list]:
    """Load HuggingFace Eagle weights into the Megatron Eagle model.

    The module hierarchy (single layer, no LM head)::

        model.fc.weight
        model.norm.weight
        model.layer.input_layernorm.weight          # embeds branch norm
        model.layer.hidden_norm.weight               # hidden branch norm
        model.layer.self_attention.linear_qkv.weight
        model.layer.self_attention.linear_proj.weight
        model.layer.pre_mlp_layernorm.weight         # post-attention norm
        model.layer.mlp.linear_fc1.weight
        model.layer.mlp.linear_fc2.weight

    Note: ``lm_head`` weights are loaded from the target model, not here.
    """
    new_state: Dict[str, Tensor] = {}

    # Temporary accumulators for fused QKV / gate-up projections.
    q_weight: Optional[Tensor] = None
    k_weight: Optional[Tensor] = None
    v_weight: Optional[Tensor] = None
    gate_weight: Optional[Tensor] = None
    up_weight: Optional[Tensor] = None

    prefix = "model.layer"

    for hf_key, hf_weight in hf_state_dict.items():
        # ---- top-level modules ----
        if "fc.weight" in hf_key:
            new_state["model.fc.weight"] = hf_weight
            continue
        if "norm.weight" in hf_key and "layernorm" not in hf_key:
            new_state["model.norm.weight"] = hf_weight
            continue
        # lm_head is reused from the target model — skip.
        if "lm_head.weight" in hf_key:
            continue

        # ---- midlayer → model.layer ----
        if "midlayer." not in hf_key:
            continue

        # Attention Q / K / V (fused later)
        if "self_attn.q_proj.weight" in hf_key:
            q_weight = hf_weight
        elif "self_attn.k_proj.weight" in hf_key:
            k_weight = hf_weight
        elif "self_attn.v_proj.weight" in hf_key:
            v_weight = hf_weight
        elif "self_attn.o_proj.weight" in hf_key:
            new_state[f"{prefix}.self_attention.linear_proj.weight"] = hf_weight

        # MLP gate / up (fused later) and down
        elif "mlp.gate_proj.weight" in hf_key:
            gate_weight = hf_weight
        elif "mlp.up_proj.weight" in hf_key:
            up_weight = hf_weight
        elif "mlp.down_proj.weight" in hf_key:
            new_state[f"{prefix}.mlp.linear_fc2.weight"] = hf_weight

        # Norms
        elif "hidden_norm.weight" in hf_key:
            new_state[f"{prefix}.hidden_norm.weight"] = hf_weight
        elif "input_layernorm.weight" in hf_key:
            new_state[f"{prefix}.input_layernorm.weight"] = hf_weight
        elif "post_attention_layernorm.weight" in hf_key:
            new_state[f"{prefix}.pre_mlp_layernorm.weight"] = hf_weight

    # Fuse Q + K + V → linear_qkv
    if q_weight is not None:
        assert k_weight is not None and v_weight is not None
        new_state[f"{prefix}.self_attention.linear_qkv.weight"] = torch.cat(
            [q_weight, k_weight, v_weight], dim=0
        )

    # Fuse gate + up → linear_fc1
    if gate_weight is not None:
        assert up_weight is not None
        new_state[f"{prefix}.mlp.linear_fc1.weight"] = torch.cat(
            [gate_weight, up_weight], dim=0
        )

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return missing, unexpected


# =============================================================================
# Convenience
# =============================================================================


def create_eagle_from_hf_config(
    hf_config,
    tensor_model_parallel_size: int = 1,
) -> Eagle3ForCausalLM:
    """Create ``Eagle3ForCausalLM`` from a HuggingFace config object."""
    config = create_config_from_hf(
        hf_config,
        tensor_model_parallel_size=tensor_model_parallel_size,
    )
    # Eagle always uses a single transformer layer.
    config = replace(config, num_layers=1)
    return Eagle3ForCausalLM(config=config)
