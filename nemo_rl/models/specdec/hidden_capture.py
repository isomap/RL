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

"""Hidden state capture for Eagle/speculative decoding training.

This module provides utilities to capture intermediate hidden states from a
policy model during forward pass, which are used as inputs for training the
Eagle speculative decoding model.

The capture mechanism supports both PP=1 and PP>1 configurations with a unified API.
For PP>1, hidden states are gathered from all pipeline stages to the last stage
where the loss is computed.

Based on vLLM's Eagle3 implementation pattern:
- Auxiliary layers are typically: (2, num_layers // 2, num_layers - 3) for 0-indexed layers
- Hidden states are captured BEFORE each auxiliary layer's forward pass
- The captured state is: hidden_states + residual (pre-normalized input)
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import ContextManager, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from torch import Tensor, nn


def get_eagle3_aux_hidden_state_layers(num_layers: int) -> Tuple[int, ...]:
    """Get default auxiliary hidden state layer indices for Eagle3.

    Following vLLM's convention, returns 0-indexed global layer indices.

    Args:
        num_layers: Total number of transformer layers in the model.

    Returns:
        Tuple of layer indices (0-indexed) to capture hidden states from.
    """
    return (2, num_layers // 2, num_layers - 3)


@dataclass
class CapturedStates:
    """Container for captured hidden states from policy model."""

    # Auxiliary hidden states from specified layers, concatenated along hidden dim
    # Shape: [S, B, num_aux_layers * H] in Megatron format
    hidden_states: Optional[Tensor] = None

    # Token embeddings from the embedding layer
    # Shape: [S, B, H] in Megatron format
    inputs_embeds: Optional[Tensor] = None

    def is_complete(self) -> bool:
        """Check if all required states have been captured."""
        return self.hidden_states is not None and self.inputs_embeds is not None


class HiddenStateCapture:
    """Captures hidden states from policy model for Eagle training.

    This class provides a unified API for capturing hidden states that works
    with both PP=1 and PP>1 configurations. For PP>1, it handles the distributed
    gathering of hidden states to the last pipeline stage.

    Usage:
        capture = HiddenStateCapture(model, num_layers=32)

        with capture.capture_context():
            logits = model(input_ids, ...)

        states = capture.get_captured_states()
        # states.hidden_states: [S, B, 3*H]
        # states.inputs_embeds: [S, B, H]

    Note:
        - For PP>1, `get_captured_states()` returns `CapturedStates` only on the
          last PP stage. Other stages return `CapturedStates` with None fields.
        - Hidden states are always detached (no gradients flow to policy model).
    """

    def __init__(
        self,
        model: nn.Module,
        aux_layer_indices: Optional[Tuple[int, ...]] = None,
    ):
        """Initialize the hidden state capture.

        Args:
            model: The policy model (GPTModel or wrapped version).
                   Must have `decoder.layers` attribute.
            num_layers: Total number of transformer layers in the full model.
            aux_layer_indices: 0-indexed global layer indices to capture.
                              If None, uses Eagle3 default: (2, num_layers//2, num_layers-3).
        """
        self.model = self._unwrap_model(model)
        self.num_layers = self.model.config.num_layers

        # Determine auxiliary layer indices
        if aux_layer_indices is None:
            self.aux_layer_indices = get_eagle3_aux_hidden_state_layers(self.num_layers)
        else:
            self.aux_layer_indices = tuple(aux_layer_indices)

        # Pipeline parallelism info
        self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.is_first_stage = parallel_state.is_pipeline_first_stage()
        self.is_last_stage = parallel_state.is_pipeline_last_stage()

        # Calculate which layers are on this PP stage
        self._compute_local_layer_mapping()

        # Storage for captured states
        self._captured: Dict[str, Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Unwrap model from DDP/Float16Module wrappers."""
        while hasattr(model, "module"):
            model = model.module
        return model

    def _compute_local_layer_mapping(self) -> None:
        """Compute mapping between global and local layer indices.

        For PP>1, each stage only has a subset of layers. We need to map
        global aux layer indices to local layer indices within this stage.
        """
        # Get layers on this stage
        decoder_layers = self.model.decoder.layers

        # Build mapping: global_layer_idx (0-indexed) -> local_layer_idx
        self._global_to_local: Dict[int, int] = {}
        self._local_aux_indices: List[int] = []

        for local_idx, layer in enumerate(decoder_layers):
            # layer.layer_number is 1-indexed global
            global_idx = layer.layer_number - 1  # Convert to 0-indexed

            if global_idx in self.aux_layer_indices:
                self._global_to_local[global_idx] = local_idx
                self._local_aux_indices.append(local_idx)

        # Track which global indices are on this stage
        self._local_global_indices = [
            layer.layer_number - 1 for layer in decoder_layers
        ]

    def _make_layer_hook(self, global_idx: int):
        """Create a forward hook for capturing layer input.

        We capture the INPUT to each auxiliary layer (before the layer processes it).
        In Megatron's TransformerLayer, the input is typically (hidden_states, ...).

        For proper Eagle input, we want the pre-normalized hidden state which is
        hidden_states + residual in fused norm cases, or just hidden_states otherwise.
        """

        def hook(module, args, kwargs):
            # TransformerLayer.forward signature:
            # forward(hidden_states, attention_mask=None, context=None, ...)
            # We capture hidden_states (first positional arg)
            if args:
                hidden_states = args[0]
            elif "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            else:
                return  # Can't capture

            # Detach to prevent gradients flowing to policy
            self._captured[f"layer_{global_idx}"] = hidden_states.detach().clone()

        return hook

    def _make_embedding_hook(self):
        """Create a forward hook for capturing embedding output."""

        def hook(module, args, output):
            # LanguageModelEmbedding returns the embedded tokens
            # Output shape: [S, B, H]
            self._captured["embeds"] = output.detach().clone()

        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on the model.

        This registers hooks on:
        1. Embedding layer (if on first PP stage) - captures token embeddings
        2. Auxiliary layers (that are on this PP stage) - captures hidden states
        """
        self.clear_hooks()

        # Register embedding hook on first stage
        if self.is_first_stage and hasattr(self.model, "embedding"):
            hook = self.model.embedding.register_forward_hook(
                self._make_embedding_hook()
            )
            self._hooks.append(hook)

        # Register hooks on auxiliary layers that are on this stage
        decoder_layers = self.model.decoder.layers
        for local_idx in self._local_aux_indices:
            layer = decoder_layers[local_idx]
            global_idx = layer.layer_number - 1

            # Use pre-forward hook to capture input to the layer
            hook = layer.register_forward_pre_hook(
                self._make_layer_hook(global_idx),
                with_kwargs=True,
            )
            self._hooks.append(hook)

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        # self._captured.clear()

    @contextmanager
    def capture_context(self):
        """Context manager for capturing hidden states during forward pass.

        Usage:
            with capture.capture_context():
                output = model(input_ids, ...)
        """
        try:
            self.register_hooks()
            yield self
        finally:
            self.clear_hooks()

    def _gather_to_last_stage(self) -> CapturedStates:
        """Gather captured states from all PP stages to the last stage.

        For PP=1, this is a no-op that just assembles local captures.
        For PP>1, this performs distributed communication to collect states.

        Returns:
            CapturedStates with all hidden states on last stage,
            or CapturedStates with None fields on other stages.
        """
        if self.pp_size == 1:
            return self._assemble_local_states()

        # PP > 1: Need to gather states across stages
        return self._gather_distributed()

    def _assemble_local_states(self) -> CapturedStates:
        """Assemble captured states for PP=1 case."""
        # Get embeddings
        embeds = self._captured.get("embeds")

        # Get and concatenate auxiliary hidden states
        hidden_list = []
        for global_idx in sorted(self.aux_layer_indices):
            key = f"layer_{global_idx}"
            if key in self._captured:
                hidden_list.append(self._captured[key])

        if not hidden_list:
            return CapturedStates(inputs_embeds=embeds)

        # Concatenate along hidden dimension: [S, B, H] * N -> [S, B, N*H]
        hidden_states = torch.cat(hidden_list, dim=-1)

        return CapturedStates(
            hidden_states=hidden_states,
            inputs_embeds=embeds,
        )

    def _gather_distributed(self) -> CapturedStates:
        """Gather states across PP stages using point-to-point communication.

        Communication pattern:
        - Each stage sends its local captures to the last stage
        - Last stage receives and assembles all captures
        - Non-last stages return empty CapturedStates
        """
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        last_rank = self.pp_size - 1

        # Determine tensor metadata for communication
        # We need to know shapes before receiving
        sample_tensor = None
        for v in self._captured.values():
            if v is not None:
                sample_tensor = v
                break

        # If this stage has no captures and is not last stage, nothing to do
        if sample_tensor is None and not self.is_last_stage:
            return CapturedStates()

        # Gather hidden states from auxiliary layers
        gathered_hidden: Dict[int, Tensor] = {}

        for global_idx in self.aux_layer_indices:
            owner_rank = self._get_owner_rank(global_idx)
            key = f"layer_{global_idx}"

            if self.pp_rank == owner_rank:
                # I own this layer
                tensor = self._captured.get(key)
                if tensor is not None:
                    if self.is_last_stage:
                        gathered_hidden[global_idx] = tensor
                    else:
                        # Send to last stage
                        dist.send(tensor.contiguous(), dst=last_rank, group=pp_group)

            elif self.is_last_stage:
                # Receive from owner
                # Need to get shape from a local tensor or use a handshake
                # For simplicity, assume all hidden states have same shape as local ones
                if sample_tensor is not None:
                    recv_tensor = torch.empty_like(sample_tensor)
                else:
                    # Last stage has no local captures - need shape info
                    # This is a limitation - in practice, last stage usually has captures
                    continue

                dist.recv(recv_tensor, src=owner_rank, group=pp_group)
                gathered_hidden[global_idx] = recv_tensor

        # Gather embeddings from first stage
        gathered_embeds = None

        if self.is_first_stage:
            embeds = self._captured.get("embeds")
            if embeds is not None:
                if self.is_last_stage:
                    # First == Last (PP=1 shouldn't reach here, but handle anyway)
                    gathered_embeds = embeds
                else:
                    # Send to last stage
                    dist.send(embeds.contiguous(), dst=last_rank, group=pp_group)

        elif self.is_last_stage:
            # Receive from first stage
            if sample_tensor is not None:
                # Embeddings have same S, B dims but just H (not N*H)
                # Infer shape from hidden states
                s, b, _ = sample_tensor.shape
                h = sample_tensor.shape[-1] // len(self.aux_layer_indices)
                recv_embeds = torch.empty(
                    s, b, h, dtype=sample_tensor.dtype, device=sample_tensor.device
                )
                dist.recv(recv_embeds, src=0, group=pp_group)
                gathered_embeds = recv_embeds

        # Only last stage assembles the result
        if not self.is_last_stage:
            return CapturedStates()

        # Assemble on last stage
        if gathered_hidden:
            hidden_list = [
                gathered_hidden[idx] for idx in sorted(gathered_hidden.keys())
            ]
            hidden_states = torch.cat(hidden_list, dim=-1)
        else:
            hidden_states = None

        return CapturedStates(
            hidden_states=hidden_states,
            inputs_embeds=gathered_embeds,
        )

    def _get_owner_rank(self, global_layer_idx: int) -> int:
        """Determine which PP rank owns a given global layer index.

        Args:
            global_layer_idx: 0-indexed global layer number.

        Returns:
            PP rank (0-indexed) that owns this layer.
        """
        if self.pp_size == 1:
            return 0

        # Simple uniform distribution (may need adjustment for uneven PP)
        layers_per_rank = self.num_layers // self.pp_size
        return min(global_layer_idx // layers_per_rank, self.pp_size - 1)

    def get_captured_states(self) -> CapturedStates:
        """Get the captured hidden states after forward pass.

        For PP=1: Returns complete CapturedStates.
        For PP>1: Returns complete CapturedStates only on last stage,
                  other stages return CapturedStates with None fields.

        Returns:
            CapturedStates containing:
            - hidden_states: [S, B, num_aux_layers * H] - concatenated aux hidden states
            - inputs_embeds: [S, B, H] - token embeddings
        """
        return self._gather_to_last_stage()


def create_hidden_capture(
    model: nn.Module,
    num_layers: int,
    aux_layer_indices: Optional[List[int]] = None,
) -> HiddenStateCapture:
    """Factory function to create a HiddenStateCapture instance.

    This provides a simple API for creating the capture utility.

    Args:
        model: The policy model.
        num_layers: Total number of transformer layers.
        aux_layer_indices: Optional list of 0-indexed layer indices to capture.
                          Defaults to Eagle3 pattern: (2, num_layers//2, num_layers-3).

    Returns:
        HiddenStateCapture instance ready for use.

    Example:
        capture = create_hidden_capture(policy_model, num_layers=32)

        with capture.capture_context():
            logits = policy_model(input_ids)

        states = capture.get_captured_states()
        if states.is_complete():
            eagle_input = states.hidden_states  # [S, B, 3*H]
            embeds = states.inputs_embeds       # [S, B, H]
    """
    return HiddenStateCapture(
        model=model,
        num_layers=num_layers,
        aux_layer_indices=tuple(aux_layer_indices) if aux_layer_indices else None,
    )


def get_capture_context(
    model: nn.Module,
    specdec_config: Optional[Dict] = None,
) -> Tuple[ContextManager, Optional[HiddenStateCapture]]:
    """Get capture context manager and capture instance for clean integration.

    This factory function provides a clean way to integrate hidden state capture
    into forward passes using nullcontext pattern. Returns (nullcontext(), None)
    when specdec is disabled, allowing clean code without nested conditionals.

    Args:
        model: The policy model (GPTModel or wrapped version).
        specdec_config: Speculative decoding configuration dict.
                       Required keys:
                       - "enabled" (bool): Whether specdec is enabled
                       Optional keys:
                       - "aux_layer_indices" (List[int]): Custom layer indices

    Returns:
        Tuple of:
        - ContextManager: Either capture.capture_context() or nullcontext()
        - Optional[HiddenStateCapture]: The capture instance, or None if disabled

    Example:
        # Clean integration in forward_step_arbitrary_loss:
        capture_ctx, capture = get_capture_context(model, num_layers, specdec_config)

        with capture_ctx:
            output_tensor = model(input_ids, ...)

        states = capture.get_captured_states() if capture else None

        if states and states.is_complete():
            # On last PP stage with complete states
            loss_fn = SpecDecLossWrapper(
                loss_fn=loss_fn,
                specdec_model=specdec_model,
                captured_hidden_states=states.hidden_states,
                captured_inputs_embeds=states.inputs_embeds,
                ...
            )
    """
    # Return no-op context if specdec is disabled
    if not specdec_config or not specdec_config.get("enabled", False):
        return nullcontext(), None

    # Create capture instance
    aux_indices = specdec_config.get("aux_layer_indices", None)
    capture = HiddenStateCapture(
        model=model,
        aux_layer_indices=tuple(aux_indices) if aux_indices else None,
    )

    return capture.capture_context(), capture
