# Copyright 2023–2025 Google LLC
# Modifications Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A MaxText model as the `MaxTextForCausalLM` the vLLM plugin serves.

MaxText ships an adapter of the same shape for the TPU plugin, which registers
into tpu_inference and pads MoE layers for TPU kernels. This one serves the GPU
platform: it registers into whatever registry it is handed and has no TPU
dependency.
"""

import os
from typing import Any

from flax import nnx
import flax.linen as nn
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig
from maxtext.integration.vllm.hybrid_cache_utils import normalize_vllm_input_positions
from maxtext.utils import lora_utils
from maxtext.utils import max_logging
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from vllm.config import VllmConfig

MODEL_NAME = "MaxTextForCausalLM"


def parallel_degrees(vllm_config: VllmConfig) -> tuple[int, int]:
  """Tensor and expert parallel degrees, as (tp, ep)."""
  parallel_config = vllm_config.parallel_config
  return (
      parallel_config.tensor_parallel_size,
      getattr(parallel_config, "expert_parallel_size", 1) or 1,
  )


def generate_maxtext_config(vllm_config: VllmConfig) -> pyconfig.HyperParameters:
  """The MaxText config for a vLLM config.

  Starts from MaxText's `configs/inference/vllm.yml` and applies
  `additional_config["maxtext_config"]`, which the platform has already given
  `attention="gpu_paged"` and the page size. With `load_format="dummy"` no
  checkpoint is loaded.
  """
  if "maxtext_config" in vllm_config.additional_config:
    overrides = vllm_config.additional_config["maxtext_config"]
  else:
    overrides = {}

  if vllm_config.load_config.load_format == "dummy":
    if overrides.get("load_parameters_path") is not None:
      max_logging.log(
          "Warning: load_parameters_path is set when using dummy load format. Checkpoint loading will be skipped."
      )
      overrides["load_parameters_path"] = None

  argv_list = ["", os.path.join(MAXTEXT_CONFIGS_DIR, "inference", "vllm.yml")]

  tp, ep = parallel_degrees(vllm_config)
  kv_tp_size = tp * ep

  hf_config = (
      vllm_config.model_config.hf_config.text_config
      if hasattr(vllm_config.model_config.hf_config, "text_config")
      else vllm_config.model_config.hf_config
  )
  num_kv_heads = hf_config.num_key_value_heads

  # Number of KV heads in global attention layers (None if the field is absent or unset).
  num_global_kv_heads = getattr(hf_config, "num_global_key_value_heads", None)
  use_global_kv_heads = num_global_kv_heads is not None

  max_logging.log(
      f"vLLM parallel config: kv_heads={num_kv_heads}, global_kv_heads={num_global_kv_heads}, tp={tp}, ep={ep}"
  )

  # Replicate the KV heads when there are fewer than the degree of model
  # parallelism, so every shard holds a whole head.
  if kv_tp_size % num_kv_heads == 0 and num_kv_heads < kv_tp_size:
    max_logging.log(
        f"Padding num_kv_heads from {num_kv_heads} to {kv_tp_size} to match the degree of tensor parallelism."
    )
    overrides["base_num_kv_heads"] = kv_tp_size

  if use_global_kv_heads and kv_tp_size % num_global_kv_heads == 0 and num_global_kv_heads < kv_tp_size:
    max_logging.log(
        f"Padding num_global_kv_heads from {num_global_kv_heads} "
        f"to {kv_tp_size} to match the degree of tensor parallelism."
    )
    overrides["global_num_kv_heads"] = kv_tp_size

  return pyconfig.initialize(argv_list, **overrides)


class MaxTextForCausalLM(nnx.Module):
  """A MaxText causal language model behind vLLM's model interface."""

  def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh):
    self.vllm_config = vllm_config
    self.cfg = vllm_config.model_config
    self.maxtext_config = generate_maxtext_config(vllm_config)

    self.mesh = mesh
    self.model_mode = MODEL_MODE_AUTOREGRESSIVE
    self.is_text_generation_model = True

    self.model: nnx.Module | None = None

    if vllm_config.load_config.load_format == "dummy":
      self.load_weights(rng_key)

    elif self.maxtext_config.load_parameters_path is None:
      max_logging.log("Warning: No load_parameters_path provided. The model will be initialized with random weights.")

  def modules(self):
    """Dummy method to satisfy vLLM's internal cleanup logic."""
    return []

  def __call__(
      self,
      kv_caches: list[jax.Array],
      input_ids: jax.Array,
      attention_metadata: Any,
      *args,
      **kwargs,
  ) -> tuple[list[jax.Array], jax.Array, list[jax.Array], list[jax.Array] | None]:
    """One forward pass over vLLM's flattened tokens.

    Returns `(kv_caches, hidden, aux_hidden_states, expert_indices)`. Under
    `attention="gpu_paged"` MaxText's model returns logits in the `hidden`
    position, so a caller must sample from it directly and not apply the output
    head again.
    """
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model must be an instance of type nnx.Module.")

    # Hybrid models pass metadata per layer. Their linear-attention layers
    # index state by `mamba_state_indices` rather than by block tables, and all
    # full-attention layers share one block table, so take the metadata of a
    # full-attention layer when there is one.
    if isinstance(attention_metadata, dict):
      hf_text_config = getattr(self.cfg, "hf_text_config", getattr(self.cfg, "hf_config", None))
      layer_types = getattr(hf_text_config, "layer_types", None) or []
      attention_metadata_picked = None
      for i, lt in enumerate(layer_types):
        if lt != "linear_attention":
          attention_metadata_picked = attention_metadata.get(f"layer.{i}")
          if attention_metadata_picked is not None:
            break
      if attention_metadata_picked is None:
        attention_metadata_picked = next(iter(attention_metadata.values()))
      attention_metadata = attention_metadata_picked

    # MaxText decode treats vLLM's flattened tokens as a batch with seq_len=1.
    # MRoPE positions arrive channel-first and must also move their 3 channels
    # to MaxText's trailing dimension.
    input_ids = jnp.expand_dims(input_ids, axis=1)
    input_positions = normalize_vllm_input_positions(attention_metadata.input_positions)

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      aux_hidden_states = []
      expert_indices = None
      hidden, kv_caches = self.model(
          decoder_input_tokens=input_ids,
          decoder_positions=input_positions,
          kv_caches=kv_caches,
          attention_metadata=attention_metadata,
          model_mode=self.model_mode,
          **kwargs,
      )

      # To be compatible with vLLM, we reshape to (batch * seq, dim).
      hidden = hidden.reshape((-1, hidden.shape[-1]))

    return kv_caches, hidden, aux_hidden_states, expert_indices

  def forward(self, *args, **kwargs):
    """Alias for `__call__`."""
    return self(*args, **kwargs)

  def get_input_embeddings(self) -> jax.Array:
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      return self.model.token_embedder.embedding

  def embed_input_ids(self, input_ids: jax.Array) -> jax.Array:
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      return self.model.token_embedder(input_ids)

  def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
    """Logits from hidden states through MaxText's output head.

    Not for `attention="gpu_paged"` output, which is already logits.
    """
    if not isinstance(self.model, nnx.Module):
      raise ValueError("Model is not initialized.")

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      y = jnp.expand_dims(hidden_states, axis=1)
      logits = self.model.decoder.apply_output_head(self.model.token_embedder, y, True, self.model_mode)
      return logits.squeeze(1)

  def load_weights(self, rng_key: jax.Array) -> None:
    """Build the model and load its parameters, once."""
    if self.model is not None:
      return

    with self.mesh, nn.logical_axis_rules(self.maxtext_config.logical_axis_rules):
      model = model_creation_utils.from_pretrained(
          self.maxtext_config, mesh=self.mesh, model_mode=self.model_mode, rng_key=rng_key
      )
      if self.maxtext_config.lora.enable_lora:
        model = lora_utils.apply_lora_to_model(model, self.mesh, self.maxtext_config)
        if self.maxtext_config.lora.lora_restore_path:
          lora_utils.restore_lora_from_path(model, self.maxtext_config)
      self.model = nnx.data(model)

  def get_mrope_input_positions(
      self,
      input_tokens: list[int],
      mm_features: list = None,
  ) -> tuple[jax.Array, int]:
    """Dummy M-RoPE positions and delta for text-only MaxText."""
    seq_len = len(input_tokens)
    pos_range = jnp.arange(seq_len, dtype=jnp.int32)
    positions = jnp.stack([pos_range, pos_range, pos_range], axis=0)
    return positions, 0


def register(register_model) -> None:
  """Register `MaxTextForCausalLM` with a platform's model registry.

  `register_model` takes `(name, cls)`.
  """
  register_model(MODEL_NAME, MaxTextForCausalLM)
