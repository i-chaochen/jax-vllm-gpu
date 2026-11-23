# SPDX-License-Identifier: Apache-2.0
"""The standalone engine's paged settings, which MaxText's config does not carry."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class PagedConfig:
  """The page pool `PagedEngine` serves from.

  Attributes:
    page_size: tokens per KV page, `block_size` in vLLM's vocabulary. Smaller
      wastes less on partial final pages and costs more page-table indirection.
      The kernels accept 16 and 32.
    num_blocks: usable pages, which sets serving capacity: the pool holds
      `num_blocks * page_size` tokens across all requests at once. One more page
      is allocated as the padding target. There is no safe default: too small
      silently caps concurrency, too large fails at allocation.
    max_context_len: the longest context one request may reach, which bounds
      its page-table row and the sequence-length bucket ladder. 0 means the
      model's `max_target_length`.
    enable_prefix_cache: share computed pages between requests whose prompts
      begin with the same tokens. Off by default, because a hit is sound only
      when the caller supplies a cache namespace that distinguishes everything
      affecting the KV: weights, adapter, tokenizer, RoPE and quantisation.
    poison_freed_pages: fill a freed page with a sentinel, so a missed scrub
      produces an obviously wrong value rather than a plausible one. Costs a
      pool write per freed page.
  """

  page_size: int = 16
  num_blocks: int = 0
  max_context_len: int = 0
  enable_prefix_cache: bool = False
  poison_freed_pages: bool = False

  def resolve(self, model_config: Any) -> "PagedConfig":
    """Check these settings against the MaxText config they serve.

    Returns a copy with `max_context_len` filled in from the model when it was
    left at 0.
    """
    if model_config.attention != "gpu_paged":
      raise ValueError(
          f"the paged engine requires attention='gpu_paged', but the model is configured for "
          f"'{model_config.attention}'."
      )
    if model_config.scan_layers:
      raise ValueError(
          "attention='gpu_paged' requires scan_layers=False. Scanning stacks the per-layer KV "
          "caches into the scan carry, which copies the whole page pool every step."
      )
    if self.page_size < 1:
      raise ValueError(f"page_size must be positive, got {self.page_size}")
    if self.num_blocks < 1:
      raise ValueError(
          "num_blocks must be set: it is the pool's capacity. Pool tokens = num_blocks * "
          f"page_size (currently {self.page_size})."
      )

    max_context_len = self.max_context_len or int(model_config.max_target_length)
    request_pages = -(-max_context_len // self.page_size)
    if request_pages > self.num_blocks:
      raise ValueError(
          f"a single max_context_len={max_context_len} request needs {request_pages} pages but "
          f"num_blocks is {self.num_blocks}, so no request could ever finish. Raise num_blocks or "
          "lower max_context_len."
      )
    return dataclasses.replace(self, max_context_len=max_context_len)
