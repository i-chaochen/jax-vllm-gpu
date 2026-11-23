# SPDX-License-Identifier: Apache-2.0
"""Translate vLLM's per-request block ids into MaxText's neutral page table.

vLLM owns allocation, so it hands the runner block ids plus a count of tokens
already computed. MaxText's kernels want a `KvPageTableV1`. This module is the
only place that conversion happens.

Two details carry the correctness of the whole step:

* vLLM over-allocates. A request's block list can be longer than its context
  needs, and `KvPageTableV1.validate` requires exactly `ceil(seq_len/page)`
  pages, so the list is truncated rather than passed through.
* `write_positions` are absolute positions within the sequence, which is what
  makes RoPE agree between prefill and decode. They start at
  `num_computed_tokens`, not at zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from vllm_xla_gpu.kv.common import KvPageTableV1


@dataclass(frozen=True)
class RequestPages:
    """One request's contribution to a step, as vLLM describes it.

    `num_computed_tokens` counts tokens whose KV is already in the pool, whether
    written by this request earlier or supplied by a prefix-cache hit. The
    distinction does not matter here: either way those positions are not written
    again this step.
    """

    request_id: str
    block_ids: Sequence[int]
    num_computed_tokens: int
    num_scheduled_tokens: int

    @property
    def seq_len(self) -> int:
        """Context length after this step completes."""
        return self.num_computed_tokens + self.num_scheduled_tokens

    def pages_needed(self, tokens_per_page: int) -> int:
        return -(-self.seq_len // tokens_per_page)


def build_page_table(
    requests: Sequence[RequestPages],
    *,
    tokens_per_page: int,
) -> KvPageTableV1:
    """Build the page table for one step, in the order `requests` is given."""
    if tokens_per_page <= 0:
        raise ValueError(f"tokens_per_page must be positive, got {tokens_per_page}")

    page_ids: list[list[int]] = []
    seq_lens: list[int] = []
    query_lens: list[int] = []
    write_positions: list[np.ndarray] = []

    for request in requests:
        if request.num_scheduled_tokens <= 0:
            raise ValueError(
                f"request {request.request_id!r} has no scheduled tokens; it "
                "should not be in the batch"
            )
        if request.num_computed_tokens < 0:
            raise ValueError(
                f"request {request.request_id!r} has negative num_computed_tokens"
            )

        needed = request.pages_needed(tokens_per_page)
        available = len(request.block_ids)
        if available < needed:
            raise ValueError(
                f"request {request.request_id!r} needs {needed} pages for "
                f"{request.seq_len} tokens but vLLM supplied {available}"
            )

        page_ids.append(list(request.block_ids[:needed]))
        seq_lens.append(request.seq_len)
        query_lens.append(request.num_scheduled_tokens)
        write_positions.append(
            np.arange(
                request.num_computed_tokens,
                request.seq_len,
                dtype=np.int32,
            )
        )

    table = KvPageTableV1(
        page_ids=page_ids,
        seq_lens=np.asarray(seq_lens, dtype=np.int32),
        query_lens=np.asarray(query_lens, dtype=np.int32),
        write_positions=(
            np.concatenate(write_positions)
            if write_positions
            else np.empty((0,), dtype=np.int32)
        ),
        request_order=np.arange(len(requests), dtype=np.int32),
    )
    table.validate(tokens_per_page)
    return table
