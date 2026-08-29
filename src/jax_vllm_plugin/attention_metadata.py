# SPDX-License-Identifier: Apache-2.0
"""The attention metadata MaxText's model receives.

`tpu_inference.AttentionMetadata` fills this role for the TPU plugin, and
MaxText's adapter is typed against it. MaxText's GPU paged attention does not
depend on that class though -- it duck-types, requiring only `block_tables`,
`seq_lens` and `query_start_loc`:

    def is_vllm_metadata(metadata: Any) -> bool:
      return all(hasattr(metadata, name)
                 for name in ("block_tables", "seq_lens", "query_start_loc"))

so a plain dataclass carrying those is sufficient, plus `input_positions` which
the adapter reads directly.

Arrays stay numpy. `plan_from_vllm` calls `jnp.asarray` on them itself, and
keeping host arrays here means the metadata can be built and asserted on without
initialising a device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maxtext.inference.kv_common import KvPageTableV1


@dataclass
class JaxAttentionMetadata:
    """Duck-typed to satisfy MaxText's `is_vllm_metadata`."""

    block_tables: np.ndarray
    seq_lens: np.ndarray
    query_start_loc: np.ndarray
    input_positions: np.ndarray

    @property
    def num_seqs(self) -> int:
        return int(self.block_tables.shape[0])


def build_attention_metadata(
    table: KvPageTableV1,
    *,
    max_blocks_per_seq: int | None = None,
) -> JaxAttentionMetadata:
    """Convert the neutral page table into vLLM-shaped metadata.

    `block_tables` is the 2-D `[num_seqs, max_blocks_per_seq]` form vLLM uses,
    with short rows padded. Padding uses the reserved page rather than an
    arbitrary value, so a row read past its length addresses a page that is
    guaranteed to read as zero rather than another request's KV.
    """
    num_seqs = table.num_requests
    widest = max((len(pages) for pages in table.page_ids), default=0)
    width = max(widest, 1) if max_blocks_per_seq is None else max_blocks_per_seq
    if width < widest:
        raise ValueError(
            f"max_blocks_per_seq={max_blocks_per_seq} is too small for a request "
            f"holding {widest} pages"
        )

    block_tables = np.zeros((num_seqs, width), dtype=np.int32)
    for row, pages in enumerate(table.page_ids):
        block_tables[row, : len(pages)] = pages

    query_start_loc = np.zeros(num_seqs + 1, dtype=np.int32)
    np.cumsum(table.query_lens, out=query_start_loc[1:])

    return JaxAttentionMetadata(
        block_tables=block_tables,
        seq_lens=np.asarray(table.seq_lens, dtype=np.int32),
        query_start_loc=query_start_loc,
        input_positions=np.asarray(table.write_positions, dtype=np.int32),
    )
