# SPDX-License-Identifier: Apache-2.0
"""The metadata carrier handed to MaxText's model."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="PagedPlan holds device arrays")

from maxtext.inference.kv_common import KvPageTableV1
from maxtext.inference.kv_execution.bucketing import StepShape
from maxtext.inference.kv_execution.step_view import build_step_view
from maxtext.layers.gpu_paged_attention import PagedPlan, build_plan

from jax_vllm_gpu.plugin import paged_plan


def a_step_view(tokens_per_page=16):
    table = KvPageTableV1(
        page_ids=[[1, 2]],
        seq_lens=np.asarray([20], dtype=np.int32),
        query_lens=np.asarray([20], dtype=np.int32),
        write_positions=np.arange(20, dtype=np.int32),
        request_order=np.asarray([0], dtype=np.int32),
    )
    shape = StepShape(
        num_requests=4, num_tokens=32, num_pages=8, max_seqlen_k=64, is_decode=False
    )
    return build_step_view(table, shape, tokens_per_page)


def test_maxtext_accepts_it_as_a_plan_without_reconstructing_one():
    """`build_plan` short-circuits on `isinstance(metadata, PagedPlan)`. Failing
    that check would send it down `plan_from_vllm` instead, which rebuilds from
    block tables that this object does not carry."""
    plan = paged_plan.from_step_view(a_step_view(), np.arange(32, dtype=np.int32))

    assert isinstance(plan, PagedPlan)
    assert build_plan(plan, tokens_per_page=16, total_tokens=32, max_seqlen_k=64) is plan


def test_it_carries_the_positions_the_adapter_reads():
    """MaxText's vLLM adapter drives RoPE from `metadata.input_positions`, and a
    bare PagedPlan has no such field."""
    positions = np.arange(32, dtype=np.int32)
    plan = paged_plan.from_step_view(a_step_view(), positions)

    assert np.array_equal(np.asarray(plan.input_positions), positions)


def test_padded_tokens_are_marked_to_be_skipped_rather_than_written():
    """`append_kv` treats a negative slot as "skip this token". This is why the
    StepView route is used instead of the vLLM-shaped one: a plan derived from
    padded block tables resolves those rows to real pages and writes junk."""
    plan = paged_plan.from_step_view(a_step_view(), np.arange(32, dtype=np.int32))
    slots = np.asarray(plan.slot_mapping)

    assert (slots[:20] >= 0).all()
    assert (slots[20:] == -1).all()
