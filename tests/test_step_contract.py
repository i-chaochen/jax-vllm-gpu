# SPDX-License-Identifier: Apache-2.0
"""The execute_model / sample_tokens split, which vLLM requires and which this
plugin originally got wrong.

The first implementation sampled inside `execute_model`, returned a complete
`ModelRunnerOutput`, and made `sample_tokens` raise. That reading came from
`EngineCore.step`, which calls `sample_tokens` only when `execute_model` returns
None -- so on that path it worked. The engine has a second path,
`step_with_batch_queue`, and it queues the **sample_tokens** future and reads the
output from that, discarding whatever `execute_model` returned. Under it the
worker raised on every request.

The lesson is why these tests are shaped the way they are. A unit test written
at the time would have encoded the same misreading and passed, because it would
have been written against the same belief. What has teeth is pinning the state
machine so the two calls cannot be collapsed back together by a later refactor
that only checks `step`, and running the real engine, which `tools/smoke_engine.py`
now does across four scheduling modes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax", reason="the runner holds device arrays")
pytest.importorskip("vllm", reason="vLLM supplies ModelRunnerOutput")

import jax
import jax.numpy as jnp

from maxtext.inference.kv_common import KvStorageLayoutV1
from maxtext.inference.kv_execution.bucketing import StepShapePlanner
from maxtext.inference.kv_execution.pool_factory import allocate_pool

from jax_vllm_plugin import pool as pool_module
from jax_vllm_plugin.kv_reuse import PageReuseTracker
from jax_vllm_plugin.model_runner import JaxModelRunner

PAGE = 16
NUM_PAGES = 32
VOCAB = 64


def a_runner():
    """A runner wired far enough to step, with the model replaced by a stub.

    Everything below the model call is real -- the tracker, the pool, the
    bucketing, the page tables -- because those are what the step sequence
    manipulates. Only the forward pass is faked, since a real one needs a
    checkpoint and proves nothing about call ordering.
    """
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=256, seed=0, get_vocab_size=lambda: VOCAB
        ),
        cache_config=SimpleNamespace(block_size=PAGE),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=256),
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1), ("model",))
    runner = JaxModelRunner(vllm_config, mesh)

    runner.layout = KvStorageLayoutV1(
        tokens_per_page=PAGE,
        num_pages=NUM_PAGES,
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype="bfloat16",
        padding_page_id=0,
    )
    runner.pool = allocate_pool(runner.layout)
    runner.scrub = pool_module.make_scrub_fn(runner.pool)
    runner.tracker = PageReuseTracker(num_pages=NUM_PAGES, tokens_per_page=PAGE)
    runner.planner = StepShapePlanner(
        tokens_per_page=PAGE, max_batch=4, max_context_len=256, pool_pages=NUM_PAGES
    )
    runner._rng = jax.random.key(0)
    runner._state_leaves = ()

    def forward(state_leaves, kv_caches, *args, **kwargs):
        rows = kwargs.get("_rows") or args[-1].shape[0]
        return kv_caches, jnp.zeros((rows, VOCAB), jnp.float32)

    runner._forward_fn = forward
    runner._sample_fn = lambda rng, logits, *a, **k: jnp.argmax(logits, axis=-1)
    return runner


def a_scheduler_output(prompt_len=4, blocks=(1,)):
    """The minimum a first prefill step carries, duck-typed."""
    request = SimpleNamespace(
        req_id="r0",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SimpleNamespace(temperature=0.0, top_p=1.0, top_k=0),
        block_ids=(list(blocks),),
        num_computed_tokens=0,
    )
    return SimpleNamespace(
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=None,
        finished_req_ids=(),
        num_scheduled_tokens={"r0": prompt_len},
        total_num_scheduled_tokens=prompt_len,
    )


def test_execute_model_returns_none_so_the_engine_asks_for_tokens():
    """None is the signal. `step` samples only when it sees None, and
    `step_with_batch_queue` reads the output from `sample_tokens` regardless --
    so None is the one answer both paths handle."""
    runner = a_runner()

    assert runner.execute_model(a_scheduler_output()) is None


def test_sample_tokens_produces_the_output_the_engine_consumes():
    runner = a_runner()
    runner.execute_model(a_scheduler_output())

    output = runner.sample_tokens(None)

    assert output.req_ids == ["r0"]
    assert output.req_id_to_index == {"r0": 0}
    assert len(output.sampled_token_ids) == 1


def test_a_second_execute_model_without_sampling_is_refused():
    """The stashed logits would be dropped silently, and the step they belonged
    to would report no token."""
    runner = a_runner()
    runner.execute_model(a_scheduler_output())

    with pytest.raises(RuntimeError, match="twice without sample_tokens"):
        runner.execute_model(a_scheduler_output())


def test_sampling_with_nothing_pending_is_not_an_error():
    """Reached whenever a step scheduled no tokens. The engine calls
    `sample_tokens` anyway and treats an empty output as normal."""
    runner = a_runner()

    assert runner.sample_tokens(None).req_ids == []


def test_the_pending_step_is_cleared_so_the_next_step_can_run():
    runner = a_runner()
    runner.execute_model(a_scheduler_output())
    runner.sample_tokens(None)

    assert runner.execute_model(a_scheduler_output(blocks=(2,))) is None


def test_structured_output_is_refused_rather_than_ignored():
    """A grammar bitmask has to mask the logits before sampling. Accepting one
    and not applying it would return tokens the grammar forbids, which is worse
    than declining."""
    runner = a_runner()
    runner.execute_model(a_scheduler_output())

    with pytest.raises(NotImplementedError, match="structured output"):
        runner.sample_tokens(object())


def test_the_worker_delegates_sampling_rather_than_raising():
    """The original worker raised here, on the theory that the engine would
    never call it. It does."""
    from jax_vllm_plugin.worker import JaxWorker

    runner = a_runner()
    worker = JaxWorker.__new__(JaxWorker)
    worker.model_runner = runner

    assert worker.execute_model(a_scheduler_output()) is None
    assert worker.sample_tokens(None).req_ids == ["r0"]
