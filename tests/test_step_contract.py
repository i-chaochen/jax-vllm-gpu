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

from vllm_xla_gpu.host.shapes import StepShapePlanner
from vllm_xla_gpu.kv.common import KvStorageLayoutV1
from vllm_xla_gpu.kv.execution.pool_factory import allocate_pool

from vllm_xla_gpu.plugin import pool as pool_module
from vllm_xla_gpu.plugin.kv_reuse import PageReuseTracker
from vllm_xla_gpu.plugin.model_runner import JaxModelRunner

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


def a_scheduler_output(
    prompt_len=4, blocks=(1,), req_id="r0", *, prefill=None, scheduled=None, preempted=None
):
    """The minimum a first prefill step carries, duck-typed.

    `prefill` defaults to the prompt; a resumed request also replays what it had
    generated. `scheduled` defaults to all of it in one step.
    """
    prompt = list(range(prompt_len))
    prefill = prompt if prefill is None else list(prefill)
    scheduled = len(prefill) if scheduled is None else scheduled
    request = SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=prompt,
        prefill_token_ids=prefill,
        sampling_params=SimpleNamespace(temperature=0.0, top_p=1.0, top_k=0),
        block_ids=(list(blocks),),
        num_computed_tokens=0,
    )
    return SimpleNamespace(
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=None,
        finished_req_ids=(),
        preempted_req_ids=preempted,
        num_scheduled_tokens={req_id: scheduled},
        total_num_scheduled_tokens=scheduled,
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

    assert runner.execute_model(a_scheduler_output(blocks=(2,), req_id="r1")) is None


def test_a_preempted_request_comes_back_as_new_with_fresh_pages():
    """vLLM resumes a preempted request by sending it again as a new one, its
    generated tokens appended to the prompt. Its old pages are gone, and the
    first token emitted is the one after the replay."""
    runner = a_runner()
    runner.execute_model(a_scheduler_output())
    (first,) = runner.sample_tokens(None).sampled_token_ids
    replay = list(range(4)) + first

    runner.execute_model(a_scheduler_output(blocks=(2,), prefill=replay, preempted={"r0"}))
    (second,) = runner.sample_tokens(None).sampled_token_ids

    assert len(second) == 1
    assert runner.batch_state.get("r0").block_ids == [2]
    assert runner.requests["r0"].tokens == replay + second


def test_a_request_resent_while_active_is_refused():
    """vLLM re-sends an active request only to continue a streaming-input
    session, which this platform does not serve."""
    runner = a_runner()
    runner.execute_model(a_scheduler_output())
    runner.sample_tokens(None)

    with pytest.raises(ValueError, match="streaming-input"):
        runner.execute_model(a_scheduler_output(blocks=(2,)))


def test_a_partial_replay_emits_nothing():
    """The replayed tokens were emitted before the preemption; a token predicted
    from part of them continues nothing."""
    runner = a_runner()
    replay = list(range(4)) + [7]

    runner.execute_model(a_scheduler_output(prefill=replay, scheduled=3))

    assert runner.sample_tokens(None).sampled_token_ids == [[]]


@pytest.mark.parametrize("field", ["new_block_ids_to_zero", "kv_cache_block_copies"])
def test_block_edits_from_the_scheduler_are_refused(field):
    """Skipping a zeroing or a copy-on-write would leave the request reading
    whatever its page last held."""
    runner = a_runner()
    output = a_scheduler_output()
    setattr(output, field, [1])

    with pytest.raises(RuntimeError, match=field):
        runner.execute_model(output)


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
    from vllm_xla_gpu.plugin.worker import JaxWorker

    runner = a_runner()
    worker = JaxWorker.__new__(JaxWorker)
    worker.model_runner = runner

    assert worker.execute_model(a_scheduler_output()) is None
    assert worker.sample_tokens(None).req_ids == ["r0"]


def test_a_roctx_range_spans_the_step_from_dispatch_to_the_sampled_tokens():
    """execute_model only enqueues the forward pass, so the range has to stay
    open until sample_tokens holds the tokens; closing it at dispatch would
    cover none of the GPU work."""
    runner = a_runner()
    events = []

    class FakeRoctx:
        def start(self, name):
            events.append(("start", name.split()[0]))
            return 7

        def stop(self, range_id):
            events.append(("stop", range_id))

    runner._roctx = FakeRoctx()
    runner.execute_model(a_scheduler_output())
    assert events == [("start", "VXG_STEP")]

    runner.sample_tokens(None)
    assert events == [("start", "VXG_STEP"), ("stop", 7)]
