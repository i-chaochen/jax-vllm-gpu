# SPDX-License-Identifier: Apache-2.0
"""Drive the worker's lifecycle by hand, without vLLM's engine.

The engine brings a scheduler, a tokenizer, output processing and an async
front end, and every one of those can fail for reasons that have nothing to do
with this plugin. Calling the worker directly in the order the executor would
keeps a failure attributable.

    python3 tools/smoke_worker.py --model /data/cchen104/tiny_llama_hf --dummy
"""

from __future__ import annotations

import argparse
import logging
import sys
import time


def build_config(args):
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.model,
        load_format="dummy" if args.dummy else "auto",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        additional_config={"maxtext_config": dict(args.maxtext)},
    )
    return engine_args.create_engine_config()


def scheduler_output(prompts, block_size, next_block):
    """Hand-build the `SchedulerOutput` a first prefill step would carry."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

    new_reqs = []
    scheduled = {}
    for index, tokens in enumerate(prompts):
        request_id = f"req-{index}"
        blocks = [next_block() for _ in range(-(-len(tokens) // block_size))]
        new_reqs.append(
            NewRequestData(
                req_id=request_id,
                prompt_token_ids=list(tokens),
                mm_features=[],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
                pooling_params=None,
                block_ids=(blocks,),
                num_computed_tokens=0,
                lora_request=None,
                prefill_token_ids=list(tokens),
            )
        )
        scheduled[request_id] = len(tokens)

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def decode_output(previous, sampled, block_size, next_block, lengths):
    """The follow-up step: one token per request, blocks grown as needed."""
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    req_ids = list(previous)
    new_block_ids = []
    for request_id in req_ids:
        length = lengths[request_id]
        # A block is appended exactly when the new token opens a fresh page.
        new_block_ids.append(([next_block()],) if length % block_size == 0 else None)

    cached = CachedRequestData(
        req_ids=req_ids,
        new_token_ids=[[] for _ in req_ids],
        new_block_ids=new_block_ids,
        num_computed_tokens=[lengths[r] for r in req_ids],
        num_output_tokens=[1 for _ in req_ids],
    )
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={r: 1 for r in req_ids},
        total_num_scheduled_tokens=len(req_ids),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dummy", action="store_true", help="random weights")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument(
        "--kv-blocks",
        type=int,
        default=4096,
        help=(
            "cap on the negotiated block count. A toy model's pages are tiny, so "
            "the honest memory budget buys millions of them and allocating that "
            "many wastes the whole point of a fast smoke loop."
        ),
    )
    parser.add_argument("--steps", type=int, default=4, help="decode steps")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--maxtext",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="passed through to additional_config['maxtext_config']",
    )
    args = parser.parse_args()
    args.maxtext = [_split(item) for item in args.maxtext]

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("smoke")

    from transformers import AutoTokenizer
    from vllm.platforms import current_platform

    log.info("Platform resolved to %s.", type(current_platform).__name__)

    config = build_config(args)
    from jax_vllm_gpu.plugin.worker import JaxWorker

    worker = JaxWorker(
        vllm_config=config,
        local_rank=0,
        rank=0,
        distributed_init_method="env://",
        is_driver_worker=True,
    )

    stage = _timer(log)
    worker.init_device()
    stage("init_device")
    worker.load_model()
    stage("load_model")

    layer_configs = worker.get_kv_cache_layer_configs()
    log.info(
        "KV cache layers: %d, first=%s",
        len(layer_configs),
        next(iter(layer_configs.values())).spec,
    )
    available = worker.determine_available_memory()
    stage("determine_available_memory")

    kv_cache_config = _kv_cache_config(
        config, worker, layer_configs, available, args.kv_blocks
    )
    log.info("vLLM sized the pool at %d blocks.", kv_cache_config.num_blocks)
    worker.initialize_from_config(kv_cache_config)
    stage("initialize_from_config")
    worker.compile_or_warm_up_model()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tokenizer.encode(args.prompt)
    log.info("Prompt %r -> %d tokens.", args.prompt, len(prompt_ids))

    block_size = config.cache_config.block_size
    counter = iter(range(1, kv_cache_config.num_blocks))
    next_block = lambda: next(counter)  # noqa: E731

    output = _step(worker, scheduler_output([prompt_ids], block_size, next_block))
    stage("prefill")
    log.info("Prefill sampled: %s", output.sampled_token_ids)

    generated = [t for row in output.sampled_token_ids for t in row]
    # num_computed_tokens counts tokens whose KV is in the pool. The token just
    # sampled has none yet -- writing it is what the next step does -- so this
    # is the prompt length, not the prompt length plus one.
    lengths = {"req-0": len(prompt_ids)}
    for step in range(args.steps):
        out = _step(
            worker, decode_output(["req-0"], generated, block_size, next_block, lengths)
        )
        stage(f"decode {step}")
        token = out.sampled_token_ids[0]
        if not token:
            log.warning("Step %d produced no token.", step)
            break
        generated.extend(token)
        lengths["req-0"] += 1

    log.info("Generated token ids: %s", generated)
    log.info("Generated text: %r", tokenizer.decode(generated))
    return 0


def _step(worker, scheduler_output):
    """One step in the order the engine uses.

    `execute_model` then `sample_tokens`, because that is what both of vLLM's
    step paths do and the tokens only come out of the second. Fabricating the
    scheduler's messages is already a compromise this harness makes; fabricating
    the *call sequence* as well is how it came to encode a worker design that no
    engine path actually supports.
    """
    output = worker.execute_model(scheduler_output)
    return worker.sample_tokens(None) if output is None else output


def _kv_cache_config(config, worker, layer_configs, available_bytes, block_cap):
    """Plan the KV cache the way the engine core does, capped at `block_cap`.

    The cap goes in as `num_gpu_blocks_override`, which vLLM's planner honours,
    so the plan and the block count the worker receives agree.
    """
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_planning import (
        get_global_kv_cache_groups,
        get_kv_cache_configs,
    )

    per_block = sum(layer.spec.page_size_bytes for layer in layer_configs.values())
    if available_bytes < per_block:
        raise ValueError(
            f"{available_bytes} bytes does not hold one block of {per_block}"
        )
    config.cache_config.num_gpu_blocks_override = min(
        available_bytes // per_block, block_cap
    )

    layout = resolve_kv_cache_layout(
        config, [worker.get_supported_kv_cache_layouts()], layer_configs.values()
    )
    groups = get_global_kv_cache_groups(config, [layer_configs])
    worker.set_kv_cache_layout(
        layout.name, config.parallel_config.cp_kv_cache_interleave_size
    )
    (kv_cache_config,) = get_kv_cache_configs(
        config, [layer_configs], [available_bytes], groups
    )
    return kv_cache_config


def _split(item: str):
    key, _, value = item.partition("=")
    for cast in (int, float):
        try:
            return key, cast(value)
        except ValueError:
            pass
    if value in ("true", "false"):
        return key, value == "true"
    return key, value


def _timer(log):
    last = [time.time()]

    def mark(label: str) -> None:
        now = time.time()
        log.info("[%s] %.1fs", label, now - last[0])
        last[0] = now

    return mark


if __name__ == "__main__":
    sys.exit(main())
