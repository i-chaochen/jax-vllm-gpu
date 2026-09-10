# SPDX-License-Identifier: Apache-2.0
"""Drive the worker from vLLM's real engine, not a hand-built SchedulerOutput.

`smoke_worker.py` calls the worker in the executor's order but fabricates the
scheduler's messages, so it can only ever confirm that the worker agrees with
*this repository's* reading of what vLLM sends. Every assumption about the
scheduler's vocabulary — when `num_computed_tokens` advances, whether a resumed
request's blocks replace or extend, which block ids can be recycled — is encoded
identically in the harness and in the code under test, so a shared misreading
passes green.

This harness removes that by putting vLLM's own scheduler, block manager and
prefix cache in the loop. The three modes exist because the default one never
reaches the interesting paths:

    --mode plain     one short prompt, the easy case
    --mode chunked   a prompt longer than max_num_batched_tokens, so a request
                     spans several steps and must emit no token until done
    --mode preempt   more concurrent context than the pool holds, so vLLM must
                     evict and later resume, replacing a request's block table
    --mode prefix    a shared prompt prefix, so blocks arrive via a cache hit
                     rather than a fresh allocation and must not be scrubbed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def maxtext_overrides(pairs):
    config = {}
    for item in pairs:
        key, _, value = item.partition("=")
        for cast in (int, float):
            try:
                config[key] = cast(value)
                break
            except ValueError:
                continue
        else:
            config[key] = value == "true" if value in ("true", "false") else value
    return config


PROMPTS = {
    "plain": ["The capital of France is"],
    # Long enough to exceed a deliberately small token budget, so vLLM has to
    # split it across steps.
    "chunked": [" ".join(["the quick brown fox jumps over the lazy dog"] * 40)],
    # Short prompts, so vLLM admits them all, paired with a long generation so
    # they then grow into contention. Long prompts would merely queue, which
    # looks similar from the outside and never exercises preemption at all.
    "preempt": [f"Story {i} begins:" for i in range(6)],
    # A shared opening, so the second and later requests hit the prefix cache
    # and receive blocks another request wrote -- deliberately, not by recycling.
    "prefix": [
        "You are a helpful assistant. Answer concisely and accurately. Question: "
        + tail
        for tail in ("what is 2+2?", "name a colour.", "what is the sky?")
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--mode", choices=sorted(PROMPTS), default="plain")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--dump-tokens",
        metavar="PATH",
        help=(
            "write the generated token ids as JSON. Two runs of the same prompt "
            "under different scheduling must agree exactly under greedy "
            "sampling -- chunking a prefill or preempting a request changes when "
            "KV is written, never what it is -- so diffing two dumps is the "
            "sharpest correctness check this harness offers. Written to a file "
            "rather than compared in process because a second engine in the "
            "same process would contend for the same devices."
        ),
    )
    parser.add_argument("--maxtext", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("engine")

    # The scheduler has to be in this process to be observable, and being able
    # to say what a run exercised is the entire point of this harness.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams

    prompts = PROMPTS[args.mode]
    log.info(
        "Mode %s: %d prompt(s), longest %d chars, token budget %d, pool override %s.",
        args.mode,
        len(prompts),
        max(len(p) for p in prompts),
        args.max_num_batched_tokens,
        args.num_gpu_blocks_override,
    )

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        load_format="dummy" if args.dummy else "auto",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=True,
        additional_config={"maxtext_config": maxtext_overrides(args.maxtext)},
    )

    observable = observe_scheduler(llm)
    if not observable:
        log.warning(
            "The scheduler is out of process, so this run cannot report what it "
            "exercised. Set VLLM_ENABLE_V1_MULTIPROCESSING=0."
        )

    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )

    log.info(
        "Scheduler did %d step(s): %d resumed-after-preemption, %d chunked-prefill, "
        "%d prefix-cache-hit.",
        Observed.steps,
        Observed.resumed,
        Observed.chunked,
        Observed.cache_hits,
    )

    for index, output in enumerate(outputs):
        completion = output.outputs[0]
        log.info(
            "[%d] prompt %d tok -> %d tok (%s): %r",
            index,
            len(output.prompt_token_ids),
            len(completion.token_ids),
            completion.finish_reason,
            completion.text[:160],
        )

    if args.dump_tokens:
        with open(args.dump_tokens, "w") as handle:
            json.dump([list(o.outputs[0].token_ids) for o in outputs], handle)
        log.info("Wrote token ids to %s.", args.dump_tokens)

    short = [o for o in outputs if not o.outputs[0].token_ids]
    if short:
        log.error("%d request(s) produced no tokens at all.", len(short))
        return 1

    # Each mode has to prove it reached its case. Without this a pool that
    # turned out to be large enough, or a budget that turned out to fit the
    # prompt, looks exactly like the hard path handled correctly.
    required = {"preempt": ("resumed", Observed.resumed),
                "chunked": ("chunked-prefill", Observed.chunked),
                "prefix": ("prefix-cache-hit", Observed.cache_hits)}
    if observable and args.mode in required:
        name, count = required[args.mode]
        if not count:
            log.error(
                "No %s steps occurred, so this run proves nothing. Lower "
                "--num-gpu-blocks-override / --max-num-batched-tokens, or raise "
                "--max-tokens.",
                name,
            )
            return 1

    log.info("OK: %d/%d requests completed through the real scheduler.",
             len(outputs), len(prompts))
    return 0


class Observed:
    """What the real scheduler actually did, counted from its own output.

    Watching log text was the first attempt and it was unsound twice over: the
    scheduler emits no preemption message, and with multiprocessing on it runs
    in another process where this handler never sees anything. Reading
    `SchedulerOutput` directly also observes the thing that matters rather than
    a proxy for it -- `resumed_req_ids` is precisely the field whose meaning
    `BatchState` had to assume.
    """

    resumed = 0
    chunked = 0
    cache_hits = 0
    steps = 0


def observe_scheduler(llm) -> bool:
    """Wrap the in-process scheduler so each `SchedulerOutput` is counted.

    Returns False when the engine is out of process, where there is nothing to
    wrap -- the caller then cannot make claims about what was exercised.
    """
    engine_core = getattr(llm.llm_engine.engine_core, "engine_core", None)
    scheduler = getattr(engine_core, "scheduler", None)
    if scheduler is None:
        return False

    original = scheduler.schedule

    def schedule(*args, **kwargs):
        output = original(*args, **kwargs)
        Observed.steps += 1
        cached = getattr(output, "scheduled_cached_reqs", None)
        Observed.resumed += len(getattr(cached, "resumed_req_ids", ()) or ())

        # These two look identical if you only compare scheduled tokens against
        # prompt length -- both schedule fewer -- and they exercise opposite
        # halves of the reuse tracker, so they are counted apart. A cache hit
        # arrives with tokens *already computed*; a chunk is simply cut short.
        for new in getattr(output, "scheduled_new_reqs", ()) or ():
            prompt_len = len(new.prompt_token_ids or ())
            computed = new.num_computed_tokens
            scheduled = output.num_scheduled_tokens.get(new.req_id, 0)
            if computed:
                Observed.cache_hits += 1
            if computed + scheduled < prompt_len:
                Observed.chunked += 1
        return output

    scheduler.schedule = schedule
    return True


if __name__ == "__main__":
    sys.exit(main())
