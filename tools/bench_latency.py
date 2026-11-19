# SPDX-License-Identifier: Apache-2.0
"""TTFT, ITL and tokens/second through the real vLLM engine.

Compilation dominates any naive timing on this platform -- a new shape costs
around 90 s on a 70B model -- so vLLM's own reported throughput measures the
compiler, not the runtime. Every configuration here is therefore warmed at the
exact shapes it will be timed at, and the timed calls are the second and third
time each program runs.

**ITL is read from the runner, not derived by subtraction, and the first version
of this script got that wrong.** Timing `generate(N)` minus `generate(1)` and
dividing by N-1 assumes every step in the window costs the same. On this platform
one unseen shape costs seconds, so a single compile is smeared across the whole
window -- which produced a table where throughput *fell* from batch 4 to batch 8
while step time rose 2.5x. That cannot be true: every batch size reads the same
weights per step, so step time should be near flat. `jax_vllm_gpu.plugin.metrics`
therefore records each step where it happens and reports a median over the steps
that did not compile.

TTFT is still measured as a whole `generate(max_tokens=1)`, which needs no
subtraction: it is one prefill and nothing else.

**Batch scaling is the point.** Section 1.1's claim is that decode is
bandwidth-bound, so a larger batch amortises the per-step weight read and
throughput rises while step time stays flat. That is measured on 8B; at 70B only
the capacity ratio has been. Sweeping the batch is what tests it.

Prompts are distinct but the same length, so no two requests can share a prefix
and shape bucketing stays predictable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time


def build_prompts(count: int, words: int) -> list[str]:
  """Distinct prompts of equal length, so nothing dedups and shapes are stable."""
  return [
      f"Report {index:04d} concerning " + " ".join(["alpha beta gamma delta"] * words)
      + " . In summary,"
      for index in range(count)
  ]


def timed_generate(llm, prompts, max_tokens):
  from vllm import SamplingParams

  start = time.perf_counter()
  outputs = llm.generate(
      prompts,
      SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
      use_tqdm=False,
  )
  elapsed = time.perf_counter() - start
  produced = sum(len(o.outputs[0].token_ids) for o in outputs)
  return elapsed, produced, outputs


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True)
  parser.add_argument("--dummy", action="store_true")
  parser.add_argument("--tensor-parallel-size", type=int, default=8)
  parser.add_argument("--max-model-len", type=int, default=2048)
  parser.add_argument("--num-gpu-blocks-override", type=int, default=8192)
  parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
  parser.add_argument("--prompt-words", type=int, default=30, help="~4 tokens each")
  parser.add_argument("--gen-tokens", type=int, default=17)
  parser.add_argument(
      "--batch", type=int, action="append", default=None,
      help="repeatable; each is a separate warm-then-time run",
  )
  parser.add_argument("--maxtext", action="append", default=[], metavar="KEY=VALUE")
  args = parser.parse_args()
  batches = args.batch or [1, 4, 8]

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
  log = logging.getLogger("bench")
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

  maxtext_config = {}
  for item in args.maxtext:
    key, _, value = item.partition("=")
    try:
      maxtext_config[key] = int(value)
    except ValueError:
      maxtext_config[key] = value

  from vllm import LLM

  from jax_vllm_gpu.plugin.metrics import STEPS

  started = time.perf_counter()
  llm = LLM(
      model=args.model,
      tokenizer=args.model,
      load_format="dummy" if args.dummy else "auto",
      max_model_len=args.max_model_len,
      max_num_seqs=max(batches),
      max_num_batched_tokens=args.max_model_len,
      tensor_parallel_size=args.tensor_parallel_size,
      gpu_memory_utilization=args.gpu_memory_utilization,
      num_gpu_blocks_override=args.num_gpu_blocks_override,
      enforce_eager=True,
      additional_config={"maxtext_config": maxtext_config},
  )
  log.info("engine up in %.1f s", time.perf_counter() - started)

  prompts = build_prompts(max(batches), args.prompt_words)
  tokenizer = llm.get_tokenizer()
  prompt_tokens = len(tokenizer.encode(prompts[0]))
  log.info(
      "prompt %d tokens, generating %d, batches %s",
      prompt_tokens, args.gen_tokens, batches,
  )

  rows = []
  for batch in batches:
    subset = prompts[:batch]

    # Warm at both shapes that will be timed. The long one compiles every decode
    # bucket the sequence will pass through, so neither timed call pays a compile.
    warm_started = time.perf_counter()
    timed_generate(llm, subset, args.gen_tokens)
    timed_generate(llm, subset, 1)
    log.info("batch %d warmed in %.1f s", batch, time.perf_counter() - warm_started)

    ttft, _, _ = timed_generate(llm, subset, 1)

    # Reset after warm-up and after the TTFT run, so the recorded steps are only
    # the ones being reported. Shapes already compiled stay remembered.
    STEPS.reset()
    wall, produced, _ = timed_generate(llm, subset, args.gen_tokens)
    stats = STEPS.summary()

    rows.append({
        "batch": batch,
        "ttft_ms": ttft * 1e3,
        "itl_ms": stats.get("median_s", float("nan")) * 1e3,
        "p90_ms": stats.get("p90_s", float("nan")) * 1e3,
        "per_request_tok_s": 1.0 / stats["median_s"] if stats.get("median_s") else float("nan"),
        "total_tok_s": stats.get("tokens_per_second", float("nan")),
        "compiles": stats["compiles"],
        "compile_s": stats["compile_seconds"],
        "repeats": stats["repeats"],
        "wall_tok_s": produced / wall,
    })
    log.info(
        "batch %2d: TTFT %6.1f ms  ITL %6.1f ms (p90 %6.1f)  %6.1f tok/s total  "
        "[%d timed steps, %d compiles costing %.1f s]",
        batch, rows[-1]["ttft_ms"], rows[-1]["itl_ms"], rows[-1]["p90_ms"],
        rows[-1]["total_tok_s"], rows[-1]["repeats"], rows[-1]["compiles"],
        rows[-1]["compile_s"],
    )

  print("\n" + "=" * 92)
  print(f"Llama-style decode, TP={args.tensor_parallel_size}, prompt {prompt_tokens} tok, "
        f"{args.gen_tokens} generated, enforce_eager")
  print("median over timed steps that did not compile; compiles reported separately")
  print("=" * 92)
  print(f"{'batch':>6} {'TTFT ms':>9} {'ITL ms':>8} {'p90 ms':>8} {'tok/s/req':>10}"
        f" {'tok/s tot':>10} {'steps':>6} {'compiles':>9} {'compile s':>10}")
  for row in rows:
    print(f"{row['batch']:>6} {row['ttft_ms']:>9.1f} {row['itl_ms']:>8.1f} "
          f"{row['p90_ms']:>8.1f} {row['per_request_tok_s']:>10.1f} "
          f"{row['total_tok_s']:>10.1f} {row['repeats']:>6} {row['compiles']:>9} "
          f"{row['compile_s']:>10.1f}")

  if len(rows) > 1:
    first, last = rows[0], rows[-1]
    step_ratio = last["itl_ms"] / first["itl_ms"]
    thru_ratio = last["total_tok_s"] / first["total_tok_s"]
    print(
        f"\nbatch {first['batch']} -> {last['batch']}: "
        f"step time {step_ratio:.2f}x, throughput {thru_ratio:.2f}x"
    )
    batch_ratio = last["batch"] / first["batch"]
    if step_ratio < 1.5 and thru_ratio > batch_ratio * 0.7:
      print(
          "Near-flat step time with throughput rising close to the batch ratio: the\n"
          "bandwidth-bound signature Section 1.1 depends on. A larger batch amortises\n"
          "the per-step weight read, so KV capacity converts to throughput."
      )
    else:
      print(
          "Step time rising with batch, or throughput well short of the batch ratio.\n"
          "That is not the bandwidth-bound signature, so the capacity-to-throughput\n"
          "argument does not hold at this size and configuration -- check the compile\n"
          "columns before concluding anything about the runtime."
      )
  return 0


if __name__ == "__main__":
  sys.exit(main())
