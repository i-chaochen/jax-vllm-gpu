---
name: jax_vllm_gpu_coding
description: >-
  Architecture rules, coding standards and test discipline for jax-vllm-gpu,
  JAX inference on AMD GPUs with a paged KV cache, served through a vLLM
  platform plugin or a native offline engine, with MaxText as the model
  provider. Use when writing, reviewing or changing code in the jax-vllm-gpu
  repo: the KV runtime, the paged-attention kernels, the MaxText adapters, the
  engine, the vLLM plugin, tests and docs.
---

# jax-vllm-gpu Coding Guide

## Core Principle

jax-vllm-gpu is built AMD-native first: ROCm, AITER kernels through
JAX-AITER's XLA FFI, and MORI for transport later. MaxText is a pinned model
provider: this repository never edits MaxText's source, and anything MaxText
must carry lives as a patch that is also proposed upstream. vLLM is never
patched either: the plugin is found through the `vllm.platform_plugins` and
`vllm.general_plugins` entry points.

## Architecture Rules

- **Layers:** `kv/common` imports only the standard library and numpy;
  `kv/control` may also import `kv/common`; `kv/execution` holds the device
  side and is the lowest layer that may import jax or a kernel provider.
  `tests/test_kv_import_rule.py` enforces this. Never import jax, jax_aiter,
  maxtext or vllm below `kv/execution`.
- **One page owner per mode:** the native engine owns pages through
  `kv/control`'s allocator and prefix index. Under the vLLM plugin, vLLM's
  `KVCacheManager` owns them, and `plugin/` never imports `kv/control`
  (`tests/test_ownership_mode.py`).
- **Attention hook:** MaxText dispatches `attention="gpu_paged"` to
  `Attention.forward_serve_gpu_paged`. `models/adapters/maxtext.install()`
  binds it to this repository's implementation and raises if MaxText has no
  such hook. Bind before building a model; `PagedEngine` and
  `plugin.registry.register_maxtext` do.
- **Scale:** pass `scale=1.0` to the kernels. MaxText folds 1/sqrt(head_dim)
  into the query projection, so the kernel's default scales twice.
- **Donation:** the pool is donated into every step and returned aliased.
  Always rebind the returned arrays. "Some donated buffers were not usable"
  naming the pool means a pool-sized copy on every step: treat it as a bug.
- **Padding:** `slot_mapping` is -1 beyond the live tokens, and `append_kv`
  skips -1. Never build a plan from padded vLLM block tables, which resolve
  padding to real page slots. Page 0 is the padding sentinel.
- **Recycled pages:** a page another request wrote is scrubbed before it
  enters a request's write region. A prefix-cache hit is not recycling and is
  not scrubbed.
- **Shapes:** every step shape goes through `StepShapePlanner`'s buckets. An
  unseen shape costs about 90 s to compile on the 70B. Build step inputs in
  numpy, not jnp, so a prompt length never becomes a jaxpr literal.
- **Positions:** always absolute. After a prefix-cache hit or a resume, the
  query starts at the cached length; RoPE is not translation invariant and a
  wrong start gives plausible text, not an error.
- **Logits:** under `gpu_paged` MaxText returns logits, not hidden states.
  Sample from them and never apply the output head again. Gather one row per
  request at its (row, position): prefill packs several requests in one row.

## vLLM Contract

- `execute_model` returns None and `sample_tokens` returns the
  `ModelRunnerOutput`; vLLM's two engine paths accept only this split.
- One worker drives every device with JAX SPMD (`distributed_executor_backend`
  "uni", a gloo world of one). torch is CPU-only.
- `KVConnectorFactory.register_connector` raises on a duplicate name and the
  general-plugins hook can load more than once, so registration swallows the
  duplicate.

## Coding Standards

- Match the file you are in: 2-space indentation in `kv/`, `ops/`, `engine/`
  and `models/` (MaxText style), 4-space in `plugin/` and its tests.
- Comments state constraints the code cannot show. Commit messages carry the
  history.
- Keep license headers. Code derived from MaxText keeps Google's header with
  an AMD modifications line, and `NOTICE` lists the file.

## Tests

- Mark every test that needs a GPU `@pytest.mark.gpu_only`; `tests/conftest.py`
  skips those when JAX sees no GPU and never skips unmarked tests. Read the
  pass and skip counts, not just the exit status.
- Run one GPU test or smoke invocation at a time, always under `timeout`. A
  hung `docker exec` blocks every later command in the same shell.
- The dense-reference parity test needs exactly one visible device
  (`HIP_VISIBLE_DEVICES=0`). `MaxEngine` needs JetStream or
  `DECOUPLE_GCLOUD=TRUE`.
- With an older plugin installed beside this one, set
  `VLLM_PLUGINS=jax_vllm_gpu,jax_vllm_gpu_kv_connector`, or vLLM refuses two
  platform plugins.
- Check the plugin against vLLM's real engine (`tools/smoke_engine.py`), not
  only against fabricated `SchedulerOutput`. Each mode must reach its case.
  Greedy output must not depend on scheduling: diff `--dump-tokens` across
  token budgets.
- Hold a physical expectation before reading a benchmark: at a fixed weight
  read per step, throughput cannot fall as the batch grows. A number that
  breaks it indicts the harness first.

## Environment

- Never let pip replace jax or jaxlib, which are self-built ROCm wheels.
  Install anything new under a constraints file that pins them.
- torch must be the CPU build. A ROCm torch adds a second HIP runtime, which
  aborts RCCL clique setup above one device.
- `VLLM_TARGET_DEVICE=cpu` makes vLLM return its CPU platform before loading
  any plugin. Keep it unset.
