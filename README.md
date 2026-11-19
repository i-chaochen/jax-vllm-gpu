# jax-vllm-gpu

JAX inference on GPUs, served through vLLM, built AMD-native first.

jax-vllm-gpu runs MaxText models on AMD Instinct GPUs with a paged KV cache.
KV lives in donated `jax.Array` page pools, and AITER's paged kernels, reached
through JAX-AITER's XLA FFI, append to them and attend over them. The same
runtime serves models two ways:

- through **vLLM**, as an out-of-tree platform plugin. vLLM supplies the
  OpenAI-compatible endpoint, the scheduler and the block manager; this package
  supplies the platform, the worker, the model runner and the KV pool.
- through a **native offline engine** with its own page allocator, prefix cache
  and continuous-batching driver, and no vLLM in the process.

> **Status:** early development. APIs will change.

## What works

- Llama-3.3-70B at TP=8 on 8x MI300X through vLLM's engine, with the KV pool
  sharded over KV heads.
- vLLM's chunked prefill, preemption with recompute and prefix caching, each
  driven by vLLM's own scheduler (`tools/smoke_engine.py`). Greedy output is
  the same whatever the token budget.
- The native engine: paged decode token-identical to MaxText's dense path and
  to a full forward pass, batched prefill, prefix sharing between requests, and
  a host tier that pages KV out and back.
- Pages another request wrote are scrubbed before reuse, tested against vLLM's
  block lifecycle with a negative control.

On 8x MI300X, Llama-3.3-70B at TP=8 through vLLM decodes at 34.9 ms per step
for one request and 23.9 ms for eight, 335 tokens/s, with 130-token prompts.

## How MaxText is used

MaxText supplies the model definitions, configs and checkpoint conversion.
This repository supplies what inference needs around them:

- `models/adapters/maxtext.py` binds MaxText's `gpu_paged` attention hook to
  this repository's kernels;
- `models/adapters/maxtext_vllm.py` wraps a MaxText model as the
  `MaxTextForCausalLM` the plugin serves;
- `engine/paged_engine.py` adds the paged entry points to an unmodified
  `MaxEngine`.

MaxText is pinned at ROCm/maxtext `30105a4c0`, which carries the hook. The next
pin is `maxtext-v0.2.5`, the release jax-titan uses, with the hook as one small
patch. MaxDiffusion is the second model provider and is not served yet.

## A plugin, not a fork

No file in vLLM's tree changes. vLLM finds platforms through
`importlib.metadata.entry_points(group="vllm.platform_plugins")`, so installing
this wheel is the whole integration step, and out-of-tree plugins take
precedence over vLLM's built-in ROCm platform.

| what | value |
|---|---|
| repository and distribution | `jax-vllm-gpu` |
| import package | `jax_vllm_gpu` |
| platform entry point | `jax_vllm_gpu` |
| connector entry point | `jax_vllm_gpu_kv_connector` |

The entry point names are what `VLLM_PLUGINS` allowlists, and the platform one
appears in vLLM's `Platform plugin jax_vllm_gpu is activated` log line.

## Layout

| Path | Contents |
|---|---|
| `src/jax_vllm_gpu/kv/common` | Neutral vocabulary: page table, storage layout and cache namespace. Standard library and numpy only |
| `src/jax_vllm_gpu/kv/control` | The native engine's page allocator, page map, prefix index and residency. Standard library, numpy and `kv/common` only |
| `src/jax_vllm_gpu/kv/execution` | The device side: pool allocation and sharding, step views and inputs, shape buckets, the continuous-batching driver, page staging and the host tier |
| `src/jax_vllm_gpu/ops` | Paged attention and KV append through JAX-AITER, under `shard_map` for tensor parallelism |
| `src/jax_vllm_gpu/models/adapters` | MaxText: the attention hook and the vLLM model |
| `src/jax_vllm_gpu/engine` | The native offline engine (`PagedEngine`, `PagedInferenceWorker`) and a torch-free tokenizer |
| `src/jax_vllm_gpu/plugin` | The vLLM platform plugin: platform, worker, model runner, step pipeline, KV spec and pool, page-table translation, recycled-page tracking and connector registration |
| `tests/` | CPU and GPU tests. `gpu_only` tests skip when JAX sees no GPU |
| `tools/` | Smoke runs through vLLM's real engine, and the latency benchmark |
| `benchmarks/` | Native-engine serving and prefix-cache benchmarks |
| `.cursor/skills/`, `.claude/skills/` | Agent skill (identical copies): architecture rules and test discipline |

[docs/getting_started.md](docs/getting_started.md) covers the environment,
the tests and the smoke runs.

## Environment variables

| variable | effect |
|---|---|
| `VLLM_JAX_PLATFORM` | `1` forces the plugin on, `0` forces it off. vLLM swallows exceptions from platform probes, so a probe failure otherwise looks like the wheel being absent. |
| `JAX_VLLM_ROCTX` | `1` opens one ROCTX range per serving step, from dispatch until the sampled tokens reach the host, labelled with the step's shape, for lining up steps with GPU work in rocprofv3. Fails at startup if `librocprofiler-sdk-roctx` is missing. |
| `VLLM_PLUGINS` | Allowlist of entry point names. Needed when another JAX platform plugin is installed beside this one, since vLLM refuses to activate two. |
| `VLLM_TARGET_DEVICE` | **Hazard.** `cpu` makes vLLM return its CPU platform before loading any plugin, so this one is never probed. Keep it unset. |

## Roadmap

- Move to vLLM `5ec99956ad`, tpu-inference's last-known-good. vLLM's KV-cache
  interface changed (`get_kv_cache_layer_configs`, `KVCacheLayerConfig`), and a
  preempted request now comes back as a new request.
- MaxText `maxtext-v0.2.5` with the hook as one carried patch, on jax-titan's
  container and JAX build, with JAX-AITER as a submodule.
- Compile the bucketed shape set at startup: an unseen shape costs about 90 s
  on the 70B today.
- A KV connector for prefill-decode disaggregation over MORI. It is a stub today.
- MaxDiffusion models.

## License

Apache License 2.0, see [LICENSE](LICENSE). [NOTICE](NOTICE) lists the code
derived from MaxText.
