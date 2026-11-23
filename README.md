# vLLM-XLA-GPU

JAX inference on GPUs, served through vLLM, built AMD-native first.

vLLM-XLA-GPU runs MaxText models on AMD Instinct GPUs with a paged KV cache.
KV lives in donated `jax.Array` page pools, and AITER's paged kernels, reached
through JAX-AITER's XLA FFI, append to them and attend over them. The same
runtime serves models two ways:

- through **vLLM**, as an out-of-tree platform plugin. vLLM supplies the
  OpenAI-compatible endpoint, the scheduler and the block manager; this package
  supplies the platform, the worker, the model runner and the KV pool.
- through a **standalone offline engine** with its own page allocator, prefix
  cache and continuous-batching driver, and no vLLM in the process.

> **Status:** early development. APIs will change.

## What works

- Llama-3.3-70B at TP=8 on 8x MI325X through vLLM's engine, with the KV pool
  sharded over KV heads.
- vLLM's chunked prefill, preemption with recompute and prefix caching, each
  driven by vLLM's own scheduler (`tools/smoke_engine.py`). Greedy output is
  the same whatever the token budget.
- The standalone engine: every paged-decode token is the argmax of a full
  forward pass, as the dense path's are, with near-ties judged at bf16
  resolution. Batched prefill, prefix sharing that reproduces a cold run token
  for token, and a host tier that pages KV out and back.
- Pages another request wrote are scrubbed before reuse, tested against vLLM's
  block lifecycle with a negative control.

On 8x MI325X, Llama-3.3-70B at TP=8 through vLLM decodes at 34.8 ms per step
for one request and 25.2 ms for eight, 318 tokens/s, with 130-token prompts.

## How MaxText is used

MaxText supplies the model definitions, configs and checkpoint conversion.
This repository supplies what inference needs around them:

- `models/adapters/maxtext.py` binds MaxText's `gpu_paged` attention hook to
  this repository's kernels;
- `models/adapters/maxtext_vllm.py` wraps a MaxText model as the
  `MaxTextForCausalLM` the plugin serves;
- `engine/paged_engine.py` adds the paged entry points to an unmodified
  `MaxEngine`, sized by `engine/config.PagedConfig`.

MaxText is pinned at `maxtext-v0.2.5`, the release jax-titan uses, with the
hook as one 35-line patch, `third_party/patches/maxtext-gpu-paged-hook.patch`:
the `gpu_paged` dispatch, the skipped dense-cache init and the returned pools.
It holds no vendor code, and is to be proposed upstream as an external
paged-attention provider. The pool's settings are this repository's, not
MaxText's. MaxDiffusion is the second model provider, at `821afc36` as in
jax-titan, and is not served yet.

## A plugin, not a fork

No file in vLLM's tree changes. vLLM finds platforms through
`importlib.metadata.entry_points(group="vllm.platform_plugins")`, so installing
this wheel is the whole integration step, and out-of-tree plugins take
precedence over vLLM's built-in ROCm platform. The supported vLLM commit is
tpu-inference's last-known-good, recorded in `vllm_lkg.version`.

| what | value |
|---|---|
| repository and distribution | `vllm-xla-gpu` |
| import package | `vllm_xla_gpu` |
| platform entry point | `vllm_xla_gpu` |
| connector entry point | `vllm_xla_gpu_kv_connector` |

The entry point names are what `VLLM_PLUGINS` allowlists, and the platform one
appears in vLLM's `Platform plugin vllm_xla_gpu is activated` log line.

## Layout

| Path | Contents |
|---|---|
| `src/vllm_xla_gpu/kv/common` | Neutral vocabulary: page table, storage layout and cache namespace. Standard library and numpy only |
| `src/vllm_xla_gpu/kv/control` | The standalone engine's page allocator, page map, prefix index and residency. Standard library, numpy and `kv/common` only |
| `src/vllm_xla_gpu/kv/execution` | The device side: pool allocation and sharding, step views and page staging |
| `src/vllm_xla_gpu/host` | What a step needs before it reaches a device: the ladder of step shapes, and the tokens, positions and sampling indices. Standard library, numpy and `kv/common` only |
| `src/vllm_xla_gpu/ops` | Paged attention and KV append through JAX-AITER, under `shard_map` for tensor parallelism |
| `src/vllm_xla_gpu/models/adapters` | MaxText: the attention hook and the vLLM model |
| `src/vllm_xla_gpu/engine` | The standalone offline engine (`PagedEngine`, `PagedInferenceWorker`), its continuous-batching driver, the runtime that wires it to the control plane, the host tier, its pool settings (`PagedConfig`) and a torch-free tokenizer |
| `src/vllm_xla_gpu/plugin` | The vLLM platform plugin: platform, worker, model runner, step pipeline, KV spec and pool, page-table translation, recycled-page tracking and connector registration |
| `tests/` | CPU and GPU tests. `gpu_only` tests skip when JAX sees no GPU |
| `tools/` | Smoke runs through vLLM's real engine and worker |
| `benchmarks/` | `vllm_latency.py`: TTFT and per-step latency through vLLM's engine. `standalone_*.py`: the standalone engine's serving and prefix-cache benchmarks, with their harness |
| `third_party/` | JAX-AITER as a submodule at jax-titan's pin, and the patches carried for MaxText and JAX-AITER |
| `scripts/` | The container, the source trees, the Python environment, the JAX-AITER build and `env.sh` |
| `vllm_lkg.version` | The vLLM commit |
| `.github/` | CI, with the pinned wheels and package versions it installs |
| `.cursor/skills/`, `.claude/skills/` | Agent skill (identical copies): architecture rules and test discipline |

[docs/getting_started.md](docs/getting_started.md) covers the environment,
the tests and the smoke runs.

## Environment variables

| variable | effect |
|---|---|
| `VXG_PLATFORM` | `1` forces the plugin on, `0` forces it off. vLLM swallows exceptions from platform probes, so a probe failure otherwise looks like the wheel being absent. |
| `VXG_ROCTX` | `1` opens one ROCTX range per serving step, from dispatch until the sampled tokens reach the host, labelled with the step's shape, for lining up steps with GPU work in rocprofv3. Fails at startup if `librocprofiler-sdk-roctx` is missing. |
| `VLLM_PLUGINS` | Allowlist of entry point names. Needed when another JAX platform plugin is installed beside this one, since vLLM refuses to activate two. |
| `VLLM_TARGET_DEVICE` | **Hazard.** `cpu` makes vLLM return its CPU platform before loading any plugin, so this one is never probed. Keep it unset. |

This project's own variables start with `VXG_`, not `VLLM_`: vLLM warns about
every `VLLM_*` variable it does not define, and rejects them when
`fail_on_environ_validation` is set. The scripts use `VXG_ROOT` for the
directory that holds the source trees and the venv, and the tests use
`VXG_REQUIRE_GPU`.

## Roadmap

- A native XLA executor: the step functions the JAX runner compiles today,
  exported once and run by a C++ runtime that keeps the KV pool on the GPU,
  with no JAX in the serving worker. The project is named for it.
- Propose the hook to MaxText as a vendor-neutral external paged-attention
  provider, and drop the patch once a release carries it. Send the KV-append
  fix to JAX-AITER, and report the embedding all-gather to XLA.
- Compile the bucketed shape set at startup: an unseen shape costs about 90 s
  on the 70B today.
- A KV connector for prefill-decode disaggregation over MORI. It is a stub today.
- MaxDiffusion models.

## License

Apache License 2.0, see [LICENSE](LICENSE). [NOTICE](NOTICE) lists the code
derived from MaxText.
