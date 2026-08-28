# jax-vllm-plugin

A vLLM **platform plugin** and **KV connector** for running JAX models on AMD
GPUs, backed by the paged KV runtime in MaxText and the paged attention kernels
in jax-aiter.

This is milestone M10a of the AMD paged KV runtime plan, and it delivers the
`maxtext-vllm-inference` image's serving half: vLLM supplies the OpenAI-compatible
endpoint, the scheduler and the block manager, while this plugin supplies the
platform, the worker, the model runner and the attention backend.

## It is a plugin, not a fork

No file in vLLM's tree changes, and there is no pull request against
`vllm-project/vllm`. vLLM discovers platforms through
`importlib.metadata.entry_points(group="vllm.platform_plugins")`, which reads the
metadata of whatever distributions are installed, so declaring the entry point
here and `pip install`ing the wheel is the entire integration step.

Out-of-tree plugins are also *preferred over* the built-ins, so on a ROCm host
this takes precedence over `vllm.platforms.rocm.RocmPlatform` with no patch and
no environment variable.

## Why this name

The discriminator is **JAX**, not the hardware. vLLM already supports ROCm — it
has a built-in `RocmPlatform` for PyTorch — so what is new here is JAX on AMD,
not AMD. A `vllm-rocm`-style name would be ambiguous with that existing support,
and every `vllm-*` backend name (`vllm-ascend`, `vllm-gaudi`, `vllm-spyre`,
`vllm-tpu`) belongs to the vllm-project organisation, so the prefix would imply
an ownership this does not have.

`-plugin` rather than `-inference` is deliberate too: in the surrounding project
the `-inference` suffix means a release *image* (`maxtext-vllm-inference`,
`maxtext-native-inference`), and this is a wheel that goes into one.

Names derived from the repository, which are not all the same string:

| what | value |
|---|---|
| repository and distribution | `jax-vllm-plugin` |
| import package | `jax_vllm_plugin` |
| platform entry point name | `jax_vllm_plugin` |
| connector entry point name | `jax_vllm_plugin_kv_connector` |

The entry point *names* are what `VLLM_PLUGINS` allowlists, and the platform one
appears in vLLM's `Platform plugin jax_vllm_plugin is activated` log line.

## Scope: one repository per plug socket, not per framework

This repo is vLLM-only, and that is a deliberate rule rather than a current
state. vLLM offers an entry-point socket, so a wheel is the right artifact and
this is its home. Other serving frameworks do not necessarily offer one — for
example sglang-jax declares no entry points and selects backends through
hardcoded conditionals, so its integration would be commits in *their* tree,
which no repository of ours can host.

A second frontend joins this repo only if it has a real plugin mechanism **and**
shares substantial code with the vLLM plugin. Both halves are required; the
second is what makes co-location pay rather than merely look tidy.

## What lives here, and what deliberately does not

**Here:** the `Platform` implementation, the worker and JAX model runner, the
attention backend selection, physical KV pool allocation, bucketing and
precompile, and the KV connector for disaggregation and host offload.

**Not here:**

- **The MaxText model plugin.** `MaxTextForCausalLM` stays in MaxText, because it
  is built from MaxText internals and has to track model changes; relocating it
  would fork it from the models it wraps. This repository's job is to *provide
  the registry it registers into* — the role `tpu_inference` plays for TPU today.
- **The scheduler, block manager and prefix cache.** vLLM owns all three. MaxText's
  `PagedDriver` and `kv_control/`'s allocator are bypassed in this mode, while its
  layout descriptor, page-table translation and bucketing stay in use.
- **The release image.** `maxtext-vllm-inference` is assembled by
  `unified-framework-builds` from four wheels — `jax-rocm-pjrt`, `jax-aiter`,
  `maxtext`, and this one. UFB builds SHA-pinned source; it is not a source home.

## Installation

Dependencies are deliberately **not** declared in `pyproject.toml`. vLLM, JAX and
jax-aiter all carry artifacts pinned to a ROCm version and a gfx target, so
letting pip resolve them invites a mismatched (typically CUDA) build that fails
at load or at first kernel launch. The runtime image supplies a matched set, and
this wheel is pure Python installed on top.

```bash
pip install -e .
pip install -e '.[test]'
pytest
```

## Status

Early scaffold, and **nothing in it has been executed yet.** Discovery is written
and covered by tests; execution is not. `JaxWorker` and `get_attn_backend_cls`
raise `NotImplementedError` on purpose, so the plugin is discoverable now and
honest about being unfinished. Note that `test_the_entry_points_are_declared_in_installed_metadata`
will fail until `pip install -e .` has run, by design — `entry_points` only sees
the plugin once the distribution is installed.

Two decisions are recorded in code comments rather than settled:

- **`device_type = "cuda"`.** Torch reuses the CUDA device type for ROCm rather
  than adding one, so anything in vLLM constructing `torch.device(device_type)`
  needs `"cuda"` here even though this platform does no torch compute. Revisit if
  a torch-free image makes that construction fail rather than merely go unused.
- **How the connector reaches MORI.** vLLM already ships a MoRI-IO connector at
  `vllm/distributed/kv_transfer/kv_connector/v1/moriio/`, but it is thoroughly
  torch-bound (`register_kv_caches(dict[str, torch.Tensor])`, then `.data_ptr()`
  and `.element_size()`), so it cannot be handed XLA-owned pool arrays. It is a
  protocol reference, not a dependency. Whether our connector goes through the
  host-side `mori.io` API as it does, or through the XLA FFI transfer op over
  MORI's device-side one-sided puts, is still open.

## Environment variables

| variable | effect |
|---|---|
| `VLLM_JAX_PLATFORM` | `1` forces activation, `0` forces the plugin off. Useful because vLLM swallows exceptions from platform probes, making a probe failure indistinguishable from the wheel being absent. |
| `VLLM_TARGET_DEVICE` | **Hazard.** Setting it to `cpu` makes vLLM return the CPU platform *before loading any plugin*, so this plugin is never probed and the failure looks like a missing install. Keep it unset. |
