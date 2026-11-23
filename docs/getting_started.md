# Getting started

vLLM-XLA-GPU runs in the TheRock ROCm container jax-titan uses, on jax-titan's
JAX build, with a CPU-only torch and vLLM built with no kernels of its own. The
source trees, the venv and the compilation cache live in `VXG_ROOT`,
which defaults to `.vllm_xla_gpu` next to this repository.

## 1. Container

On the host:

```bash
scripts/container.sh     # starts ${USER}_therock-10.1_vllm_xla_gpu from
                         # ghcr.io/rocm/jax-base-ubu24.therock-10.1, at jax-titan's CI digest
docker exec -it -u ${USER} ${USER}_therock-10.1_vllm_xla_gpu bash
```

The container mounts the repository's parent directory and `/data` at their
own paths and runs as your uid, so files keep their host owner. Everything
below runs inside it.

## 2. Sources

```bash
git submodule update --init --recursive third_party/jax-aiter   # with AITER
scripts/setup_sources.sh
```

`setup_sources.sh` reads the revisions in `third_party/pins.env`. It makes the
MaxText tree at `maxtext-v0.2.5` and applies
`third_party/patches/maxtext-gpu-paged-hook.patch`, and makes the vLLM tree at
tpu-inference's last-known-good. Each is a worktree of the sibling checkout
(`../maxtext`, `../vllm`) when there is one, and a clone otherwise.

The tested set, 2026-11-19:

| Component | Version |
|---|---|
| ROCm | TheRock 10.1, gfx942 (MI300X, MI325X) |
| jax, jaxlib | 0.12.1.dev0, jax-titan's build of jax `6b39d495e5` and XLA `d23c30982a` |
| jax-rocm10-plugin, jax-rocm10-pjrt | the same build, for gfx942 |
| MaxText | `maxtext-v0.2.5` (`493d59281`) with the hook patch |
| JAX-AITER | `third_party/jax-aiter` at `c290cdd`, jax-titan's pin, with `third_party/patches/jax-aiter-int32-slots.patch` |
| vLLM | `5ec99956ad` (`third_party/pins.env`, tpu-inference's last-known-good), built with `VLLM_TARGET_DEVICE=empty` |
| torch | 2.14.0, CPU build |
| Everything else | `.github/ci-constraints.txt` |

The constraints follow jax-titan's on the JAX and MaxText side (flax 0.12.12,
orbax-checkpoint 0.12.10) and vLLM's on the serving side: numpy 2.3.5, below
the 2.4 that vLLM's mistral-common allows on Python 3.12, and transformers
5.22.2, below the 5.23 vLLM allows.

## 3. JAX

jax and jaxlib are the wheels of jax-titan's release
[`jax-wheels-2026-11-10-6b39d49-rocm`](https://github.com/ROCm/jax-titan/releases/tag/jax-wheels-2026-11-10-6b39d49-rocm);
their CPU code is the same for every GPU. That release's ROCm plugin wheels
hold gfx950 code only, so for MI300X and MI325X the two plugin wheels are
built for gfx942 with jax-titan's own scripts, from a jax-titan checkout
(`e1edc4c` was used) whose `scripts/setup_sources.sh` names the release's
revisions:

```bash
# jax, xla and shardy trees at those revisions, with jax-titan's patches. Git only.
JAX_TITAN_ROOT=/data/$USER/.jax_titan ../jax-titan/scripts/setup_sources.sh
# About 26 minutes at BAZEL_CPUS=128; the Bazel tree takes 55 GB.
JAX_TITAN_ROOT=/data/$USER/.jax_titan AMDGPU_TARGETS=gfx942 WHEELS=jax-rocm-plugin,jax-rocm-pjrt \
  BAZEL_CPUS=128 ../jax-titan/scripts/build_jax_rocm.sh
```

Put the four wheels in one directory. `.github/ci-wheels.sha256` names them
with their checksums.

## 4. Python environment

```bash
R=../.vllm_xla_gpu
python3 -m venv --system-site-packages $R/venv
source $R/venv/bin/activate
WHEELS=<the four wheels> VLLM_SRC=$R/src/vllm MAXTEXT_SRC=$R/src/maxtext \
  scripts/install_python_env.sh
scripts/build_jax_aiter.sh    # JAX-AITER's KV shims for gfx942, with third_party/patches/jax-aiter-*
```

`--system-site-packages` exposes the container's TheRock `rocm` wheels.
`install_python_env.sh` installs the wheels, CPU-only torch (a ROCm torch adds
a second HIP runtime, which aborts RCCL clique setup above one device), vLLM,
the patched MaxText and this package, under the constraints file; CI runs the
same script. In every new shell, run `source scripts/env.sh`.

## 5. Tests

Run one GPU invocation at a time, and from the host wrap each in `timeout`: a
`docker exec` that never returns blocks every later command in the same shell.

```bash
# CPU: gpu_only tests skip. 423 passed, 48 skipped on 2026-11-19.
JAX_PLATFORMS=cpu DECOUPLE_GCLOUD=TRUE python -m pytest tests -q

# GPU: everything, including the KV kernels and the engine parity tests.
# 470 passed, 1 skipped on 8x MI325X. VXG_REQUIRE_GPU=1 fails the run if
# JAX falls back to the CPU, instead of skipping every GPU test.
DECOUPLE_GCLOUD=TRUE VXG_REQUIRE_GPU=1 python -m pytest tests -q

# The skipped one compares against MaxText's dense path, which needs exactly
# the engine's device visible. 11 passed, with the attention-layer tests.
DECOUPLE_GCLOUD=TRUE HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_decode_parity.py tests/test_attention_layer.py -q
```

`DECOUPLE_GCLOUD=TRUE` gives MaxEngine the JetStream stubs it imports. Read the
pass and skip counts: a GPU test missing its `gpu_only` mark runs on CPU, and
one that skips reports success.

The control plane and the engine's config need only numpy:

```bash
PYTHONPATH=src python -m pytest tests/test_kv_common.py tests/test_kv_control.py \
  tests/test_kv_import_rule.py tests/test_kv_prefix_cache.py tests/test_kv_residency.py \
  tests/test_engine_config.py tests/test_no_maxtext_runtime_copies.py tests/test_profiling.py \
  tests/test_agent_skills.py
```

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs that, the whole
CPU suite on the full stack, and the GPU suite on a self-hosted gfx942 runner
when the repository variable `GPU_RUNNER` names one. The GPU job downloads the
gfx942 plugin wheels from this repository's release `GPU_WHEELS`.

## 6. Smoke runs through vLLM's engine

`tools/smoke_engine.py` drives `LLM.generate` with vLLM's own scheduler in
the process and refuses to pass a mode that did not reach its case. With
`--dummy` the weights are random, so a two-layer model boots in seconds.
`/data/cchen104/tiny_llama_hf` is a Hugging Face Llama `config.json` with two
layers and head_dim 128, beside the Llama 3 tokenizer.

```bash
TINY=(--model /data/cchen104/tiny_llama_hf --dummy
      --maxtext model_name=llama3.1-8b --maxtext base_num_decoder_layers=2
      --maxtext base_emb_dim=1024 --maxtext base_mlp_dim=2048
      --maxtext base_num_query_heads=8 --maxtext base_num_kv_heads=2 --maxtext head_dim=128)

python tools/smoke_engine.py "${TINY[@]}" --mode plain
python tools/smoke_engine.py "${TINY[@]}" --mode chunked --max-num-batched-tokens 64 --max-tokens 8
python tools/smoke_engine.py "${TINY[@]}" --mode preempt \
  --max-model-len 256 --max-num-seqs 6 --num-gpu-blocks-override 24 --max-tokens 200
python tools/smoke_engine.py "${TINY[@]}" --mode prefix --enable-prefix-caching --max-tokens 12
```

Greedy output must not depend on scheduling. Run `chunked` at
`--max-num-batched-tokens 64` and `512` with `--dump-tokens`, and the two dumps
must be byte-identical. The 512 run exits 1, because at that budget the prompt
is not chunked and the mode did not reach its case. On 2026-11-19 `preempt`
preempted and resumed 7 times and `prefix` hit the cache twice.

Llama-3.3-70B at TP=8, from a MaxText checkpoint converted with MaxText's
`to_maxtext`; one converted before `maxtext-v0.2.5` loads unchanged. The run
takes about three and a half minutes, most of it loading and compiling the
first prefill and decode shapes:

```bash
python tools/smoke_engine.py --model /data/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 --mode plain --max-model-len 2048 --num-gpu-blocks-override 8192 \
  --max-tokens 24 --maxtext model_name=llama3.3-70b \
  --maxtext load_parameters_path=/data/cchen104/maxtext_ckpt_70b/0/items
```

`"The capital of France is"` continues `" a city of love, fashion, art, and
romance, the Eiffel Tower"`. "fashion, art" and "art, fashion" are a measured
near-tie, and either order is correct.

## 7. Latency

`benchmarks/vllm_latency.py` times each step at the `device_get` that is its
only synchronization point, and reports the median over steps that did not
compile, counting compiles apart:

```bash
python benchmarks/vllm_latency.py \
  --model /data/Llama-3.3-70B-Instruct --tensor-parallel-size 8 --num-gpu-blocks-override 8192 \
  --batch 1 --batch 4 --batch 8 --gen-tokens 17 --maxtext model_name=llama3.3-70b \
  --maxtext load_parameters_path=/data/cchen104/maxtext_ckpt_70b/0/items
```

On 8x MI325X, 2026-11-19, with 130-token prompts and no compiles in any timed
window, against the previous stack (ROCm 7.2, jax 0.10.2, MaxText
`30105a4c0`) measured the same day:

| batch | TTFT | per step | p90 | tokens/s | previous stack, per step | tokens/s |
|---|---|---|---|---|---|---|
| 1 | 64.9 ms | 34.8 ms | 39.9 ms | 28.7 | 35.0 ms | 28.6 |
| 4 | 55.0 ms | 23.4 ms | 42.2 ms | 171.2 | 22.8 ms | 175.3 |
| 8 | 62.4 ms | 25.2 ms | 36.3 ms | 318.1 | 24.0 ms | 333.3 |

Two things this stack does differently cost 14 to 24% per step until fixed:
- jax 0.12.1's XLA all-gathers the vocab-sharded embedding table, 2.1 GB, to
  look up one decode token. The platform sets MaxText's `use_iota_embed`,
  whose one-hot lookup partitions locally. A standalone-engine caller at TP
  above 1 should set it too.
- JAX-AITER's KV append widened int32 slots into a buffer from
  `hipMallocAsync`, which blocks the host until the stream drains, once per
  layer. The widening worked around an AITER pin that read slots as int64;
  the pinned AITER reads int32 itself.
  `third_party/patches/jax-aiter-int32-slots.patch` passes the slots straight
  through, as proposed upstream.

The table was measured with an earlier fix that kept the widening and took
its buffer from XLA's scratch allocator. At TP=4 on 2026-11-22, batch 1, the
two match: 29.6 and 29.3 ms per step, against 34.7 ms with `hipMallocAsync`.

Repeated runs vary by up to 10%, so compare within a session.

Warming up the shapes took 221 s, 119 s and 140 s for the three batch sizes,
because each unseen shape compiles on first use. Compiling the bucketed shape
set at startup is on the roadmap.
