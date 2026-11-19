# Getting started

jax-vllm-gpu runs in a ROCm container with a self-built ROCm JAX, a CPU-only
torch and vLLM built with no kernels of its own. MaxText and JAX-AITER are
reached from checkouts beside this repository. The container mounts the
directory holding all of them at `/tf`, and `/data` for checkpoints.

## 1. Environment

The tested set, 2026-11-18:

| Component | Version |
|---|---|
| ROCm | 7.2, gfx942 (MI300X) |
| jax, jaxlib, jax-rocm7-plugin, jax-rocm7-pjrt | 0.10.2, built from source |
| flax | 0.12.8 (0.12.9 needs jax 0.11.1) |
| torch | 2.13.0, CPU build |
| vLLM | `94a54f581e`, built with `VLLM_TARGET_DEVICE=empty` |
| MaxText | ROCm/maxtext `30105a4c0` (branch `chao/kv_paged_kernels`), on `PYTHONPATH` |
| JAX-AITER | ROCm/jax-aiter `bf64025`, KV shims built with `Makefile.kv`, on `PYTHONPATH` |

The container used so far was built this way, inside it:

```bash
# Never let pip replace the self-built jax and jaxlib.
printf 'jax==0.10.2.dev0+selfbuilt\njaxlib==0.10.2.dev0+selfbuilt\n' > /tmp/pin.txt

# torch, CPU only: a ROCm torch adds a second HIP runtime, which aborts RCCL
# clique setup above one device.
pip install -c /tmp/pin.txt torch --index-url https://download.pytorch.org/whl/cpu

# vLLM with no kernels. `empty` stamps the version +empty; a +cpu version would
# activate vLLM's own CPU platform.
pip install -c /tmp/pin.txt -r /tf/vllm/requirements/common.txt
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation --no-deps -e /tf/vllm
# xgrammar's published 0.2.x wheels need a newer libstdc++ than AlmaLinux 8 has.
pip install --no-deps --no-binary xgrammar "xgrammar==0.2.3"

# JAX-AITER's KV shims, about 3 minutes at -j16.
cd /tf/jax-aiter && make -f Makefile.kv ja_kv GFX=gfx942 -j16 && python3 scripts/prebuild_pa_ragged.py

# This package: pure Python, no dependencies declared.
pip install --no-deps --no-build-isolation -e /tf/jax-vllm-gpu
```

Git refuses to read a mounted repository as root until it is marked safe:
`git config --global --add safe.directory '*'` inside the container.

The commands below run from the host. `ENV` is what every run needs:

```bash
C=<container>
ENV=(-e PYTHONDONTWRITEBYTECODE=1 -e JA_ROOT_DIR=/tf/jax-aiter
     -e PYTHONPATH=/tf/jax-aiter:/tf/maxtext/src
     -e VLLM_PLUGINS=jax_vllm_gpu,jax_vllm_gpu_kv_connector)
```

`VLLM_PLUGINS` matters only while another JAX platform plugin is installed in
the same environment, since vLLM refuses to activate two.

## 2. Tests

Wrap every container command in `timeout`, and run one GPU invocation at a
time: a `docker exec` that never returns blocks every later command in the
same shell.

```bash
# CPU: gpu_only tests skip. 355 passed, 62 skipped on 2026-11-18.
timeout 900 docker exec "${ENV[@]}" -e JAX_PLATFORMS=cpu -e DECOUPLE_GCLOUD=TRUE $C \
  bash -lc 'cd /tf/jax-vllm-gpu && python3 -m pytest tests -q'

# GPU: everything, including the KV kernels and the engine parity tests.
# 416 passed, 1 skipped on 8x MI300X.
timeout 2400 docker exec "${ENV[@]}" -e DECOUPLE_GCLOUD=TRUE $C \
  bash -lc 'cd /tf/jax-vllm-gpu && python3 -m pytest tests -q'

# The skipped one compares against MaxText's dense path, which needs exactly
# the engine's device visible. 7 passed.
timeout 1200 docker exec "${ENV[@]}" -e DECOUPLE_GCLOUD=TRUE -e HIP_VISIBLE_DEVICES=0 $C \
  bash -lc 'cd /tf/jax-vllm-gpu && python3 -m pytest tests/test_decode_parity.py -q'
```

`DECOUPLE_GCLOUD=TRUE` gives MaxEngine the JetStream stubs it imports. Read the
pass and skip counts: a GPU test missing its `gpu_only` mark runs on CPU, and
one that skips reports success.

The control plane needs only numpy, which is what CI runs:

```bash
PYTHONPATH=src python -m pytest tests/test_kv_common.py tests/test_kv_control.py \
  tests/test_kv_import_rule.py tests/test_kv_prefix_cache.py tests/test_kv_residency.py
```

## 3. Smoke runs through vLLM's engine

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
S=/tf/jax-vllm-gpu/tools/smoke_engine.py

timeout 900 docker exec "${ENV[@]}" $C python3 $S "${TINY[@]}" --mode plain
timeout 900 docker exec "${ENV[@]}" $C python3 $S "${TINY[@]}" --mode chunked --max-num-batched-tokens 64 --max-tokens 8
timeout 900 docker exec "${ENV[@]}" $C python3 $S "${TINY[@]}" --mode preempt \
  --max-model-len 256 --max-num-seqs 6 --num-gpu-blocks-override 24 --max-tokens 200
timeout 900 docker exec "${ENV[@]}" $C python3 $S "${TINY[@]}" --mode prefix --enable-prefix-caching --max-tokens 12
```

Greedy output must not depend on scheduling. Run `chunked` at
`--max-num-batched-tokens 64` and `512` with `--dump-tokens`, and the two dumps
must be byte-identical. The 512 run exits 1, because at that budget the prompt
is not chunked and the mode did not reach its case.

Llama-3.3-70B at TP=8, from a MaxText checkpoint converted with MaxText's
`to_maxtext`. Loading takes about 40 s, the first prefill about a minute to
compile and the first decode shape about 90 s:

```bash
timeout 2400 docker exec "${ENV[@]}" $C python3 $S --model /data/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 --mode plain --max-model-len 2048 --num-gpu-blocks-override 8192 \
  --max-tokens 24 --maxtext model_name=llama3.3-70b \
  --maxtext load_parameters_path=/data/cchen104/maxtext_ckpt_70b/0/items
```

`"The capital of France is"` continues `" a city of love, fashion, art, and
romance, the Eiffel Tower"`. "fashion, art" and "art, fashion" are a measured
near-tie, and either order is correct.

## 4. Latency

`tools/bench_latency.py` times each step at the `device_get` that is its only
synchronization point, and reports the median over steps that did not
compile, counting compiles apart:

```bash
timeout 3000 docker exec "${ENV[@]}" $C python3 /tf/jax-vllm-gpu/tools/bench_latency.py \
  --model /data/Llama-3.3-70B-Instruct --tensor-parallel-size 8 --num-gpu-blocks-override 8192 \
  --batch 1 --batch 4 --batch 8 --gen-tokens 17 --maxtext model_name=llama3.3-70b \
  --maxtext load_parameters_path=/data/cchen104/maxtext_ckpt_70b/0/items
```

On 8x MI300X, 2026-11-19, with 130-token prompts and no compiles in any timed
window:

| batch | TTFT | per step | p90 | tokens/s |
|---|---|---|---|---|
| 1 | 55.5 ms | 34.9 ms | 42.1 ms | 28.6 |
| 4 | 58.7 ms | 24.4 ms | 39.6 ms | 163.8 |
| 8 | 56.4 ms | 23.9 ms | 44.2 ms | 335.1 |

Warming up the shapes took 304 s, 645 s and 1,100 s for the three batch sizes,
because each unseen shape compiles on first use. Compiling the bucketed shape
set at startup is on the roadmap.
