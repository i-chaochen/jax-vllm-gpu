#!/usr/bin/env bash
# Builds JAX-AITER's paged-KV FFI shims in the third_party/jax-aiter submodule,
# against the jax in the active venv. Run inside the container after
# `source scripts/env.sh`, with the submodule and its AITER fetched:
#
#   git submodule update --init --recursive third_party/jax-aiter
#
# The shims compile AITER's kernels into their own modules and link none of its
# JIT libraries, so nothing is downloaded. About three minutes at -j16.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/jax-aiter"
GPU_ARCHS=${GPU_ARCHS:-gfx942}

make -f Makefile.kv ja_kv GFX="${GPU_ARCHS}" -j"${JOBS:-16}"
python -c "import jax_aiter.kv; print('JAX-AITER KV shims built for ${GPU_ARCHS}')"
