#!/usr/bin/env bash
# Builds JAX-AITER's paged-KV FFI shims in the third_party/jax-aiter submodule,
# against the jax in the active venv. Run inside the container after
# `source scripts/env.sh`, with the submodule and its AITER fetched:
#
#   git submodule update --init --recursive third_party/jax-aiter
#
# Applies third_party/patches/jax-aiter-*.patch to the submodule first, skipping
# any already applied. The shims compile AITER's kernels into their own modules
# and link none of its JIT libraries, so nothing is downloaded. About three
# minutes at -j16.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/jax-aiter"
GPU_ARCHS=${GPU_ARCHS:-gfx942}

for p in ../patches/jax-aiter-*.patch; do
  if git apply --reverse --check "$p" 2>/dev/null; then
    echo "$(basename "$p"): already applied"
  else
    git apply --check "$p" || { echo "$p does not apply to third_party/jax-aiter" >&2; exit 1; }
    git apply "$p"
    echo "applied $(basename "$p")"
  fi
done

make -f Makefile.kv ja_kv GFX="${GPU_ARCHS}" -j"${JOBS:-16}"
python -c "import jax_aiter.kv; print('JAX-AITER KV shims built for ${GPU_ARCHS}')"
