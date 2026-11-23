# Source inside the TheRock container: `source scripts/env.sh`.
VXG_REPO=${VXG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export VXG_ROOT=${VXG_ROOT:-$(dirname "${VXG_REPO}")/.vllm_xla_gpu}

source "${VXG_ROOT}/venv/bin/activate"

# The ROCm plugin wheels do not bundle ROCm; use the container's TheRock libs.
_rocm_root=$(rocm-sdk path --root)
export LD_LIBRARY_PATH="${_rocm_root}/lib:${_rocm_root}/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _rocm_root

# JAX-AITER runs from the submodule, where scripts/build_jax_aiter.sh builds
# its KV shims.
export JA_ROOT_DIR="${VXG_REPO}/third_party/jax-aiter"
export PYTHONPATH="${JA_ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_COMPILATION_CACHE_DIR="${VXG_ROOT}/cache/jax"

# A GPU memory fault otherwise writes a core dump of all GPU memory to the
# working directory.
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
