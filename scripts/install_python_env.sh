#!/usr/bin/env bash
# Installs this repository and the stack it runs on into the active Python:
#
#   - jax and jaxlib from WHEELS, and the ROCm plugin wheels when WHEELS has
#     them (a CPU-only machine needs none);
#   - torch, CPU only, at the version vLLM is built against;
#   - vLLM from the checkout VLLM_SRC, with no kernels;
#   - MaxText from the patched checkout MAXTEXT_SRC (scripts/setup_sources.sh);
#   - this package with its test and maxtext extras.
#
# Every other package is pinned by .github/ci-constraints.txt. CI and
# docs/getting_started.md both run this.
#
# Environment: WHEELS, VLLM_SRC (at VLLM_REV in third_party/pins.env),
# MAXTEXT_SRC.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CONSTRAINTS=$REPO/.github/ci-constraints.txt
: "${WHEELS:?directory holding the jax and jaxlib wheels}"
: "${VLLM_SRC:?vLLM checkout at VLLM_REV in third_party/pins.env}"
: "${MAXTEXT_SRC:?MaxText checkout with the hook patch, from scripts/setup_sources.sh}"
pip_install() { python -m pip install -c "$CONSTRAINTS" "$@"; }

pip_install "$WHEELS"/jax-*.whl "$WHEELS"/jaxlib-*.whl
shopt -s nullglob
plugins=("$WHEELS"/jax_rocm*_plugin-*.whl "$WHEELS"/jax_rocm*_pjrt-*.whl)
shopt -u nullglob
if ((${#plugins[@]})); then
  python -m pip install --no-deps "${plugins[@]}"
fi

# vLLM's build requirements name the torch it is built against. Only the CPU
# build will do: a ROCm torch adds a second HIP runtime, which aborts RCCL
# clique setup above one device.
build_requires=$(python - "$VLLM_SRC/pyproject.toml" <<'EOF'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
  print("\n".join(tomllib.load(f)["build-system"]["requires"]))
EOF
)
torch_version=$(sed -n 's/^torch *== *\([0-9.]*\)$/\1/p' <<<"$build_requires")
pip_install "torch==${torch_version:?no torch pin in the vLLM build requirements}+cpu" \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# `empty` builds no kernels and stamps the version +empty; a +cpu version would
# activate vLLM's own CPU platform instead of this one.
pip_install -r "$VLLM_SRC/requirements/common.txt"
mapfile -t build_tools < <(grep -v '^torch' <<<"$build_requires")
pip_install "${build_tools[@]}"
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation --no-deps "$VLLM_SRC"

python -m pip install --no-deps -e "$MAXTEXT_SRC"
pip_install -e "$REPO[test,maxtext]"
python -m pip check
