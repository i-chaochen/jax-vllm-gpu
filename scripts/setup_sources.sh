#!/usr/bin/env bash
# Creates the MaxText and vLLM source trees in ${VXG_ROOT}/src at their
# pins, and applies the hook patch from third_party/patches to MaxText. Git
# only, so it runs on the host or in the container. A tree that exists is left
# as it is.
#
# Each tree becomes a detached worktree of an existing checkout when
# MAXTEXT_REPO or VLLM_REPO names one (default: sibling ../maxtext and ../vllm,
# if they are git repositories), so no checkout of yours changes; otherwise it
# is cloned, with full history, which setuptools-scm needs to name vLLM's
# version.
#
# Environment: VXG_ROOT, MAXTEXT_REPO, MAXTEXT_REV, VLLM_REPO, VLLM_REV.
set -euo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
WORKSPACE=$(dirname "$REPO")
VXG_ROOT=${VXG_ROOT:-$WORKSPACE/.vllm_xla_gpu}
PATCHES=$REPO/third_party/patches
SRC=$VXG_ROOT/src
# maxtext-v0.2.5, the release jax-titan uses.
MAXTEXT_REV=${MAXTEXT_REV:-493d5928156869b70f7198ba7bbd1a7f779233d9}
MAXTEXT_REPO=${MAXTEXT_REPO:-$WORKSPACE/maxtext}
VLLM_REV=${VLLM_REV:-$(cat "$REPO/vllm_lkg.version")}
VLLM_REPO=${VLLM_REPO:-$WORKSPACE/vllm}
mkdir -p "$SRC"

# tree NAME LOCAL_REPO URL REV: sets CREATED=true when it made the tree.
# Not called from an if condition, where set -e would not stop on a failure.
tree() {
  local name=$1 local_repo=$2 url=$3 rev=$4 dst=$SRC/$1
  CREATED=false
  if [[ -e $dst ]]; then
    echo "$name: $dst exists ($(git -C "$dst" rev-parse --short HEAD)), leaving it"
    return 0
  fi
  if git -C "$local_repo" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$local_repo" cat-file -e "$rev^{commit}" 2>/dev/null || git -C "$local_repo" fetch -q origin
    git -C "$local_repo" worktree add -q --detach "$dst" "$rev"
    echo "$name: worktree of $local_repo at $rev"
  else
    git clone -q --filter=blob:none "$url" "$dst"
    git -C "$dst" checkout -q "$rev"
    echo "$name: cloned $url at $rev"
  fi
  CREATED=true
}

# patch_tree DIR PATCH...
patch_tree() {
  local dir=$1 p
  shift
  for p in "$@"; do
    git -C "$dir" apply --check "$PATCHES/$p" || { echo "$p does not apply to $dir" >&2; exit 1; }
    git -C "$dir" apply "$PATCHES/$p"
    echo "  applied $p"
  done
}

tree maxtext "$MAXTEXT_REPO" https://github.com/AI-Hypercomputer/maxtext "$MAXTEXT_REV"
if $CREATED; then patch_tree "$SRC/maxtext" maxtext-gpu-paged-hook.patch; fi
tree vllm "$VLLM_REPO" https://github.com/vllm-project/vllm "$VLLM_REV"
echo "sources ready in $SRC"
