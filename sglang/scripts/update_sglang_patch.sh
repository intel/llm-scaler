#!/usr/bin/env bash
# Regenerate patches/sglang_for_multi_arc.patch from a local sglang clone.
#
# The Dockerfile builds sglang by cloning the pristine upstream tag
# (sgl-project/sglang @ SGLANG_UPSTREAM_TAG) and then `git apply`-ing this
# patch, so the patch must always equal:
#
#   git diff <upstream tag> <our dev-bmg branch> -- python/
#
# Run this after pushing new commits to the analytics-zoo/sglang dev-bmg
# branch (or whenever the local dev-bmg working repo has commits the patch
# doesn't reflect yet) to keep the patch in sync:
#
#   ./scripts/update_sglang_patch.sh
#
# Optional overrides:
#   SGLANG_REPO=/path/to/sglang/checkout \
#   UPSTREAM_TAG=v0.5.13 \
#   BRANCH=dev-bmg \
#     ./scripts/update_sglang_patch.sh
#
# BRANCH may be a local branch, a remote-tracking ref (origin/dev-bmg), or
# any other committish -- it must point at a *commit* (uncommitted working
# tree changes are NOT included; commit them first).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_FILE="${SGLANG_DIR}/patches/sglang_for_multi_arc.patch"

SGLANG_REPO="${SGLANG_REPO:-/home/intel/cengguang/sglang-bmg/sglang}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/sgl-project/sglang.git}"
UPSTREAM_TAG="${UPSTREAM_TAG:-v0.5.13}"
ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
BRANCH="${BRANCH:-dev-bmg}"

if [[ ! -d "${SGLANG_REPO}/.git" ]]; then
    echo "error: SGLANG_REPO (${SGLANG_REPO}) is not a git checkout" >&2
    exit 1
fi

cd "${SGLANG_REPO}"

if git remote get-url "${UPSTREAM_REMOTE}" >/dev/null 2>&1; then
    echo "Fetching ${UPSTREAM_REMOTE} tag ${UPSTREAM_TAG}..."
    git fetch "${UPSTREAM_REMOTE}" "tag" "${UPSTREAM_TAG}" --no-tags
else
    echo "Fetching ${UPSTREAM_TAG} directly from ${UPSTREAM_URL}..."
    git fetch "${UPSTREAM_URL}" "tag" "${UPSTREAM_TAG}" --no-tags
fi

echo "Fetching ${ORIGIN_REMOTE} ${BRANCH}..."
git fetch "${ORIGIN_REMOTE}" "${BRANCH}" || true

# Prefer the up-to-date remote-tracking ref if it exists, else fall back to
# whatever BRANCH resolves to locally (e.g. a plain branch name).
if git rev-parse --verify -q "${ORIGIN_REMOTE}/${BRANCH}" >/dev/null; then
    BRANCH_REF="${ORIGIN_REMOTE}/${BRANCH}"
else
    BRANCH_REF="${BRANCH}"
fi

# Warn (but don't fail) if the local working tree has uncommitted changes
# relative to BRANCH_REF -- those won't be picked up by `git diff <a> <b>`.
if ! git diff --quiet "${BRANCH_REF}" -- python/ 2>/dev/null; then
    echo "warning: working tree differs from ${BRANCH_REF} under python/;" >&2
    echo "         uncommitted changes will NOT be included in the patch." >&2
    echo "         Commit (and push) first if they should be." >&2
fi

echo "Generating patch: diff(${UPSTREAM_TAG}, ${BRANCH_REF}) -- python/"
git -c core.fileMode=false diff "${UPSTREAM_TAG}" "${BRANCH_REF}" -- python/ > "${PATCH_FILE}.new"

if [[ ! -s "${PATCH_FILE}.new" ]]; then
    echo "error: generated patch is empty -- refusing to overwrite ${PATCH_FILE}" >&2
    rm -f "${PATCH_FILE}.new"
    exit 1
fi

# Sanity check: the patch must still apply cleanly to a pristine checkout of
# UPSTREAM_TAG, exactly as the Dockerfile does (`git apply` after
# `git clone --depth 1 -b <tag>`).
VERIFY_DIR="$(mktemp -d)"
trap 'rm -rf "${VERIFY_DIR}"' EXIT
echo "Verifying patch applies cleanly against a fresh worktree of ${UPSTREAM_TAG}..."
git worktree add --detach -f "${VERIFY_DIR}" "${UPSTREAM_TAG}" >/dev/null
if ! git -C "${VERIFY_DIR}" apply --whitespace=nowarn --check "${PATCH_FILE}.new"; then
    echo "error: regenerated patch does NOT apply cleanly to ${UPSTREAM_TAG} -- aborting" >&2
    git worktree remove --force "${VERIFY_DIR}"
    rm -f "${PATCH_FILE}.new"
    exit 1
fi
git worktree remove --force "${VERIFY_DIR}"

if diff -q "${PATCH_FILE}" "${PATCH_FILE}.new" >/dev/null 2>&1; then
    echo "Patch unchanged (already up to date)."
    rm -f "${PATCH_FILE}.new"
else
    mv "${PATCH_FILE}.new" "${PATCH_FILE}"
    echo "Updated ${PATCH_FILE}"
    echo "  $(git -C "${SGLANG_REPO}" diff --stat "${UPSTREAM_TAG}" "${BRANCH_REF}" -- python/ | tail -1)"
fi
