#!/usr/bin/env bash
# Re-run from any directory with uv 0.12.5. Existing pins remain preferred;
# pass --upgrade to deliberately refresh the transitive dependencies.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
uv_version=$(uv --version)
[[ "$uv_version" == "uv 0.12.5"* ]] || {
    echo "Expected uv 0.12.5; found $uv_version" >&2
    exit 1
}
uv pip compile lmcache_xpu_runtime_requirements.in \
    --constraints runtime-xpu.constraints.txt \
    --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
    --index-url https://pypi.org/simple --generate-hashes \
    --custom-compile-command 'bash docker/compile_lmcache_xpu_requirements.sh' \
    --output-file lmcache_xpu_runtime_requirements.txt "$@" >/dev/null
