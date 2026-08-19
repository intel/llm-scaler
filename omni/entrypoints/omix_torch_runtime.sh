#!/usr/bin/env bash

set -eo pipefail

source /opt/intel/oneapi/setvars.sh --force

shopt -s nullglob
torch_library_directories=(
    "${VIRTUAL_ENV}"/lib/python*/site-packages/torch/lib
)
shopt -u nullglob

if [[ ${#torch_library_directories[@]} -ne 1 ]]; then
    echo "expected one Torch library directory under ${VIRTUAL_ENV}" >&2
    exit 1
fi

# OMIX initialization prepends its compiler libraries. Restore the runtime
# packages installed alongside Torch so libsycl and libur_loader stay matched.
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${torch_library_directories[0]}:${LD_LIBRARY_PATH:-}"

exec "$@"
