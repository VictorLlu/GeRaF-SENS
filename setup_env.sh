#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/venv"

export UV_CACHE_DIR="${PROJECT_ROOT}/.cache"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"

rm -rf "${VENV_DIR}"
uv sync --python 3.11 --frozen

source "${VENV_DIR}/bin/activate"
python setup.py build_ext --inplace

echo "Environment ready at ${VENV_DIR}"
