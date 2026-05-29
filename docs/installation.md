# Installation

This page collects all installation routes for GeRaF. **We recommend [uv](https://github.com/astral-sh/uv)**
(also summarized in the main [README](../README.md#installation)); the `requirements.txt` and
Docker workflows are kept here for completeness.

## Requirements

- Linux
- Python 3.11 recommended
- NVIDIA GPU for training / fast inference
- CUDA-compatible PyTorch environment

## Recommended: `uv`

The easiest way to reproduce the Python environment is with `uv sync`. This repo ships:

- [pyproject.toml](../pyproject.toml) for dependency definitions
- [uv.lock](../uv.lock) for exact dependency resolution
- `uv` source configuration for CUDA 12.1 PyTorch wheels
- [setup_env.sh](../setup_env.sh) as a convenience wrapper

### Option 1: one-command setup

```bash
bash setup_env.sh
source venv/bin/activate
```

This will:

- create `venv/`
- sync dependencies with `uv`
- build the CUDA extensions in place

### Option 2: manual `uv sync`

```bash
export UV_CACHE_DIR=$PWD/.cache
export UV_PROJECT_ENVIRONMENT=$PWD/venv
uv sync --python 3.11 --frozen
source venv/bin/activate
python setup.py build_ext --inplace
```

Notes:

- `uv sync --frozen` recreates the exact environment from `uv.lock`
- the CUDA extensions are still built separately with `python setup.py build_ext --inplace`
- this is more portable than copying `venv/` between machines

## Alternative: `requirements.txt`

If you prefer the older direct install flow, you can still install from
[requirements.txt](../requirements.txt).

Create and activate a virtual environment:

```bash
uv venv venv --python 3.11
source venv/bin/activate
```

Install PyTorch for CUDA 12.1:

```bash
UV_CACHE_DIR=$PWD/.cache uv pip install --python venv/bin/python --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1
```

Install the remaining Python dependencies:

```bash
UV_CACHE_DIR=$PWD/.cache uv pip install --python venv/bin/python -r requirements.txt
```

Build the CUDA extensions:

```bash
python setup.py build_ext --inplace
```

This path is still supported, but `uv sync --frozen` is the more reproducible option because it
follows the checked-in [uv.lock](../uv.lock).

## Docker

For containerized setup, a [Dockerfile](../Dockerfile) is provided at the repo root. The image:

- uses `nvcr.io/nvidia/pytorch:23.10-py3`
- installs Python requirements from `requirements.txt`
- builds CUDA extensions during image build
