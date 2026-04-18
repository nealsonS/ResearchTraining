#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh

# python -m venv .venv
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate

uv pip install --upgrade blinker


# check using nvidia-smi
# match cuda version of drivers
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

uv pip install -e .

# use nvidia-smi and change the `cuxxx` to have xxx match the cuda version right now
# uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp311-cp311-linux_x86_64.whl
