#!/bin/bash

pip install --upgrade pip
pip install uv

# python -m venv .venv
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate

uv pip install --upgrade blinker
uv pip install -r requirements.txt
# uv pip install flash-attn --no-build-isolation
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp311-cp311-linux_x86_64.whl
