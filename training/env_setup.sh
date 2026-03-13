#!/bin/bash

pip install --upgrade pip
pip install uv

# python -m venv .venv
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate

uv pip install --upgrade --ignore-installed blinker
uv pip install -r requirements.txt
