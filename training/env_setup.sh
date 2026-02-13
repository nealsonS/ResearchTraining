#!/bin/bash

pip install --upgrade pip
pip install uv

uv venv .venv
source .venv/bin/activate
pip install -r requirements
