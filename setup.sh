#!/bin/bash

[[ -d .venv/ ]] || python3 -mvenv .venv

export VIRTUAL_ENV_DISABLE_PROMPT=true
source .venv/bin/activate

python -mpip install --upgrade pip
pip install -r requirements.txt
