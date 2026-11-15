#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs data artifacts
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
# Ubuntu's system flatbuffers has a Debian version string rejected by new pip.
# Shadow it inside this environment; never uninstall the provider's system package.
.venv/bin/python -m pip install --ignore-installed 'flatbuffers>=24,<26'
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip freeze > artifacts/environment-lock.txt
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
