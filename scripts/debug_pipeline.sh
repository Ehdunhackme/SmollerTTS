#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m xiaotts.lexicon --subset debug
.venv/bin/python -m xiaotts.alignment --limit 32 --label debug-augmented --lexicons data/lexicons-debug --parallel-languages 5 --jobs 2
.venv/bin/python -m xiaotts.features --label debug-augmented --workers 8
.venv/bin/python -m xiaotts.train --features data/features/debug-augmented --output runs/debug-train-v1 --steps 2000 --eval-every 250
