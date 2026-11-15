#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .mamba logs artifacts
if [ ! -x .mamba/bin/micromamba ]; then
  curl -fL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest -o .mamba/micromamba.tar.bz2
  tar -xjf .mamba/micromamba.tar.bz2 -C .mamba bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$PWD/.mamba/root"
.mamba/bin/micromamba create -y -p "$PWD/.mfa" -c conda-forge 'montreal-forced-aligner>=3.3,<3.4'
.mamba/bin/micromamba run -p "$PWD/.mfa" mfa version
.mamba/bin/micromamba list -p "$PWD/.mfa" --explicit > artifacts/aligner-environment-lock.txt
