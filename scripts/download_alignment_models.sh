#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MFA_ROOT_DIR="$PWD/data/mfa"
export PATH="$PWD/.mfa/bin:$PATH"
mfa model download acoustic english_mfa mandarin_mfa spanish_mfa french_mfa german_mfa
mfa model download dictionary english_us_mfa mandarin_china_mfa spanish_spain_mfa french_mfa german_mfa
