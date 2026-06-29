#!/usr/bin/env bash
# One-shot verification. Run from repo root:
#   bash scripts/verify_blas_stable.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/scripts/activate_mlbot.sh"

echo "Python: $(which python)"
python "$ROOT/scripts/verify_blas_stable.py"
