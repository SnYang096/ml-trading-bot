#!/usr/bin/env bash
# Project venv + macOS BLAS stability env. Usage:
#   source scripts/activate_mlbot.sh
set -euo pipefail

_MLBOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$_MLBOT_ROOT"

if [[ ! -f "$_MLBOT_ROOT/.venv/bin/activate" ]]; then
  echo "Missing .venv — create with: python3.12 -m venv .venv && pip install -e '.[dev]'" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$_MLBOT_ROOT/.venv/bin/activate"
# shellcheck source=/dev/null
source "$_MLBOT_ROOT/scripts/env_macos_blas.sh"

export MLBOT_ROOT="$_MLBOT_ROOT"
