#!/usr/bin/env bash
# Start business console on port 8800 (read-only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "$ROOT"
PORT="${MLBOT_CONSOLE_PORT:-8800}"
exec python -m uvicorn mlbot_console.main:app --host 127.0.0.1 --port "$PORT" --reload
