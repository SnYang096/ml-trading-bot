#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:scripts
EXP=config/experiments/20260602_fast_scalp_tree_validate

echo "=== Rerun follow-up segments (inject + adverse tree gate fixes) ==="
python -m scripts.event_backtest --variant-grid "$EXP/segment_validate_followup_rerun.yaml"
echo "=== RERUN DONE ==="
