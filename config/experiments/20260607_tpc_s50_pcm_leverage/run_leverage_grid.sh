#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python scripts/research/prepare_tpc_s50_pcm_leverage_experiments.py
python -m scripts.event_backtest \
  --variant-grid config/experiments/20260607_tpc_s50_pcm_leverage/tpc_s50_leverage_grid.yaml
