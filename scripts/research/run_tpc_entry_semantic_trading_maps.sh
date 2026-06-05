#!/usr/bin/env bash
# Wrapper → canonical experiment under config/experiments/
set -euo pipefail
exec "$(dirname "$0")/../../config/experiments/20260604_tpc_entry_semantic_trading_maps/run_trading_maps.sh"
