#!/usr/bin/env bash
# Pre-deploy smoke: constitution live strategies + optional running console HTTP check.
#
# Local (no server):
#   ./scripts/smoke_console_live_strategies.sh
#
# Against prod console via SSH (no CI wait):
#   ./scripts/smoke_console_live_strategies.sh --remote ubuntu@13.113.18.30 -i ~/.ssh/awskeypair.pem
#
# Against local console:
#   ./scripts/smoke_console_live_strategies.sh --url http://127.0.0.1:8800
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE=""
SSH_KEY=""
URL=""
REQUIRED_MULTI="chop_grid,trend_scalp"
REQUIRED_TREND="tpc"
REQUIRED_SPOT="spot_accum_simple"

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    -i) SSH_KEY="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown arg: $1" >&2; usage 1 ;;
  esac
done

SSH_OPTS=(-o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

ok() {
  echo "OK: $*"
}

echo "== Local constitution (repo) =="
PYTHONPATH=src python3 - <<'PY' || fail "local constitution check"
import os
import sys
from pathlib import Path

from mlbot_console.config import SETTINGS
from mlbot_console.services.strategy_registry import get_live_console_strategies

get_live_console_strategies.cache_clear()
path = SETTINGS.constitution_yaml
if not path.is_file():
    raise SystemExit(f"constitution missing: {path}")
live = get_live_console_strategies()
ids = [s["id"] for s in live]
by_layer = {s["id"]: s["account_layer"] for s in live}
need = {
    "tpc": "trend",
    "chop_grid": "multi_leg",
    "trend_scalp": "multi_leg",
    "spot_accum_simple": "spot",
}
missing = [k for k in need if k not in ids]
if missing:
    raise SystemExit(f"missing live ids {missing}; got {ids}")
for sid, layer in need.items():
    if by_layer.get(sid) != layer:
        raise SystemExit(f"{sid} layer {by_layer.get(sid)!r} != {layer!r}")
bad = [i for i in ids if i in ("bpc", "me", "srb")]
if bad:
    raise SystemExit(f"research archetypes must not be live: {bad}")
print("live_strategy_ids", ids)
print("constitution", path)
PY
ok "repo constitution → tpc + chop_grid + trend_scalp + spot_accum_simple"

echo "== pytest (business_console subset) =="
pytest tests/business_console/test_feature_columns_api.py \
  tests/business_console/test_frontend_core.py \
  tests/business_console/test_console_live_strategies_smoke.py \
  -q --tb=short
ok "pytest subset passed"

fetch_taxonomy_json() {
  if [[ -n "$REMOTE" ]]; then
    ssh "${SSH_OPTS[@]}" "$REMOTE" \
      'curl -sS -m 8 http://127.0.0.1:8800/api/bus/features/taxonomy'
  elif [[ -n "$URL" ]]; then
    curl -sS -m 8 "${URL%/}/api/bus/features/taxonomy"
  else
    return 1
  fi
}

if [[ -n "$REMOTE" ]]; then
  echo "== Remote live/highcap/config (host mount) =="
  ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s <<'REMOTE_CHECK' || fail "live/highcap/config incomplete on host (CI must pack live/highcap/config)"
set -euo pipefail
test -f /opt/quant-engine/live/highcap/config/constitution/constitution.yaml
for sid in chop_grid trend_scalp tpc spot_accum_simple; do
  test -d "/opt/quant-engine/live/highcap/config/strategies/${sid}"
done
REMOTE_CHECK
  ok "constitution + live strategy dirs present on host"
fi

if [[ -n "$REMOTE" || -n "$URL" ]]; then
  echo "== HTTP taxonomy (running console) =="
  JSON="$(fetch_taxonomy_json)" || fail "could not reach console taxonomy API"
  PYTHONPATH=src python3 - <<'PY' "$JSON"
import json
import sys

data = json.loads(sys.argv[1]).get("data") or {}
src = data.get("constitution_source") or ""
live_ids = data.get("live_strategy_ids") or []
live = data.get("live_strategies") or []
ids = [str(s.get("id")) for s in live] if live else [str(x) for x in live_ids]
by_layer = {str(s["id"]): s.get("account_layer") for s in live} if live else {}
need = ("tpc", "chop_grid", "trend_scalp", "spot_accum_simple")
missing = [k for k in need if k not in ids]
if missing:
    raise SystemExit(
        f"API missing live strategies {missing}; live_strategy_ids={live_ids!r} "
        f"constitution_source={src!r} live_strategies_count={len(live)}"
    )
if not live:
    raise SystemExit(
        "API has no live_strategies[] (deploy newer mlbot_console); "
        f"live_strategy_ids={live_ids!r}"
    )
for sid in ("chop_grid", "trend_scalp"):
    if by_layer.get(sid) != "multi_leg":
        raise SystemExit(f"{sid} account_layer={by_layer.get(sid)!r}")
bad = [i for i in ids if i in ("bpc", "me", "srb")]
if bad:
    raise SystemExit(f"API still lists research strategies as live: {bad}")
print("HTTP live ids", ids)
print("constitution_source", src)
PY
  ok "running console taxonomy matches constitution"
else
  echo "== HTTP taxonomy skipped (use --url or --remote to verify deployed console) =="
fi

echo "All smoke checks passed."
