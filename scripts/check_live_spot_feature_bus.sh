#!/usr/bin/env bash
# Post-deploy checks for spot_accum + feature-bus weekly_ema (run on server or via SSH).
#
# Macro seed: quant-macro-seed-prepare (oneshot) + quant-macro-seed-prepare.timer (daily 02:00 UTC).
# NOT built inside feature-bus startup. Shared by spot/trend/multileg via bus column weekly_ema_200_position.
set -euo pipefail

BUS_ROOT="${MLBOT_FEATURE_BUS_ROOT:-/opt/quant-engine/live/shared_feature_bus}"
STRATEGIES_ROOT="${MLBOT_SPOT_STRATEGIES_ROOT:-/opt/quant-engine/live/highcap/config/strategies}"

echo "=== containers ==="
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' \
  | grep -E 'quant-feature-bus|quant-spot-accum|NAMES' || true

echo ""
echo "=== macro weekly EMA seed files ==="
MACRO_SEED="${MLBOT_WEEKLY_EMA_SEED_ROOT:-/opt/quant-engine/live/highcap/data/macro/spot_weekly_ema200}"
ls -la "${MACRO_SEED}/"*.parquet 2>/dev/null || echo "MISSING seed parquets under ${MACRO_SEED}"

echo ""
echo "=== feature-bus: macro seed + merge consumer (last 300) ==="
docker logs quant-feature-bus 2>&1 | tail -300 \
  | grep -E 'macro seed|spot weekly EMA|merge consumer|enabled_archetypes|quant-feature-bus stack' \
  || echo "(no macro/merge lines yet)"

echo ""
echo "=== spot-accum: weekly_ema / prefilter (last 80) ==="
docker logs quant-spot-accum 2>&1 | tail -80 \
  | grep -E 'weekly_ema|spot-eligibility|missing_weekly|Prefilter feature' || echo "(no spot-eligibility lines)"

echo ""
echo "=== feature bus snapshot (Python) ==="
docker exec quant-feature-bus python3 - <<'PY'
from src.live_data_stream.feature_bus import FeatureBusReader

r = FeatureBusReader("/app/live/shared_feature_bus")
for sym in ("BTCUSDT", "ETHUSDT"):
    row = r.latest_features(sym, "120T") or {}
    wk = row.get("weekly_ema_200_position")
    atr = row.get("atr_percentile")
    ok = wk is not None and wk != 0.0
    print(
        f"{sym} keys={len(row)} weekly_ema_200_position={wk!r} "
        f"(expect negative when below weekly EMA; nonzero={ok}) atr_percentile={atr!r}"
    )
PY

echo ""
echo "=== live spot features.yaml present? ==="
ls -la "${STRATEGIES_ROOT}/spot_accum_simple/features.yaml" 2>/dev/null || echo "MISSING: ${STRATEGIES_ROOT}/spot_accum_simple/features.yaml"
