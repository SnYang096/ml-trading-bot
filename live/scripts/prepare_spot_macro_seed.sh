#!/usr/bin/env bash
# Download Binance Vision spot 1d klines and build weekly EMA200 seed parquets.
# Shared macro data for any strategy using weekly_ema_200_position on the feature bus.
# Does NOT block quant-feature-bus. Production: quant-macro-seed-prepare.timer (daily 02:00 UTC).
# After seeds exist: restart quant-feature-bus (or wait ~15m), then quant-spot-accum / trend / multileg.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [ -n "${MLBOT_LIVE_SYMBOLS:-}" ]; then
  SYMBOLS="${MLBOT_LIVE_SYMBOLS}"
else
  SYMBOLS="$(python - <<'PY'
from src.live_data_stream.universe_symbols import read_universe_symbols
print(",".join(read_universe_symbols("highcap")))
PY
)"
fi
KLINE_ROOT="${MLBOT_MACRO_KLINE_ROOT:-live/highcap/data/macro/spot_klines}"
SEED_ROOT="${MLBOT_WEEKLY_EMA_SEED_ROOT:-live/highcap/data/macro/spot_weekly_ema200}"
START_DATE="${MLBOT_MACRO_SEED_START_DATE:-2017-01-01}"

echo "=== spot macro seed prepare ==="
echo "symbols=$SYMBOLS"
echo "kline_root=$KLINE_ROOT seed_root=$SEED_ROOT start=$START_DATE"

python scripts/prepare_spot_weekly_ema_seed.py \
  --symbols "$SYMBOLS" \
  --kline-root "$KLINE_ROOT" \
  --seed-root "$SEED_ROOT" \
  --start-date "$START_DATE"

echo ""
echo "=== verify seed files ==="
for sym in $(echo "$SYMBOLS" | tr ',' ' '); do
  f="${SEED_ROOT}/${sym}.parquet"
  if [[ -f "$f" ]]; then
    echo "OK $f ($(stat -c%s "$f" 2>/dev/null || stat -f%z "$f") bytes)"
  else
    echo "MISSING $f"
    exit 1
  fi
done

echo ""
echo "Next: sudo systemctl restart quant-feature-bus   # or wait ~15m for bus recompute"
echo "Then: sudo systemctl restart quant-spot-accum"
