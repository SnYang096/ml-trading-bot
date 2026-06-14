#!/usr/bin/env bash
# Phase 5 acceptance: automated checks + optional live metrics scrape.
# Usage:
#   ./scripts/ops/check_execution_truth_sync_acceptance.sh
#   TREND_METRICS_URL=http://host:9190/metrics HEDGE_METRICS_URL=http://host:9191/metrics ./scripts/ops/check_execution_truth_sync_acceptance.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TREND_METRICS_URL="${TREND_METRICS_URL:-http://127.0.0.1:9190/metrics}"
HEDGE_METRICS_URL="${HEDGE_METRICS_URL:-http://127.0.0.1:9191/metrics}"

pass=0
fail=0
skip=0

ok() { echo "[PASS] $*"; pass=$((pass + 1)); }
bad() { echo "[FAIL] $*"; fail=$((fail + 1)); }
skip_msg() { echo "[SKIP] $*"; skip=$((skip + 1)); }

echo "=== Phase 5 automated (CI/local) ==="
if python -m pytest \
  tests/unit/test_segment_lifecycle.py \
  tests/unit/test_dual_add_trend_live_engine.py \
  tests/unit/test_metrics_reconciliation_scope.py \
  tests/unit/test_execution_truth_sync.py \
  tests/deploy/test_monitoring_provisioning.py \
  tests/order_management/test_order_manager.py::test_reconcile_open_orders_syncs_canceled_pending \
  -q; then
  ok "pytest Phase 5 suite"
else
  bad "pytest Phase 5 suite"
fi

echo
echo "=== Phase 5 live metrics scrape (optional) ==="

scrape_metric() {
  local url="$1"
  local pattern="$2"
  local label="$3"
  if curl -sf --max-time 5 "$url" 2>/dev/null | rg -q "$pattern"; then
    ok "$label ($url)"
    return 0
  fi
  if [[ "${REQUIRE_LIVE_METRICS:-0}" == "1" ]]; then
    bad "$label — endpoint unreachable or series absent ($url)"
    return 1
  fi
  skip_msg "$label — endpoint unreachable or series absent ($url)"
  return 0
}

scrape_metric "$TREND_METRICS_URL" 'mlbot_reconciliation_issue_count.*issue="open_reconcile_updated"' \
  "Trend open_reconcile_updated series"
scrape_metric "$TREND_METRICS_URL" 'mlbot_reconciliation_ok.*scope="trend"' \
  "Trend reconciliation_ok series"
scrape_metric "$HEDGE_METRICS_URL" 'mlbot_strategy_event_total.*event="segment_' \
  "Hedge segment_* strategy events"
scrape_metric "$HEDGE_METRICS_URL" 'mlbot_reconciliation_ok.*scope="hedge"' \
  "Hedge reconciliation_ok series"

echo
echo "=== Manual paper/live checklist (operator) ==="
cat <<'EOF'
| 观察项 | 通过标准 | 建议命令/面板 |
| ------ | -------- | ------------- |
| TP/SL fill 后 slot | on_execution_report 后 holds_real_grid_slot()==False | 审计日志 + multileg state JSON |
| ghost 清理 | segment_ghost_cleared 有计数；并发 cap 释放 | Grafana Hedge panel id 909 |
| stale pending | open_reconcile_updated 偶发>0 OK；持续 stale_local_order 需人工 | Grafana Trend panel id 909 |
EOF

echo
echo "=== Summary: pass=$pass fail=$fail skip=$skip ==="
if [[ "$fail" -gt 0 ]]; then
  exit 1
fi
