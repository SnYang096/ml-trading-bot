#!/usr/bin/env bash
# 诊断 Strategy Map · Trend 全页 No data：宿主 metrics → Prometheus target → 样本查询。
set -euo pipefail

TREND_PORT="${MLBOT_TREND_METRICS_PORT:-9190}"
PROM_URL="${MLBOT_PROMETHEUS_URL:-http://127.0.0.1:9091}"
JOB="quant-trend-swing"

echo "=== 1) 宿主 trend /metrics (127.0.0.1:${TREND_PORT}) ==="
metrics="$(curl -sfS --max-time 5 "http://127.0.0.1:${TREND_PORT}/metrics" 2>&1)" || {
  echo "FAIL: 无法访问 trend metrics。检查: systemctl status quant-trend-swing; docker ps | grep quant-trend-swing"
  exit 1
}
printf '%s\n' "$metrics" | sed -n '1,3p'
echo ""
echo "关键序列（节选）:"
printf '%s\n' "$metrics" | grep -E '^mlbot_(cpu_percent|uptime_seconds|dashboard_catalog|funnel_total|feature_bus_snapshot_age)' | head -20 || true
echo ""

echo "=== 2) Prometheus target ${JOB} ==="
up_json="$(curl -sfS --max-time 5 "${PROM_URL}/api/v1/query?query=up%7Bjob%3D%22${JOB}%22%7D" 2>/dev/null || true)"
if [[ -z "${up_json}" ]]; then
  echo "WARN: 无法查询 Prometheus (${PROM_URL})。在 monitoring 容器内: wget -qO- 'http://localhost:9091/api/v1/query?query=up{job=\"${JOB}\"}'"
else
  echo "${up_json}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('data',{}).get('result') or []
if not r:
    print('FAIL: 无 up 样本 — Status→Targets 中 host.docker.internal:9190 可能 DOWN')
    sys.exit(1)
for x in r:
    v=x.get('value',[None,'?'])[1]
    print('up =', v, 'labels =', x.get('metric',{}))
" || exit 1
fi
echo ""

echo "=== 3) TSDB 中是否有 cpu 样本 ==="
cpu_json="$(curl -sfS --max-time 5 "${PROM_URL}/api/v1/query?query=mlbot_cpu_percent%7Bjob%3D%22${JOB}%22%7D" 2>/dev/null || true)"
if [[ -n "${cpu_json}" ]]; then
  echo "${cpu_json}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('data',{}).get('result') or []
print('mlbot_cpu_percent series:', len(r))
if not r:
    print('FAIL: Prometheus 已 UP 但无 mlbot_cpu_percent — 检查 metric_relabel keep 或进程未 export')
"
fi
echo ""
echo "若 1) OK 且 2) DOWN: 重启 monitoring 栈并确认 prometheus.yml 中 quant-trend-swing → host.docker.internal:9190"
echo "若 1) FAIL: sudo systemctl restart quant-trend-swing"
echo "若 1–3) OK 仅漏斗空: 等 bus 事件触发 StatsCollector.flush，或查 quant-feature-bus / shared_feature_bus"
