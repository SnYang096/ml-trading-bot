# Prometheus + Grafana 实盘监控面板设计

> 创建日期: 2026-02-24
> 依赖: 实盘监控系统设计.md Part B
> 目标: 交易机器人暴露 Prometheus 指标 → Grafana 可视化

---

## 架构

```
Trading Bot (run_live.py)
  └── MetricsExporter (port 9090)
        └── HTTP /metrics  ← Prometheus 每 15s 抓取
                                 ↓
                            Prometheus (port 9091)
                                 ↓
                            Grafana (port 3000)
                              └── quant.json dashboard
```

## 数据源映射

| Prometheus 指标 | 来源 | 类型 |
|----------------|------|------|
| `mlbot_bars_processed_total` | StatsCollector | Counter |
| `mlbot_funnel_total{stage}` | StatsCollector.flush() | Counter |
| `mlbot_signals_total{strategy}` | by_strategy | Counter |
| `mlbot_orders_total{strategy}` | record_order_placed() | Counter |
| `mlbot_positions_active` | _open_positions | Gauge |
| `mlbot_pnl_realized_total` | order_management.db | Gauge |
| `mlbot_drawdown` | constitution state | Gauge |
| `mlbot_loss{period}` | constitution state | Gauge |
| `mlbot_kill_switch_halted` | safety_state | Gauge |
| `mlbot_last_bar_age_seconds{symbol}` | listener 心跳 | Gauge |
| `mlbot_ws_connected{symbol}` | WS 状态 | Gauge |
| `mlbot_cpu_percent` | psutil | Gauge |
| `mlbot_memory_mb` | psutil | Gauge |
| `mlbot_uptime_seconds` | 进程启动时间 | Gauge |
| `mlbot_gate_reject_rate` | gate_passed/bars | Gauge |

---

## Grafana 面板布局

### Row 0: 一眼看全局 (Stat 面板)

```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│  宪法状态     │  今日 PnL    │  持仓数/slot │  Regime      │  运行时长     │
│  🟢 GREEN    │  +0.35%      │  1/2         │  NORMAL      │  3d 12h      │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

### Row 1: 好消息 — 赚钱了吗

| 面板 | 类型 | PromQL | 说明 |
|------|------|--------|------|
| 累计 PnL 曲线 | Time Series | `mlbot_pnl_realized_total` | 最核心的一条线 |
| 每日 PnL | Bar Chart | `delta(mlbot_pnl_realized_total[24h])` | 绿涨红跌 |
| 按策略 PnL | Stacked Bar | `mlbot_orders_total` by strategy | 各策略贡献 |
| 信号产出率 | Gauge | `rate(mlbot_orders_total[24h])` | 每天下多少单 |

### Row 2: 坏消息 — 需要处理

| 面板 | 类型 | PromQL | 说明 |
|------|------|--------|------|
| Kill Switch | State Timeline | `mlbot_kill_switch_halted` | 红=暂停 |
| Drawdown 水位 | Gauge+阈值 | `mlbot_drawdown` vs 0.20 | 接近红线变红 |
| 日/周/月亏损 | Bar Gauge | `mlbot_loss{period=~"daily\|weekly\|monthly"}` | 三条红线 |
| 数据新鲜度 | Table | `mlbot_last_bar_age_seconds` per symbol | >300s 告警 |
| WS 连接状态 | Status Map | `mlbot_ws_connected` per symbol | 绿=在线 |

### Row 3: 信号管线 (调参参考)

| 面板 | 类型 | PromQL | 说明 |
|------|------|--------|------|
| 信号漏斗 | Bar Gauge | `rate(mlbot_funnel_total[1h])` per stage | 各阶段通过率 |
| Gate 拦截率 | Time Series | `mlbot_gate_reject_rate` | 正常 70-90% |
| 策略信号量 | Stacked Area | `rate(mlbot_signals_total[1h])` per strategy | BPC/ME/FER |
| Slot 占用 | Gauge | `mlbot_positions_active` / 2 | 坑位使用率 |

### Row 4: 系统健康

| 面板 | 类型 | PromQL | 说明 |
|------|------|--------|------|
| CPU | Time Series | `mlbot_cpu_percent` | 趋势 |
| 内存 | Time Series | `mlbot_memory_mb` | 泄漏检测 |
| Uptime | Stat | `mlbot_uptime_seconds` | 有无偶发重启 |

---

## 实现文件

| 文件 | 职责 |
|------|------|
| `src/time_series_model/live/metrics_exporter.py` | Prometheus 指标定义 + HTTP server |
| `stats_collector.py` (修改) | flush() 时同步更新 Prometheus 指标 |
| `run_live.py` (修改) | 启动 metrics HTTP server |
| `terraform/monitoring/prometheus.yml` | Prometheus 抓取配置 |
| `terraform/monitoring/grafana-provisioning/dashboards/quant.json` | Grafana 面板 |
| `terraform/monitoring/docker-compose.monitoring.yml` | 本地 Prometheus+Grafana |

## 本地启动命令

```bash
# 1. 启动 Prometheus + Grafana
cd terraform/monitoring && docker compose -f docker-compose.monitoring.yml up -d

# 2. 启动交易机器人 (自带 /metrics 端口 9090)
bash live/scripts/start_live.sh

# 3. 打开 Grafana
# http://localhost:3000  (admin/admin)
# Dashboard → Quant Engine
```
