# TrendPositionTruthSync 运维巡检手册

> 版本: v1.0 (2026-06-14)
> 适用范围: P0-P5 架构演进后的 B 层 Trend 持仓管理

---

## 自动化巡检

### 每日定时任务

**触发时间**: 每天 06:30 UTC (systemd timer: `mlbot-monitor-daily.timer`)

**执行脚本**: `deploy/systemd/mlbot-monitor-docker-run.sh daily`

**包含检查**:
1. 原有 daily health monitor (feature bus, watchdog, equity snapshot)
2. **Truth Sync Health Check** (`scripts/check_truth_sync_health.py --days 1`)

**失败通知**: 通过 `mlbot-monitor-notify@.service` 发送 Telegram 告警

### 10 分钟周期对账

**触发**: `run_live.py` 内置 asyncio 任务

**日志**: `Periodic trend reconcile: symbol=X issues=Y`

**Metrics**: `data/metrics/truth_sync_reconcile_{symbol}.jsonl`

---

## 每日快速检查 (推荐)

```bash
# 1. Truth Sync 健康检查
python3 scripts/check_truth_sync_health.py --days 1

# 2. Reconcile metrics 汇总
cat data/metrics/truth_sync_reconcile_*.jsonl 2>/dev/null | jq -s '
  group_by(.symbol) |
  map({symbol: .[0].symbol, total_issues: [.[].issues] | add})'

# 3. SQLite 当前 open 行数
for db in data/live/*/trend_order_management.db; do
  echo "$db: $(sqlite3 "$db" 'SELECT COUNT(*) FROM positions WHERE status="open"' 2>/dev/null || echo 0) open"
done
```

---

## P0-P5 验收标准

### P0 止血 — 重启后能管仓

| 检查项 | 命令 | 成功标准 |
|--------|------|----------|
| tracker 恢复 | `grep "loaded position tracker" /var/log/mlbot/*.log` | 每个 symbol 有记录 |
| SL 挂单存在 | `grep "SL stop_order_id" /var/log/mlbot/*.log` | 非空 stop_order_id |
| 无 ghost segment | `grep "close.*ghost" /var/log/mlbot/*.log` | 无匹配 |

### P1 单写入口 — SQLite 唯一写入口

| 检查项 | 命令 | 成功标准 |
|--------|------|----------|
| duplicate_position_row_closed | `python3 scripts/check_truth_sync_health.py` | 计数 = 0 |
| SQLite 无重复 open 行 | 同上 Check 1 | `duplicate open rows: 0` |

### P2 统一恢复 — 新 pid 格式

| 检查项 | 命令 | 成功标准 |
|--------|------|----------|
| pid 格式 | `jq -r '.position_id' data/position_trackers/*_live.json` | `{BASE}:live_{12位}` |
| execution.yaml RR | `jq '.risk_config' data/position_trackers/*_live.json` | 有 RR/trailing 配置 |

### P3 周期对账 — 10 分钟 reconcile

| 检查项 | 命令 | 成功标准 |
|--------|------|----------|
| reconcile 日志 | `grep "Periodic trend reconcile" /var/log/mlbot/*.log` | 每 10 分钟一次 |
| 对账结果 | 日志 `issues=` 后内容 | `issues=0` 或仅有 `api_error` 可恢复 |
| sqlite_orphan_open | metrics 文件 | 计数 = 0 |

### P4 CMS 投影 — 只读 projection

| 检查项 | 命令 | 成功标准 |
|--------|------|----------|
| Open Positions 页面 | CMS → Open Positions | 显示正确，无重复行 |
| Trend 数据来源 | debug 日志搜 `Used TTS projection` | 走 projection 路径（非 fallback） |

### P5 Live 验收 — 3 个交易日

运行 **3 个交易日**后:

```bash
python3 scripts/check_truth_sync_health.py --days 3
```

**全部 PASS 即为验收通过**:
- Check 1: `duplicate open rows: 0`
- Check 2: `orphan open rows: 0`
- Check 3: `missing snapshots: 0`
- Check 4: `duplicate_position_row_closed count: 0`

---

## 异常处置

| 异常现象 | 可能原因 | 处置步骤 |
|----------|----------|----------|
| `duplicate_position_row_closed > 0` | P1 单写入口有漏洞 | 检查是否有代码绕过 TTS 写 SQLite |
| `sqlite_orphan_open > 0` | exchange flat 但 SQLite 未同步 | P3 reconcile 应自动修复；持续出现检查 `api.fetch_positions` |
| CMS 显示重复行 | P4 projection 路径未生效 | 检查 `open_positions_list.py` import 路径 |
| tracker 无 SL | JSON 快照损坏或丢失 | `python3 scripts/sync_trend_positions_from_exchange.py --disaster-recovery --dry-run` 确认后执行 |
| reconcile 失败 api_error | 网络或 API 临时故障 | 自动恢复，下次 reconcile 重试 |

---

## 灾备恢复

### DR 模式 (从 exchange 重建)

```bash
# Dry run 预览
python3 scripts/sync_trend_positions_from_exchange.py \
  --config live/highcap/config/live_config.yaml \
  --disaster-recovery --dry-run

# 实际执行 (谨慎!)
python3 scripts/sync_trend_positions_from_exchange.py \
  --config live/highcap/config/live_config.yaml \
  --disaster-recovery --no-dry-run
```

### 单 symbol 重建

```bash
python3 scripts/bootstrap_position_tracker_from_exchange.py \
  --symbol BTCUSDT --archetype tpc --bar-minutes 120 \
  --live-config live/highcap/config/live_config.yaml
```

---

## 文件位置

| 文件 | 路径 |
|------|------|
| 健康检查脚本 | `scripts/check_truth_sync_health.py` |
| TTS 核心模块 | `src/order_management/trend_position_truth_sync.py` |
| 单写入口 | `TrendPositionTruthSync.project_to_sqlite()` |
| Reconcile metrics | `data/metrics/truth_sync_reconcile_{symbol}.jsonl` |
| JSON 快照 | `data/position_trackers/{symbol}_live.json` |
| SQLite DB | `data/live/{symbol}/trend_order_management.db` |
| systemd timer | `etc/systemd/mlbot-monitor-daily.timer` |
| docker run 脚本 | `deploy/systemd/mlbot-monitor-docker-run.sh` |

---

## Concurrent Access

`periodic_reconcile` 在 `run_live.py` 中通过 `run_in_executor` 于后台线程运行，与主线程的 `enforce_all` / `PositionTracker` 更新并发。

| 层面 | 行为 |
|------|------|
| dict 读写 | CPython GIL 保证单步原子性 |
| 逻辑一致性 | **best-effort** — reconcile 与主线程之间无显式锁 |
| 设计假设 | reconcile 仅 heal SQLite orphan/duplicate、补 bootstrap；不修改 exchange 状态 |
| 风险窗口 | 极短 — 最坏情况下下次 10 分钟 reconcile 或 restart `on_restart` 再对齐 |

**运维建议**: 若 reconcile metrics 偶发 `tracker_exchange_qty_mismatch` 且下一周期自愈，可忽略；持续出现则检查是否有 bypass TTS 的 SQLite 写入。

---

## 变更日志

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-06-14 | v1.0 | 初始版本，P0-P5 架构演进完成 |
| 2026-06-15 | v1.1 | 补充 Concurrent Access 节；P4 debug 日志说明 |
