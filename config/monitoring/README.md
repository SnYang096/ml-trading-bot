# Monitoring manifests

Authoritative guide: [`docs/strategy/漂移监控_mlbot_monitor_CN.md`](../../docs/strategy/漂移监控_mlbot_monitor_CN.md).

## 远程自动跑（推荐）

**不要**手敲 `mlbot monitor run …`。用 **独立调度进程**（不要塞进 feature-bus publisher）：

```bash
mlbot monitor schedule --cadence weekly   # 或 monthly | quarterly | yearly
mlbot monitor schedule --list
```

systemd（远程）：

```bash
sudo systemctl enable --now mlbot-monitor-daily.timer
sudo systemctl enable --now mlbot-monitor-weekly.timer
sudo systemctl enable --now mlbot-monitor-monthly.timer
# quarterly / yearly 同理
```

CMS 页面：**/monitoring**（漂移监控卡片：绿/红/橙缺勤）。

Telegram：

- 每次 `schedule` 若 **exit≠0** 或 **watchdog/drift ALERT** → 自动发（`MLBOT_MONITOR_SKIP_TG=1` 可关）
- **daily** 跑完后检查各 cadence 是否缺勤 → 缺勤发 TG（`mlbot monitor check-staleness`）

```bash
# 仅缺勤检查
mlbot monitor check-staleness
```

每次跑完会写：

| 产物 | 路径 | CMS |
|------|------|-----|
| 汇总索引 | `results/monitoring/index.json` | `GET /api/monitoring/index` |
| 分 cadence 快照 | `results/monitoring/latest_<cadence>.json` | 同上 |
| 明细报告 | `…/watchdog/*/report.json`, `…/drift/*/drift_report.json` | 下钻路径在 index 里 |
| SQLite 索引 | `results/rd_registry.sqlite` → `monitor_event` | `GET /api/monitoring/events` |

本地与远程 **同路径、同 schema**，可直接 diff `index.json` 或拷贝 `rd_registry.sqlite` 对比。

## 远程数据原则（bus-only）

| 不做 | 做 |
|------|-----|
| `train_final` 全历史当 current | Publisher → **feature bus** 滚动 parquet |
| monitor 时 `prepare-only` | `export-window` 从 bus 切片 |
| 手動 cron `monitor run` | `monitor schedule --cadence …` |

## Cadence → manifest

见 [`schedules.yaml`](schedules.yaml)：

| Cadence | Manifest | 内容 |
|---------|----------|------|
| weekly | `weekly_rule_stack.yaml` | 7d + 全 bus；watchdog + drift |
| monthly | `monthly_drift.yaml` | 30d bus；drift |
| quarterly | `quarterly_drift.yaml` | 30d + 全 bus；watchdog + drift |
| yearly | `yearly_drift.yaml` | 全 bus；drift 归档 |

## Run（调试）

```bash
mlbot monitor schedule --cadence weekly --dry-run
mlbot monitor run --config config/monitoring/weekly_rule_stack.yaml   # 仅调试
```

## Environment

| Variable | Default |
|----------|---------|
| `MLBOT_FEATURE_BUS_ROOT` | `live/shared_feature_bus` |
| `MLBOT_WATCHDOG_LOOKBACK_DAYS` | `7` |
| `MLBOT_DRIFT_LOOKBACK_DAYS` | `0`（全 bus） |
| `MLBOT_RD_REGISTRY_DB` | `results/rd_registry.sqlite` |
| `MLBOT_MONITOR_FORCE_SUBPROCESS` | unset（默认进程内直调；设 1 强制走 subprocess 旧路径，作为紧急回退） |

Legacy: [`mlbot-weekly-watchdog.service`](../../etc/systemd/mlbot-weekly-watchdog.service) 已改为调用 `schedule --cadence weekly`。
