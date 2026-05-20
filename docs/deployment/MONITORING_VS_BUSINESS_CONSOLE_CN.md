# 监控栈（Grafana + Prometheus）与小后台（业务控制台）职责划分

> **业务 CMS 详细设计**（页面、API、库表、分期）：见 [BUSINESS_CONSOLE_DESIGN_CN.md](./BUSINESS_CONSOLE_DESIGN_CN.md)。

本文档回答：

- K 线等业务数据在 **Parquet 磁盘**时，与 **Prometheus 指标**如何分工；
- **Grafana + Prometheus 内存**是否值得保留，能否「全做到小后台」；
- 实施时建议的 **演进顺序**。

---

## 1. 目标与约束

| 需求 | 更合适的技术 |
|------|----------------|
| 进程存活、CPU、**磁盘/日志/warmup 目录**、错误率、漏斗/拒因 **时序** | 应用 `/metrics`（`quant-feature-bus` 暴露 `mlbot_disk_*`）→ **Prometheus** → **Grafana** |
| **告警**（Target down、拒因突增、账户更新失败） | Prometheus **Alertmanager** 或 Grafana 告警 |
| **K 线（OHLCV）**、Parquet **逐列特征**、大时间窗浏览 | **小后台**：读 Parquet/API + 图表库（Lightweight Charts / ECharts） |
| 订单/持仓/审计 **行级明细**、下钻单次决策 | **小后台 + 结构化存储**（SQLite/JSONL/审计日志），按需查询 |
| 内网可视化、不负责复杂交互 | Grafana **Explore** 查 PromQL |

**约束**：单机约 4G RAM（见 `deploy.yml` 注释）时，要控制 **常驻服务数量与 TSDB 保留**。

---

## 2. 职责划分（推荐）

### 2.1 Grafana + Prometheus（保留）

**职责**：「运行时健康 + 粗粒度策略地图」——与当前 `deploy/monitoring` 一致。

- 抓取：`quant-*` 容器暴露的 `:9090/metrics`（宿主映射 9190/9191/9192）。
- 面板：`quant_home`、`quant_strategy_map_trend`、`quant_strategy_map_hedge`、`quant_system` 等已 provision 的 JSON。
- **不做**：Parquet K 线全量浏览、任意 SQL 即席分析、逐笔订单生命周期的主界面。

**价值**：生态成熟、Alerting、多 job 对比、与部署 CI 已集成（`docker-compose.monitoring.yml`），**不必用自研重造「时序库 + 告警规则引擎」**。

### 2.2 小后台（新增，可选一期最小版）

**职责**：「业务可读性 + Parquet + 可选库表」。

- **K 线 API**：`GET /api/ohlcv?symbol=&timeframe=&from=&to=` → 从约定路径读 Parquet（与 `shared_feature_bus` / feature-store 布局对齐，只读）。
- **可选**：最近 N 条审计/拒因/下单摘要（读现有日志或 SQLite），**只读**、鉴权（Token / 内网 + 反代）。
- **前端**：单页或少量页面；**不**承担全局 infra 监控（仍交给 Grafana）。

**与 deploy 的关系**：可独立容器或 sidecar，**挂载只读**与 feature-bus 相同的宿主目录（参考 `deploy.yml`：`/opt/quant-engine/live/shared_feature_bus` 等），**不要**替代量化主进程。

---

## 3. Grafana + Prometheus 占用内存大吗？

经验量级（随指标基数、保留时间、查询并发变化，仅供规划）：

| 组件 | 粗估常驻 | 说明 |
|------|----------|------|
| **Grafana** | 约 150–400 MB | 面板数量、数据源多少会波动 |
| **Prometheus** | 约 300 MB–1.5+ GB | 与 `retention.time` / `retention.size`、series 数量强相关；仓库内配置为 **30d + 1GB 上限**（`docker-compose.monitoring.yml`） |

在 **4G 主机**上，若交易容器已占 3G+，监控栈仍通常可接受；若极端吃紧，可：

- 缩短 Prometheus 保留（例如 7d）或降低 `scrape_interval`（权衡精度）；
- 减少高基数自定义 label（业务明细不要全部进 Prom）。

**结论**：对「功能简单」若指 **只要 K 线 + 几张表**，理论上可只用小后台；但若仍要 **无侵入的进程监控、Alert、与现有 `mlbot_*` 指标一致**，**保留 Prometheus + Grafana 往往更省工程量**。更现实的折中是：**保留轻量监控栈 + 很小的只读控制台**，而不是「一个巨型后台吞下一切」。

---

## 4. 能否「全都做到小后台」？

| 方向 | 优点 | 缺点 |
|------|------|------|
| **监控也进小后台** | 少一套容器、少占一点内存的可能性 | 自研告警、TSDB 或兼容 Prom、长期维护成本高；现有 Dashboard/CI 作废 |
| **Grafana 管指标，小后台管 Parquet/业务** | 分工清晰、与当前仓库一致 | 多一个服务（可很小） |

**推荐**：**分离**——Grafana+Prometheus 继续；小后台专注 **Parquet K 线 + 业务只读**。

---

## 5. 实施阶段（建议）

1. **P0**：网络可达性（SSH 隧道 / VPN / HTTPS 反代），确保至少能打开 Grafana；Prometheus Target 全 UP。
2. **P1**：业务 CMS **Trade Map Live**（全 symbol、默认 2h、可切 15m/1m/1d、实盘开平仓标记、轮询刷新）；详见 [BUSINESS_CONSOLE_DESIGN_CN.md §4.2](./BUSINESS_CONSOLE_DESIGN_CN.md#42-实盘互动交易地图trade-map-live--核心能力)。
3. **P2**：审计/SQLite 只读 API（分页、按 symbol/time 过滤）；不与写路径竞争锁。
4. **P3**（可选）：Grafana 增加 **Loki** 采集审计日志 —— 仍建议「检索用 Loki，分析用后台」。

---

## 6. 相关文件

- 部署与卷约定：`.github/workflows/deploy.yml` 头部注释  
- 监控 compose：`deploy/monitoring/docker-compose.monitoring.yml`  
- 策略地图指标说明：`docs/deployment/STRATEGY_MAP_METRICS_CN.md`  
- 业务 CMS 设计（页面 / API / 数据源）：`docs/deployment/BUSINESS_CONSOLE_DESIGN_CN.md`  

---

## 7. 修订记录

- 2026-05-15：初版 — 职责划分、内存粗估、Parquet K 线走小后台。
