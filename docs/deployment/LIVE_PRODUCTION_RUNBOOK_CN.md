# 实盘 / 多腿上线路径说明（Runbook）

**目的**：合并到 `main` 并走现有 CI 部署后，理清三进程架构：`quant-feature-bus` 行情发布器、`quant-trend-swing` 方向性 B·Trend 账户、`quant-hedge-multileg` hedge 多腿账户。

> **命名**：systemd / Docker / Prometheus `job` 均为 `quant-trend-swing`（宿主 metrics **9190**）。旧称 `quant-trend-fattail` 已废弃。

**相关文档**：

- 流水线：`.github/workflows/deploy.yml`
- 历史部署清单（方向性多策略）：`docs/z实验_006_统一实盘/实盘部署TODO.md`
- 多腿架构与命令：`docs/architecture/live_stream/multi_leg_live_daemon.md`

---

## 1. 两条运行时路径（不要混成一个进程）

| 路径 | 入口脚本 | 典型策略 | 订单语义 |
|------|------------|----------|----------|
| **trend/fat-tail 实盘** | `scripts/run_live.py`（生产镜像默认：`live/scripts/start_live.sh` → 同路径） | 由 `constitution.yaml` 的 `enabled_archetypes` 决定；代码里已接线 **BPC、ME、SRB、TPC**（及可选 LV） | `TradeIntent` + `LivePCM` + `OrderManager` |
| **Hedge 多腿实盘 / 影子** | `scripts/run_multi_leg_live.py` | `chop_grid`、`dual_add_trend`（`--strategies` 逗号分隔） | 多腿库存 + `MultiLegExecutionAdapter` + User Stream 成交确认 |

**结论**：可以同时在一台机器上跑 **「trend/fat-tail 进程 + hedge 多腿进程」**（推荐 **两个 systemd 单元** 或两条 Docker `command`），但：

- hedge 多腿与 trend/fat-tail **强烈建议不同币安子账户**，避免持仓与挂单混在同一账户边界内。
- `run_multi_leg_live.py` 支持 **`shadow` / `testnet` / `mainnet`**；生产主网由 `MULTI_LEG_BINANCE_FUTURES_API_KEY/SECRET` 驱动。

---

## 2. B·Trend (`quant-trend-swing`) vs Hedge Multi-leg：谁能同进程？

| 名称 | 能否与 trend/fat-tail 同进程 | 说明 |
|------|------------------|------|
| **BPC / ME / SRB / TPC** | 能（同一 `run_live`） | `enabled_archetypes` 控制是否注册；SRB/TPC 使用 **独立 timeframe 的 extra FC**（默认与 `meta.yaml` 一致，多为 120T；二者 timeframe 相同时共用一个增量 FC）。 |
| **chop_grid / dual_add_trend** | **否（刻意分进程）** | 走 `run_multi_leg_live.py`：`MultiLegOrchestrator`、对冲库存、User Stream 成交语义与 `OrderManager` / `TradeIntent` 不兼容，**不是**「再开一个线程」就能合并。 |

**合并到 `main` 后的推荐组合**：`quant-feature-bus` + `quant-trend-swing` + `quant-hedge-multileg`。

### 2.1 两个进程会不会「内存加倍」？

- **不会简单 ×2**：两份进程各自有 Python 堆与缓存，但 **只读代码页** 等可由 OS 共享；大头是 **tick/bar 缓冲与特征状态** —— 若强行塞进一个进程，仍要维护 **两套决策状态**（PCM 与多腿编排），峰值内存未必低于两进程，且 **故障域耦在一起**（一处 OOM/死锁全停）。
- **想共用一份行情**：可以两条服务 **挂同一块只读 tick 盘**、或未来做「单 WS 收包 → 多订阅者」的独立小服务；那是 **IPC/架构级** 优化，不等于把 `run_live` 与 `run_multi_leg_live` 合成一个脚本。

### 2.2 4GB / 窄带宽：先用磁盘 Feature Bus，不先上 IPC

**默认结论：第一版用 B 框架的磁盘 Feature Bus。** 它只保留一个行情/特征发布进程，trend/fat-tail 与 hedge 多腿进程从磁盘读已闭合的 bar/features；不先引入 Redis/NATS/ZeroMQ 等 IPC 队列。

更省事的顺序建议：

1. **先用磁盘共享**：`scripts/run_market_feature_publisher.py` 负责 Binance WS、1m bar、60T/120T/240T/2h feature snapshots，并原子写入 `live/shared_feature_bus/`。
2. **多腿读 feature-store**：`scripts/run_multi_leg_live.py --bar-source feature-store` 直接读取 `live/shared_feature_bus/features/2h/*.parquet`，避免第二条 market WS 与重复 tick buffer。
3. **trend/fat-tail 读 Feature Bus**：`MLBOT_FEATURE_SOURCE=bus` 时，`scripts/run_live.py` 不再启动 market WS，而是轮询 `live/shared_feature_bus/features/{TIMEFRAME}/*.parquet`，再走原来的 `LivePCM` / `OrderManager`。
4. **何时才值得做 IPC/单 WS 中继**：如果磁盘轮询延迟、IO 或跨机器分发成为瓶颈，再升级 SQLite WAL / Unix socket / NATS；不要第一版就上复杂队列。

磁盘 Feature Bus 布局：

```text
live/shared_feature_bus/
  bars_1min/{SYMBOL}.parquet
  features/{TIMEFRAME}/{SYMBOL}.parquet
  latest/{kind}/{SYMBOL}.json
```

写入采用「临时文件 → rename」原子替换；消费者只读完整 parquet，并按 timestamp 去重。trend/fat-tail bus 模式默认使用 `MLBOT_FEATURE_BUS_MAX_STALENESS_SECONDS=1800` 做 freshness fail-closed；过旧特征不会触发决策。publisher 额外启用 **fast execution bar**：任意 tick 到来时，只要相对当前 10s 微窗口 open 波动超过 3%，立即写一条 `_bar_kind=fast_intraminute` 的补充执行 bar；标准 1m bar 仍照常写出，不被改写。

---

## 3. GitHub：Repository secrets（与现网 `deploy.yml` 一致）

在 **Settings → Secrets and variables → Actions → Repository secrets** 中配置：

| Secret | 用途 |
|--------|------|
| `DEPLOY_HOST` | 服务器公网 IP |
| `DEPLOY_USER` | SSH 用户名（如 `ubuntu`） |
| `DEPLOY_SSH_KEY` | SSH 私钥全文 |
| `GHCR_TOKEN` | GitHub PAT（需 `read:packages` + `write:packages`，供服务器 `docker pull`） |
| `BINANCE_API_KEY` | trend/fat-tail 实盘主账户（写入服务器 `/opt/quant-engine/live/binance_mainnet.env`） |
| `BINANCE_API_SECRET` | 同上 |
| `MULTI_LEG_BINANCE_FUTURES_API_KEY` | （可选）hedge 多腿主网，与 `run_multi_leg_live.py` 变量名一致 |
| `MULTI_LEG_BINANCE_FUTURES_API_SECRET` | （可选）同上 |

**说明**：

- **`GITHUB_TOKEN`** 由 Actions 自带，用于 `docker/login-action` 推 ghcr，无需手建为 Secret（与 `GHCR_TOKEN` 不同）。
- **hedge 多腿主网**：若上述 `MULTI_LEG_*` 两个 Secret **都已配置**，部署步骤会追加写入 **`/opt/quant-engine/live/binance_mainnet.env`**（权限 `600`）。若任一为空，`quant-hedge-multileg` 不应启动主网。
- hedge 多腿与 trend/fat-tail 主账户必须使用不同子账户；不要复用 `BINANCE_*`。

hedge 多腿脚本内的变量名（见 `scripts/run_multi_leg_live.py` 头部注释）：

- 与写入文件一致：`MULTI_LEG_BINANCE_FUTURES_API_KEY`、`MULTI_LEG_BINANCE_FUTURES_API_SECRET`
- 主网生产不使用 `--allow-shared-account`；该选项仅保留给本地/测试场景。

### 3.1 与「仅手动 env」版本的区别

此前 hedge 多腿密钥只能运维手写；现网 workflow 会在 SSH 步骤中追加写入 `binance_mainnet.env`，供 `quant-hedge-multileg` 使用。

---

## 4. 服务器侧（三进程）

现有流程（与 `实盘部署TODO.md` 一致）：

1. **Bootstrap**（一次性）：`scripts/server_bootstrap.sh` 等，创建 `/opt/quant-engine/...` runtime 目录。
2. **Push 到 `main`**（或手动 Run workflow）：构建镜像 → SSH 写入 `binance_mainnet.env` → 重启三进程。
3. **Warmup prepare**：由 `quant-warmup-prepare` 与 `quant-feature-bus` 负责；消费者不再要求本机行情 ticks/bars。

健康检查：

```bash
sudo systemctl status quant-feature-bus quant-trend-swing quant-hedge-multileg
sudo journalctl -u quant-trend-swing -n 80 --no-pager
```

---

## 5. 服务器侧（三进程：Feature Bus + B·Trend + Hedge 多腿）

现网 workflow 会刷新以下 systemd units：

- `quant-feature-bus.service`：唯一 market WS，写 `live/shared_feature_bus/`
- `quant-trend-swing.service`：方向性 B·Trend 账户（`run_live.py`），`MLBOT_FEATURE_SOURCE=bus`
- `quant-hedge-multileg.service`：hedge 多腿账户，`--bar-source feature-store`

以下命令片段用于理解 unit 语义；实际以 `.github/workflows/deploy.yml` 写入的 unit 为准。

行情/特征发布进程（推荐先跑）：

```ini
# /etc/systemd/system/quant-feature-bus.service（示例）
[Service]
WorkingDirectory=/opt/quant-engine
ExecStart=/usr/bin/docker run --rm \
  --name quant-feature-bus \
  -v /opt/quant-engine/live:/opt/quant-engine/live \
  -v /opt/quant-engine/data:/opt/quant-engine/data \
  quant-engine:latest \
  python scripts/run_market_feature_publisher.py \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --feature-bus-root live/shared_feature_bus \
  --live-storage-base data/live_storage \
  --warmup-days 0
```

方向性趋势 / fat-tail 进程消费 Feature Bus：

```ini
# /etc/systemd/system/quant-trend-swing.service（示例差异）
[Service]
Environment=MLBOT_FEATURE_SOURCE=bus
Environment=MLBOT_FEATURE_BUS_ROOT=live/shared_feature_bus
Environment=MLBOT_FEATURE_BUS_POLL_SECONDS=5
Environment=MLBOT_FEATURE_BUS_MAX_STALENESS_SECONDS=1800
```

开启后，`run_live.py` 仍会初始化 `LivePCM` / `OrderManager` / User Stream，但不会启动 market `BinanceWebSocketClient`；特征与 1m bars 来自 `quant-feature-bus`。

Hedge 多腿进程消费 Feature Bus：

```ini
# /etc/systemd/system/quant-hedge-multileg.service（示例）
[Service]
EnvironmentFile=/opt/quant-engine/live/binance_mainnet.env
WorkingDirectory=/opt/quant-engine
ExecStart=/usr/bin/docker run --rm \
  --name quant-hedge-multileg \
  -v /opt/quant-engine/live:/opt/quant-engine/live \
  -v /opt/quant-engine/data:/opt/quant-engine/data \
  quant-engine:latest \
  python scripts/run_multi_leg_live.py --mode mainnet \
  --strategies chop_grid,dual_add_trend \
  --symbols BTCUSDT \
  --bar-source feature-store \
  --feature-bus-root live/shared_feature_bus \
  --feature-store-timeframe 2h
```

上线前请确认：

- 合约账户 **双向持仓（hedge）** 与脚本要求一致（非 shadow 时 orchestrator 会要求 hedge）。
- trend/swing bus 模式依赖 `quant-feature-bus` 已写出方向性策略所需 timeframe（如 `240T`、`120T`、`60T`）；缺失 timeframe 会导致对应策略被 PCM 跳过。
- trend/swing 与 hedge 多腿都会读取 `bars_1min` 作为执行时钟；慢周期 feature row 只作为 signal context。fast execution bar 仅用于更快执行止盈/止损，不用于重写标准 1m 特征历史。
- **`--bar-source feature-store`** 依赖 `quant-feature-bus` 已写出 `features/2h/{SYMBOL}.parquet`；若无新 timestamp，hedge 多腿循环会保持空转等待。
- hedge 多腿生产入口不再支持 `--bar-source websocket`；market WS 只允许在 `quant-feature-bus` 内运行。
- hedge 多腿 **SQLite** 路径若启用：使用 `--multi-leg-db-path` 指向持久卷（见脚本 `--help`）。

### 5.1 2h signal + 1min execution 回测复核

本次已跑：

```bash
PYTHONPATH=src python3 scripts/chop_grid_backtest.py \
  --timeframe 2h --execution-timeframe 1min \
  --start 2022-01-01 --end 2026-03-31 \
  --out-dir results/feature_bus_live/chop_grid_2h_1min --no-maps
```

结果摘要：`segments=723`、`trades=2878`、`return_pct=200.65`、`max_drawdown=-1.49%`、`risk_stop_rate=0.0`。对照纯 2h：`return_pct=222.96`、`max_drawdown=-1.51%`。1min execution 更保守但未暴露出 `max_loss_per_grid` 大量 intrabar 风险。

`dual_add_trend` 的 2h signal + 1min execution 结果较差：`return_pct=-476.80`、`risk_stop_rate=3.40%`、`loser_timeout_rate=46.19%`。对照纯 2h：`return_pct=585.05`、`risk_stop_rate=2.76%`、`loser_timeout_rate=0.0`。说明该策略对 execution timeframe 非常敏感；上线前不建议直接用现有参数上 `dual_add_trend`，应单独调参或先 shadow。

---

## 6. 监控与回滚

- B·Trend：日志 `journalctl -u quant-trend-swing`；监控 DB 见 `实盘部署TODO.md` 中 `live_monitor.db`。
- 部署失败时：`deploy.yml` 在 restart 失败会 `exit 1` 并打印最近 journal；可配合 **手动 pin 镜像 tag**（workflow 已打 `sha` tag）回滚。

---

## 7. 清单速查

- [ ] GitHub Secrets 已配置且 PAT 未过期  
- [ ] `live/highcap/config/constitution/constitution.yaml` 中 `enabled_archetypes` 与预期策略一致（BPC/ME/SRB/TPC）  
- [ ] `quant-feature-bus` warmup / Feature Bus 数据目录就绪
- [ ] `quant-feature-bus` 已写出 `live/shared_feature_bus/features/2h/*.parquet`
- [ ] Hedge 多腿：独立 env、独立账户、独立 systemd、主网 `MULTI_LEG_*` 密钥与 `--mode mainnet`
- [ ] 安全组仅开放必要端口；API Key 绑定服务器出口 IP  

更细的首次部署步骤仍以 **`docs/z实验_006_统一实盘/实盘部署TODO.md`** 为准；本文侧重 **publisher / trend-swing / hedge-multileg 三进程边界**。
