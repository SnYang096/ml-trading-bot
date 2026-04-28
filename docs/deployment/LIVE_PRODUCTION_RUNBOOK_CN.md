# 实盘 / 多腿上线路径说明（Runbook）

**目的**：合并到 `main` 并走现有 CI 部署后，理清「经典 PCM 实盘」与「多腿（chop_grid / dual_add_trend）」如何并行、GitHub 要配哪些密钥、服务器上还要做什么。

**相关文档**：

- 流水线：`.github/workflows/deploy.yml`
- 历史部署清单（经典多策略）：`docs/z实验_006_统一实盘/实盘部署TODO.md`
- 多腿架构与命令：`docs/architecture/live_stream/multi_leg_live_daemon.md`

---

## 1. 两条运行时路径（不要混成一个进程）

| 路径 | 入口脚本 | 典型策略 | 订单语义 |
|------|------------|----------|----------|
| **经典实盘** | `scripts/run_live.py`（生产镜像默认：`live/scripts/start_live.sh` → 同路径） | 由 `constitution.yaml` 的 `enabled_archetypes` 决定；代码里已接线 **BPC、ME、SRB、TPC**（及可选 LV） | `TradeIntent` + `LivePCM` + `OrderManager` |
| **多腿实盘 / 影子** | `scripts/run_multi_leg_live.py` | `chop_grid`、`dual_add_trend`（`--strategies` 逗号分隔） | 多腿库存 + `MultiLegExecutionAdapter` + User Stream 成交确认 |

**结论**：可以同时在一台机器上跑 **「经典进程 + 多腿进程」**（推荐 **两个 systemd 单元** 或两条 Docker `command`），但：

- 多腿与经典 **强烈建议不同币安子账户**，避免持仓与挂单混在同一账户边界内。
- 当前 `run_multi_leg_live.py` 的 `--mode` 仅支持 **`shadow` / `testnet`**；**主网多腿**需在代码中扩展 `BinanceAPI(testnet=...)` 与密钥读取后再接生产（本 Runbook 主网部分以经典 `quant-engine` 为准）。

---

## 2. 经典 vs 多腿：谁能同进程？

| 名称 | 能否与经典同进程 | 说明 |
|------|------------------|------|
| **BPC / ME / SRB / TPC** | 能（同一 `run_live`） | `enabled_archetypes` 控制是否注册；SRB/TPC 使用 **独立 timeframe 的 extra FC**（默认与 `meta.yaml` 一致，多为 120T；二者 timeframe 相同时共用一个增量 FC）。 |
| **chop_grid / dual_add_trend** | **否（刻意分进程）** | 走 `run_multi_leg_live.py`：`MultiLegOrchestrator`、对冲库存、User Stream 成交语义与 `OrderManager` / `TradeIntent` 不兼容，**不是**「再开一个线程」就能合并。 |

**合并到 `main` 后的推荐组合**：经典一条服务（BPC+ME+SRB+TPC 视宪法） + 可选第二条服务跑多腿（testnet/shadow）。

### 2.1 两个进程会不会「内存加倍」？

- **不会简单 ×2**：两份进程各自有 Python 堆与缓存，但 **只读代码页** 等可由 OS 共享；大头是 **tick/bar 缓冲与特征状态** —— 若强行塞进一个进程，仍要维护 **两套决策状态**（PCM 与多腿编排），峰值内存未必低于两进程，且 **故障域耦在一起**（一处 OOM/死锁全停）。
- **想共用一份行情**：可以两条服务 **挂同一块只读 tick 盘**、或未来做「单 WS 收包 → 多订阅者」的独立小服务；那是 **IPC/架构级** 优化，不等于把 `run_live` 与 `run_multi_leg_live` 合成一个脚本。

### 2.2 4GB / 窄带宽：有没有必要上 IPC？

**默认结论：先不必做。** IPC 或「单 WS 多订阅」属于 **在已确认瓶颈之后** 的工程投入；在没量过 RSS、swap、OOM 和出口带宽之前，优先级通常低于 **减载**。

更省事的顺序建议：

1. **量一下再决定**：`free -h`、`systemctl status`、有无 OOM killer 日志；多腿若开 WS，看出口是否长期顶满。没有持续 OOM/严重丢包，就不必为「理论上的双倍」去写 IPC。
2. **减载比 IPC 便宜**：少挂几个 symbol、缩短 warmup / `memory_window_hours`（多腿见 `run_multi_leg_live.py --help`）、经典侧控制 tick 保留策略；多腿若可接受 **parquet 驱动 bar**（`--bar-source parquet`），可减少 **第二条行情 WS** 对带宽的占用（与经典是否 WS 无关，需你接受数据路径差异）。
3. **算力与内存真的不够时**：常见做法是 **拆机**——例如 4G 小机只跑 `quant-engine`，多腿 testnet 放到第二台低配或本机，而不是先在 4G 上叠 IPC 复杂方案。
4. **何时才值得做 IPC/单 WS 中继**：长期观测表明 **重复订阅同一组 trade/aggTrade** 占满带宽或 CPU 解压成为主因，且减载后仍不可接受——再立项单独的行情服务更划算。

---

## 3. GitHub：Repository secrets（与现网 `deploy.yml` 一致）

在 **Settings → Secrets and variables → Actions → Repository secrets** 中配置：

| Secret | 用途 |
|--------|------|
| `DEPLOY_HOST` | 服务器公网 IP |
| `DEPLOY_USER` | SSH 用户名（如 `ubuntu`） |
| `DEPLOY_SSH_KEY` | SSH 私钥全文 |
| `GHCR_TOKEN` | GitHub PAT（需 `read:packages` + `write:packages`，供服务器 `docker pull`） |
| `BINANCE_API_KEY` | 经典实盘主账户（写入服务器 `/opt/quant-engine/live/binance_mainnet.env`） |
| `BINANCE_API_SECRET` | 同上 |
| `MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY` | （可选）多腿 testnet，与 `run_multi_leg_live.py` 变量名一致 |
| `MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET` | （可选）同上 |

**说明**：

- **`GITHUB_TOKEN`** 由 Actions 自带，用于 `docker/login-action` 推 ghcr，无需手建为 Secret（与 `GHCR_TOKEN` 不同）。
- **多腿 testnet**：若上述 `MULTI_LEG_*` 两个 Secret **都已配置**，部署步骤会写入 **`/opt/quant-engine/live/binance_multi_leg_testnet.env`**（权限 `600`），与主账户 `binance_mainnet.env` 分离。若 **任一空**，流水线会 **跳过** 该文件，不报错（多腿服务需自行 `EnvironmentFile=` 或本地手写 env）。
- 多腿与主账户仍 **强烈建议不同子账户**；勿把 testnet 多腿密钥误当主网经典密钥使用。

多腿脚本内的变量名（见 `scripts/run_multi_leg_live.py` 头部注释）：

- 与写入文件一致：`MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY`、`MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET`
- 若未配置 `MULTI_LEG_*` 且必须共用 testnet 密钥：需显式 **`--allow-shared-account`**（不推荐与经典并行主策略时共用）。

### 3.1 与「仅手动 env」版本的区别

此前多腿密钥只能运维手写；**现网 workflow 已在 SSH 步骤中增加可选写入**。未配 Secret 时行为与旧版一致（跳过）；配齐后 **systemd 多腿 unit** 可直接 `EnvironmentFile=/opt/quant-engine/live/binance_multi_leg_testnet.env`。

---

## 4. 服务器侧（经典 `quant-engine`）

现有流程（与 `实盘部署TODO.md` 一致）：

1. **Bootstrap**（一次性）：`scripts/server_bootstrap.sh` 等，创建 `/opt/quant-engine/...` 与 `quant-engine.service`。
2. **Push 到 `main`**（或手动 Run workflow）：构建镜像 → SSH 写入 `binance_mainnet.env` → `systemctl restart quant-engine`。
3. **Warmup ticks**：若 `start_live.sh` 要求本地 ticks，需按 TODO 中文档的 tar/scp 流程准备，否则启动可能中止。

健康检查：

```bash
sudo systemctl status quant-engine
sudo journalctl -u quant-engine -n 80 --no-pager
```

---

## 5. 服务器侧（多腿：第二进程示例）

以下仅为 **示例**；路径与 unit 名按你方规范调整。

```ini
# /etc/systemd/system/quant-multi-leg.service（示例）
[Service]
EnvironmentFile=/opt/quant-engine/live/binance_multi_leg_testnet.env
WorkingDirectory=/opt/quant-engine
ExecStart=/usr/bin/docker run --rm \
  --name quant-multi-leg \
  -v /opt/quant-engine/live:/opt/quant-engine/live \
  quant-engine:latest \
  python scripts/run_multi_leg_live.py --mode testnet \
  --strategies chop_grid,dual_add_trend \
  --symbols BTCUSDT \
  --bar-source websocket
```

上线前请确认：

- 合约账户 **双向持仓（hedge）** 与脚本要求一致（非 shadow 时 orchestrator 会要求 hedge）。
- **`--bar-source websocket`** 时仍需磁盘上足够历史 bar/tick 或合理 **`--warmup-days`**，避免冷启动特征计算因 tick 不足失败（见 `multi_leg_live_daemon.md`）。
- 多腿 **SQLite** 路径若启用：使用 `--multi-leg-db-path` 指向持久卷（见脚本 `--help`）。

---

## 6. 监控与回滚

- 经典：日志 `journalctl -u quant-engine`；监控 DB 见 `实盘部署TODO.md` 中 `live_monitor.db`。
- 部署失败时：`deploy.yml` 在 restart 失败会 `exit 1` 并打印最近 journal；可配合 **手动 pin 镜像 tag**（workflow 已打 `sha` tag）回滚。

---

## 7. 清单速查

- [ ] GitHub 6 个 Secrets 已配置且 PAT 未过期  
- [ ] `live/highcap/config/constitution/constitution.yaml` 中 `enabled_archetypes` 与预期策略一致（BPC/ME/FER；TPC 需代码支持）  
- [ ] 经典 warmup / 数据目录就绪  
- [ ] 多腿（若启用）：独立 env、独立账户、独立 systemd、testnet 密钥与 `--mode testnet`  
- [ ] 安全组仅开放必要端口；API Key 绑定服务器出口 IP  

更细的首次部署步骤仍以 **`docs/z实验_006_统一实盘/实盘部署TODO.md`** 为准；本文侧重 **双路径并行与密钥边界**。
