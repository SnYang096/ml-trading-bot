# 实盘 / 多腿上线路径说明（Runbook）

**目的**：合并到 `main` 并走现有 CI 部署后，理清「经典 PCM 实盘」与「多腿（chop_grid / dual_add_trend）」如何并行、GitHub 要配哪些密钥、服务器上还要做什么。

**相关文档**：

- 流水线：`.github/workflows/deploy.yml`
- 历史部署清单（三策略 BPC/ME/FER）：`docs/z实验_006_统一实盘/实盘部署TODO.md`
- 多腿架构与命令：`docs/architecture/live_stream/multi_leg_live_daemon.md`

---

## 1. 两条运行时路径（不要混成一个进程）

| 路径 | 入口脚本 | 典型策略 | 订单语义 |
|------|------------|----------|----------|
| **经典实盘** | `scripts/run_live.py`（生产镜像默认：`live/scripts/start_live.sh` → 同路径） | 由 `constitution.yaml` 的 `enabled_archetypes` 决定；代码里已接线 **BPC、ME、FER**（及可选 LV） | `TradeIntent` + `LivePCM` + `OrderManager` |
| **多腿实盘 / 影子** | `scripts/run_multi_leg_live.py` | `chop_grid`、`dual_add_trend`（`--strategies` 逗号分隔） | 多腿库存 + `MultiLegExecutionAdapter` + User Stream 成交确认 |

**结论**：可以同时在一台机器上跑 **「经典进程 + 多腿进程」**（推荐 **两个 systemd 单元** 或两条 Docker `command`），但：

- 多腿与经典 **强烈建议不同币安子账户**，避免持仓与挂单混在同一账户边界内。
- 当前 `run_multi_leg_live.py` 的 `--mode` 仅支持 **`shadow` / `testnet`**；**主网多腿**需在代码中扩展 `BinanceAPI(testnet=...)` 与密钥读取后再接生产（本 Runbook 主网部分以经典 `quant-engine` 为准）。

---

## 2. 你问的五种策略能否「同时」跑？

| 名称 | 能否与经典同进程 | 说明 |
|------|------------------|------|
| **BPC** | 能（经典） | `enabled_archetypes` 含 `bpc` 时注册。 |
| **ME** | 能（经典） | `enabled_archetypes` 含 `me` / `me-long` / `me-short` 等 ME 包前缀时注册（见 `run_live.py`）。 |
| **TPC** | **当前仓库未在 `run_live.py` 接线** | 经典路径里已接线的是 **FER**（及 BPC/ME/LV），不是 TPC。若 constitution 里只写 `tpc`，**不会**像 BPC 那样自动创建 `GenericLiveStrategy("tpc", …)`，需要按 FER 的模式增加工程接线后才可与 BPC/ME 同进程运行。 |
| **chop_grid** | 否（独立多腿进程） | `run_multi_leg_live.py --strategies chop_grid,...`。 |
| **dual_add_trend** | 否（独立多腿进程） | 同上，默认与 `chop_grid` 可写在同一 `--strategies` 里一次拉起。 |

因此：**合并到 `main` 后**，默认可靠组合是：

- **经典一条进程**：BPC + ME + FER（由 live 侧 `constitution.yaml` / `pcm_regime.yaml` 与策略包配置决定实际是否下单）。
- **多腿一条进程**：`chop_grid` 与 `dual_add_trend` 并行（testnet/shadow）。

若要 **TPC 与 BPC/ME 同进程**，需要先完成 `run_live.py` 侧与 FER 同级的 TPC 注册与特征时间框配置，再写入 `enabled_archetypes`。

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

**说明**：

- **`GITHUB_TOKEN`** 由 Actions 自带，用于 `docker/login-action` 推 ghcr，无需手建为 Secret（与 `GHCR_TOKEN` 不同）。
- **多腿专用密钥**（`MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY` / `SECRET`）**当前未写入** `deploy.yml`，不会自动下发到服务器。多腿若在服务器跑，需 **手动** 在主机上创建仅含多腿变量的 env 文件（如 `/opt/quant-engine/live/binance_multi_leg_testnet.env`，权限 `600`），并在独立 unit 里 `source`；勿与 `binance_mainnet.env` 混用若要坚持子账户隔离。

多腿脚本内的变量名（见 `scripts/run_multi_leg_live.py` 头部注释）：

- 推荐：`MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY`、`MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET`
- 若未设置且必须共用 testnet 密钥：需显式传入 **`--allow-shared-account`**（不推荐与经典并行主策略时共用）。

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
