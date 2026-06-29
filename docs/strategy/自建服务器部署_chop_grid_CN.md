# 自建服务器部署 chop_grid（验证 → Testnet → Mainnet）

> **适用**：你有 VPS（AWS/阿里云等），回测满意，想在自己机器上跑 **C 系统 chop_grid**。  
> **默认不部署 trend_scalp**（constitution 里当前禁用；live 对齐与回测仍有 gap）。  
> **原则**：先 Shadow → 再 Testnet（`--no-orders`）→ 再小资金 Mainnet。**回测好 ≠ 可以直接真钱。**

---

## 0. 你需要准备

| 项 | 说明 |
|----|------|
| 服务器 | Ubuntu 22.04/24.04，**≥4GB 内存**，50GB 磁盘 |
| 网络 | 能访问 Binance API / WebSocket（国内常需代理，自行配置） |
| 代码 | `git clone` 本仓库到服务器 |
| 历史数据 | 本机 `rsync data/parquet_data/` 到服务器（Feature Bus warmup 用） |
| API Key | **你自己的** 币安合约 Key；先 **Testnet**，不要先用专家的 Key |

---

## 1. 架构（只跑 chop，最小两进程）

```text
quant-feature-bus      收行情 + 算特征 → 写 live/shared_feature_bus/
quant-hedge-multileg   读 Feature Bus → chop_grid 决策 → 下单（Testnet/Mainnet）
```

**不必**先跑 `quant-trend-swing`（那是 B 系统 BPC/TPC）。

生产 live 使用 **120T 信号 + 1min 执行**（与部分回测 `2h` 不同，见 §6）。

---

## 2. 一次性：服务器初始化

SSH 登录后：

```bash
# 依赖
sudo apt update && sudo apt install -y git docker.io docker-compose-plugin rsync

sudo usermod -aG docker $USER
# 重新登录使 docker 组生效

# 目录（与专家机类似，可改成 ~/ml-trading-bot）
export DEPLOY_ROOT=/opt/quant-engine
sudo mkdir -p $DEPLOY_ROOT/{live/shared_feature_bus,live/highcap/data,data}
sudo chown -R $USER:$USER $DEPLOY_ROOT
```

### 2.1 拉代码

```bash
cd $DEPLOY_ROOT
git clone <你的仓库地址> repo
cd repo
```

### 2.2 同步历史数据（从本机 Mac 示例）

在本机执行：

```bash
rsync -avz --progress \
  /Users/jerry/project/yin/ml-trading-bot/data/parquet_data/ \
  user@你的服务器IP:/opt/quant-engine/data/parquet_data/
```

### 2.3 构建 Docker 镜像

```bash
cd /opt/quant-engine/repo
docker build -f docker/Dockerfile.live -t quant-engine:latest .
```

首次构建约 10～20 分钟。

### 2.4 API 密钥文件（勿提交 git）

```bash
cat > /opt/quant-engine/live/binance_mainnet.env <<'EOF'
# Testnet（推荐第一步）
MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY=填你的
MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET=填你的

# Mainnet 确认后再加（与 testnet 二选一使用，见下方 mode）
# MULTI_LEG_BINANCE_FUTURES_API_KEY=
# MULTI_LEG_BINANCE_FUTURES_API_SECRET=
EOF
chmod 600 /opt/quant-engine/live/binance_mainnet.env
```

币安合约 Testnet：https://testnet.binancefuture.com  
账户需开启 **双向持仓 Hedge Mode**。

---

## 3. 阶段 A：Shadow（不下单，确认能转）

无需 API Key，用 parquet 回放：

```bash
cd /opt/quant-engine/repo
docker run --rm -it \
  -v /opt/quant-engine/data:/app/data \
  -v /opt/quant-engine/live:/app/live \
  quant-engine:latest \
  python scripts/run_multi_leg_live.py \
    --mode shadow \
    --bar-source parquet \
    --strategies chop_grid \
    --symbols BTCUSDT \
    --data-dir data/parquet_data \
    --once
```

无 traceback、末尾有 bar 处理日志 → 进入阶段 B。

---

## 4. 阶段 B：Feature Bus + 多腿（接近实盘）

### 4.1 终端 1 — Feature Bus（常驻）

```bash
docker run --rm --name quant-feature-bus \
  -p 9192:9090 \
  -e MLBOT_LIVE_STORAGE_BASE=/app/live/highcap/data \
  -e MLBOT_FEATURE_BUS_ROOT=/app/live/shared_feature_bus \
  -v /opt/quant-engine/live/shared_feature_bus:/app/live/shared_feature_bus \
  -v /opt/quant-engine/live/highcap/data:/app/live/highcap/data \
  -v /opt/quant-engine/data:/app/data \
  quant-engine:latest \
  python scripts/run_market_feature_publisher.py \
    --universe highcap \
    --symbols BTCUSDT,ETHUSDT \
    --feature-bus-root live/shared_feature_bus \
    --live-storage-base live/highcap/data \
    --strategies-root live/highcap/config/strategies \
    --warmup-days 7 \
    --max-rows 10080
```

等日志出现 **features 写入**、且存在：

```bash
ls live/shared_feature_bus/features/120T/BTCUSDT.parquet
# 路径在宿主机：/opt/quant-engine/live/shared_feature_bus/features/120T/
```

### 4.2 终端 2 — Testnet 只观察（不下单）

```bash
docker run --rm --name quant-hedge-multileg \
  -p 9191:9090 \
  --env-file /opt/quant-engine/live/binance_mainnet.env \
  -v /opt/quant-engine/live/shared_feature_bus:/app/live/shared_feature_bus \
  -v /opt/quant-engine/live/highcap/data:/app/live/highcap/data \
  -v /opt/quant-engine/data:/app/data \
  quant-engine:latest \
  python scripts/run_multi_leg_live.py \
    --mode testnet \
    --no-orders \
    --strategies chop_grid \
    --universe highcap \
    --symbols BTCUSDT \
    --bar-source feature-store \
    --feature-bus-root live/shared_feature_bus \
    --feature-store-timeframe 120T \
    --feature-store-execution-timeframe 1min \
    --poll-seconds 60 \
    --state-dir data/multi_leg_live/state
```

观察 **1～3 天**：`docker logs -f quant-hedge-multileg`，看 segment 开平、对账是否报错。

### 4.3 去掉 `--no-orders`（Testnet 小仓位真下单）

确认 Testnet 有余额后，同上命令 **去掉** `--no-orders`。

---

## 5. 阶段 C：Mainnet（小资金）

1. 在 `binance_mainnet.env` 填入 `MULTI_LEG_BINANCE_FUTURES_API_KEY/SECRET`（**专用子账户**，勿与 B 系统共用）。
2. `--mode mainnet`，去掉 `--no-orders`。
3. 确认 `live/highcap/config/constitution/constitution.yaml` 里 `multi_leg.strategies` 含 `chop_grid`。
4. IP 白名单绑定服务器出口 IP；Key **禁止提现**。

```bash
# 与 4.2 相同，仅改 --mode mainnet，且不要 --no-orders
--mode mainnet
```

---

## 6. 回测很好，但 live 可能不一样（必读）

| 点 | 说明 |
|----|------|
| **时间周期** | 生产 hedge 默认 **120T 信号 + 1min 执行**；你若回测用 `2h`，数字不会 1:1 相同 |
| **滑点 / 延迟** | 回测有费模型；实盘还有排队、部分成交 |
| **trend_scalp** | constitution **当前未启用**；勿因 trend 回测好就直接 mainnet trend |
| **风控** | live 有 `max_drawdown_pct`、`max_segment_starts_per_symbol_per_day` 等硬闸 |

建议：用 **与 live 相同参数** 再跑一遍回测对照：

```bash
python scripts/chop_grid_backtest.py \
  --timeframe 120T --execution-timeframe 1min \
  --start 2025-01-01 --end 2025-12-31 \
  --symbols BTCUSDT \
  --out-dir results/pre_live_validate/chop_120T_1min
```

---

## 7. 用 systemd 常驻（可选）

专家机用 systemd；你可把 §4 两条 `docker run` 写成 unit。参考：

- `.github/workflows/deploy.yml` 内 `quant-feature-bus.service`、`quant-hedge-multileg.service`
- `docs/deployment/LIVE_PRODUCTION_RUNBOOK_CN.md`

或先用 **tmux** 开两个窗格跑 §4.1 / §4.2，验证稳定后再上 systemd。

```bash
sudo apt install -y tmux
tmux new -s quant
# 窗格1: feature-bus  窗格2: hedge-multileg
# Ctrl+b d 脱离；tmux attach -t quant 回来
```

---

## 8. 常用检查

```bash
# 容器是否在跑
docker ps | grep quant-

# Feature Bus 是否有新特征
ls -lt /opt/quant-engine/live/shared_feature_bus/features/120T/

# 多腿审计日志
tail -f /opt/quant-engine/data/multi_leg_live/state/logs/multi_leg_audit.log

# 依赖检查（裸机 venv 时）
bash live/scripts/check_dependencies.sh highcap
```

---

## 9. 故障排查

| 现象 | 可能原因 |
|------|----------|
| hedge 空转无 bar | feature-bus 未启动或 `120T` parquet 未生成 |
| API 报错 | Key 错、IP 未白名单、未开 Hedge Mode |
| 国内连不上 WS | 需 HTTP/SOCKS 代理（在 env 或运维层配置） |
| 与回测差很多 | 信号周期不一致（2h vs 120T）、或样本区间不同 |

---

## 10. 相关文档

- 零基础总览：[`C系统零基础学习与验证指南_CN.md`](C系统零基础学习与验证指南_CN.md)
- 生产 Runbook：[`../deployment/LIVE_PRODUCTION_RUNBOOK_CN.md`](../deployment/LIVE_PRODUCTION_RUNBOOK_CN.md)
- chop 策略说明：[`../../config/strategies/chop_grid/README.md`](../../config/strategies/chop_grid/README.md)
- 一键启动脚本：[`../../scripts/ops/start_self_hosted_chop.sh`](../../scripts/ops/start_self_hosted_chop.sh)
- **命令速查**：[`本地Docker与Testnet命令手册_CN.md`](本地Docker与Testnet命令手册_CN.md)
