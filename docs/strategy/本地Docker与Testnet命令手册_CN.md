# 本地 Docker 与 Testnet 命令手册

> **适用**：Mac 本机验证 C 系统（`chop_grid` 为主；`trend_scalp` 研究态）。  
> **前提**：已 `git clone` 本仓库；`data/parquet_data` 有历史数据（Shadow 建议有）。  
> **相关**：[零基础指南](C系统零基础学习与验证指南_CN.md) · [自建服务器部署](自建服务器部署_chop_grid_CN.md)

---

## 0. 路径变量（每条终端先执行）

```bash
cd /Users/jerry/project/yin/ml-trading-bot
export REPO="$(pwd)"
export DEPLOY_ROOT="$REPO"    # 本机即仓库根；上服务器可改为 /opt/quant-engine
```

---

## 1. 启动前检查

```bash
# Docker Desktop 已打开（菜单栏鲸鱼为运行状态）
docker ps

# 历史数据（Shadow / warmup 用）
ls data/parquet_data/BTCUSDT_2025-*.parquet | head -3

# 可选：不用 Docker，本机 venv Shadow
.venv/bin/python scripts/run_multi_leg_live.py --mode shadow --bar-source parquet \
  --strategies chop_grid --symbols BTCUSDT --once
```

---

## 2. 构建 Docker 镜像（首次 / 代码更新后）

```bash
cd "$REPO"
docker build -f docker/Dockerfile.live -t quant-engine:latest .
```

预计 10～20 分钟。失败且提示 `Cannot connect to the Docker daemon` → 先打开 **Docker Desktop**。

---

## 3. 阶段 A：Shadow（不要币安 Key、不下单）

### 3.1 一条命令

```bash
docker run --rm -it \
  -v "$DEPLOY_ROOT/data:/app/data" \
  -v "$DEPLOY_ROOT/live:/app/live" \
  quant-engine:latest \
  python scripts/run_multi_leg_live.py \
    --mode shadow \
    --bar-source parquet \
    --strategies chop_grid \
    --symbols BTCUSDT \
    --data-dir data/parquet_data \
    --once
```

### 3.2 封装脚本

```bash
DEPLOY_ROOT="$REPO" ./scripts/ops/start_self_hosted_chop.sh shadow
```

**成功标志**：进程结束 exit 0，日志里有 bar/segment 处理，无 traceback。

---

## 4. 阶段 B：本机回测（可选，不用 Docker）

### chop_grid

```bash
.venv/bin/python scripts/chop_grid_backtest.py \
  --start 2025-01-01 --end 2025-12-31 \
  --symbols BTCUSDT --timeframe 2h \
  --out-dir results/my_validate/chop_btc_2025
```

### trend_scalp（研究栈 hold_scaled）

```bash
.venv/bin/python scripts/diagnose_dual_add_trend.py \
  --config config/experiments/20260618_multileg_param_tune/variants/trend_hold_scaled.yaml \
  --symbols BTCUSDT --start 2025-01-01 --end 2025-12-31 \
  --timeframe 2h --execution-timeframe 1min \
  --no-initial-hedge --no-reseed-on-flip \
  --scale-max-loser-hold-to-signal \
  --out-dir results/my_validate/trend_btc_2025
```

---

## 5. 阶段 C：Testnet（需注册 + API Key）

### 5.1 注册与设置

1. 合约 Testnet：https://testnet.binancefuture.com  
2. 创建 API Key（合约交易 + 读取；**不要**开提现）  
3. 账户 → **双向持仓 Hedge Mode**  
4. 领取测试 USDT（页面 Faucet）

### 5.2 密钥文件（勿提交 git）

```bash
cat > "$REPO/live/testnet.env" <<'EOF'
MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY=填你的key
MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET=填你的secret
EOF
chmod 600 "$REPO/live/testnet.env"
```

---

## 6. 阶段 D：两进程（接近实盘）

**信号**：Feature Bus 用 **主网** WebSocket 行情（与生产一致）。  
**下单**：Testnet 假钱（与主网成交不一致，见 §9）。

### 终端 1 — Feature Bus（保持运行）

```bash
docker run --rm --name quant-feature-bus \
  -v "$DEPLOY_ROOT/data:/app/data" \
  -v "$DEPLOY_ROOT/live:/app/live" \
  quant-engine:latest \
  python scripts/run_market_feature_publisher.py \
    --symbols BTCUSDT,ETHUSDT \
    --feature-bus-root live/shared_feature_bus \
    --live-storage-base live/highcap/data \
    --strategies-root live/highcap/config/strategies \
    --warmup-days 7 \
    --max-rows 10080
```

检查特征是否写出：

```bash
ls -lt live/shared_feature_bus/features/120T/ 2>/dev/null | head
```

### 终端 2 — Testnet 只观察（不下单）

```bash
docker run --rm -it --name quant-hedge-multileg \
  --env-file "$REPO/live/testnet.env" \
  -e MLBOT_ACCOUNT_SCOPE=multi_leg \
  -v "$DEPLOY_ROOT/data:/app/data" \
  -v "$DEPLOY_ROOT/live:/app/live" \
  quant-engine:latest \
  python scripts/run_multi_leg_live.py \
    --mode testnet \
    --no-orders \
    --strategies chop_grid \
    --symbols BTCUSDT \
    --bar-source feature-store \
    --feature-bus-root live/shared_feature_bus \
    --feature-store-timeframe 120T \
    --feature-store-execution-timeframe 1min \
    --poll-seconds 60 \
    --state-dir data/multi_leg_live/state
```

### 终端 2 — Testnet 真下单（去掉 --no-orders）

同上，**删除** `--no-orders` 一行。

### 封装脚本

```bash
# 终端 1
DEPLOY_ROOT="$REPO" ./scripts/ops/start_self_hosted_chop.sh feature-bus

# 终端 2（先观察）
ENV_FILE="$REPO/live/testnet.env" DEPLOY_ROOT="$REPO" \
  ./scripts/ops/start_self_hosted_chop.sh multileg-testnet --no-orders
```

---

## 7. 阶段 E：Mainnet（真钱，确认 Testnet 稳定后再用）

```bash
cat > "$REPO/live/binance_mainnet.env" <<'EOF'
MULTI_LEG_BINANCE_FUTURES_API_KEY=专用子账户key
MULTI_LEG_BINANCE_FUTURES_API_SECRET=专用子账户secret
EOF
chmod 600 "$REPO/live/binance_mainnet.env"
```

```bash
ENV_FILE="$REPO/live/binance_mainnet.env" DEPLOY_ROOT="$REPO" \
  ./scripts/ops/start_self_hosted_chop.sh multileg-mainnet
```

需先起 Feature Bus（§6 终端 1）。**当前 constitution 仅默认启用 chop_grid**。

---

## 8. 常用运维命令

```bash
# 查看容器
docker ps -a | grep quant-

# 停 Feature Bus
docker stop quant-feature-bus

# 看多腿审计日志
tail -f data/multi_leg_live/state/logs/multi_leg_audit.log

# 本机不用 Docker 的依赖检查
bash live/scripts/check_dependencies.sh highcap

# 生成 regime 教学图（真实 BTC 2025）
.venv/bin/python scripts/research/plot_trend_regime_btc_annotated.py --year 2025

# 研究配置同步到 live 目录（diff / deploy）
python scripts/deploy_config_to_live.py --diff --strategy chop_grid
python scripts/deploy_config_to_live.py --deploy --strategy chop_grid --yes
```

---

## 9. Testnet vs 生产：数据一致吗？

| 项目 | 一致？ |
|------|--------|
| API 用法、下单字段 | ✅ 基本一致 |
| **信号行情**（Feature Bus） | ✅ 本仓库用 **主网** `fstream.binance.com` |
| Testnet 订单簿价格 | ❌ 与主网常有偏差 |
| 成交、滑点、深度 | ❌ Testnet 更「假」 |
| 资金 | ❌ 虚拟 USDT |

**结论**：Testnet 验证 **程序与流程**；**不能**用 Testnet 盈亏推断主网收益。

---

## 10. 模式对照表

| 模式 | 需要 Key | 需要 Feature Bus | 是否下单 |
|------|----------|------------------|----------|
| `shadow` + `parquet` | 否 | 否 | 否 |
| `testnet` + `--no-orders` | 是 | 建议有 | 否 |
| `testnet` | 是 | 建议有 | 是（假钱） |
| `mainnet` | 是 | 是 | 是（真钱） |

---

## 11. 故障速查

| 现象 | 处理 |
|------|------|
| `Cannot connect to the Docker daemon` | 打开 Docker Desktop |
| hedge 空转无新 bar | 先起 `quant-feature-bus`，等 `features/120T/*.parquet` |
| API / IP 错误 | Testnet Key、IP 白名单、Hedge Mode |
| 国内 WS 连不上 | 网络/代理；Feature Bus 日志会报连接失败 |

---

## 12. 文档索引

| 文档 | 内容 |
|------|------|
| [C系统零基础学习与验证指南_CN.md](C系统零基础学习与验证指南_CN.md) | 概念、四阶段路线 |
| [自建服务器部署_chop_grid_CN.md](自建服务器部署_chop_grid_CN.md) | AWS / Lightsail |
| [config/strategies/chop_grid/README.md](../../config/strategies/chop_grid/README.md) | chop 策略说明 |
| [config/strategies/trend_scalp/TREND_SCALP_逻辑导读_CN.md](../../config/strategies/trend_scalp/TREND_SCALP_逻辑导读_CN.md) | trend 逻辑 |
| [docs/deployment/LIVE_PRODUCTION_RUNBOOK_CN.md](../deployment/LIVE_PRODUCTION_RUNBOOK_CN.md) | 生产三进程 |

---

*路径默认 Mac 本机 `~/project/yin/ml-trading-bot`；换机器请改 `REPO` / `DEPLOY_ROOT`。*
