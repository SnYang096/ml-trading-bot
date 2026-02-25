# 实盘部署 TODO

> 创建时间: 2026-02-24
> 目标: 本地验证 → 腾讯云部署 → 观察模式 → 正式实盘
> 策略: BPC(4H) + ME(1H) + FER(4H)，LV 暂缓

---

## 📌 当前状态

| 领域 | 状态 | 说明 |
|------|------|------|
| 研究 Pipeline | ✅ 完成 | BPC/ME/FER 三策略 ADOPT + PCM PASS |
| PCM 联合回测 | ✅ 完成 | conflict_rate=3.42%, sharpe_daily=28.79 |
| 多时间框架研究 | ✅ 完成 | BPC/FER→4H, ME→1H |
| **多时间框架实盘代码** | ✅ 完成 | live_pcm/order_flow_listener/run_live.py 三文件改造，meta.yaml 动态 timeframe |
| **多时间框架本地验证** | ⏳ 待验证 | 观察模式 24h+ 运行确认 |
| 配置路径隔离 | ✅ 完成 | _infer_base_dir 重写，live/ 自包含，persist_to=data/db/ |
| 配置部署工具 | ✅ 完成 | deploy_config_to_live.py 支持 GLOBAL_CONFIGS (constitution + pcm_regime) |
| .gitignore 安全 | ✅ 完成 | live/*/data/ + *.db 排除，运行时数据不进 git |
| Terraform 基础设施 | ✅ 已搭建 | 腾讯云 ap-tokyo, 2vCPU/4GB, Docker + systemd |
| CI/CD 流程 | ✅ 已搭建 | GitHub Actions → Docker image → ghcr.io → 服务器 pull |
| 监控脚本 (本地) | ✅ 完成 | weekly/monthly monitor + feature drift + retrain trigger |
| 实盘监控 (服务器) | 📋 设计完成 | 见 实盘监控系统设计.md，代码待实现 |

---

# Phase 1: 本地验证（上线前必须完成）

> 目标: 在本地模拟实盘环境，验证多时间框架系统正确性

---

## 🔴 1.1 多时间框架实盘支持 (P0 — 上线阻塞项)

> **问题**: `_setup_three_strategies()` 用单一 `bar_minutes=240` 创建所有 IncrementalFeatureComputer
> ME 研究是 1H(60T)，实盘却被喂 4H(240T) 特征 → 信号完全错误
> **代码位置**: `scripts/run_live.py` L152-205

### 1.1.1 理解当前实盘数据流

当前架构 (单 timeframe):
```
BinanceWS tick → OrderFlowListener
  → 每 15min: IncrementalFeatureComputer.compute_features(bar_minutes=240)
  → features dict → PCM.decide() → 所有策略用同一套 240T 特征
```

目标架构 (多 timeframe):
```
BinanceWS tick → OrderFlowListener
  → 每 15min: 
    fc.compute_features(bar_minutes=240) → features_4h  → BPC/FER 用
    fc.compute_features(bar_minutes=60)  → features_1h  → ME 用
  → features_by_timeframe → PCM.decide() → 每策略用对应 timeframe 特征
```

### 1.1.2 IncrementalFeatureComputer 多 timeframe

- [ ] `IncrementalFeatureComputer` 支持多 timeframe 输出
  - 方案 A (推荐): 每个 symbol 创建多个 fc 实例 (4H + 1H)
  - 方案 B: 单实例支持 `compute_features_batch(bars, ticks, timeframe)` 多次调用
  - 数据需求: 1min bars 是共享的，只是聚合粒度不同 (240 vs 60)
  - 验证: ME 特征值 = 研究时 `features_labeled.parquet` 中同时间戳值 (±1% 偏差)

### 1.1.3 OrderFlowListener 多 timeframe 特征计算

- [ ] `_compute_and_save_15min_features()` 支持多组 timeframe 特征输出
  - 当前: 只计算一组特征 (4H timeframe)
  - 目标: 输出 `{timeframe: features_dict}` 结构
  - 存储: 15min 快照按 timeframe 分别保存 (已有 Feature15MinStorage)

### 1.1.4 GenericLiveStrategy timeframe 绑定

- [ ] `GenericLiveStrategy.__init__()` 接受 `primary_timeframe` 参数
  - BPC: primary_timeframe="240T"
  - ME:  primary_timeframe="60T"
  - FER: primary_timeframe="240T"
- [ ] `GenericLiveStrategy.decide(features_by_timeframe)` 自动提取对应 timeframe 特征

### 1.1.5 LivePCM 多 timeframe 决策路由

- [ ] `LivePCM.register(name, strategy, timeframe="240T")` 记录每策略的 timeframe
- [ ] `LivePCM.decide(features_by_timeframe, symbol)`:
  - 对每个注册策略，取 `features_by_timeframe[strategy.timeframe]`
  - 传给对应 strategy.decide()
  - 仲裁逻辑不变 (优先级 + evidence score)

### 1.1.6 run_live.py 升级

- [ ] `_setup_three_strategies()` 改为多 timeframe:
  ```python
  bpc = GenericLiveStrategy("bpc", strategies_root, primary_timeframe="240T")
  me  = GenericLiveStrategy("me",  strategies_root, primary_timeframe="60T")
  fer = GenericLiveStrategy("fer", strategies_root, primary_timeframe="240T")
  
  pcm.register("bpc", bpc, timeframe="240T")
  pcm.register("me",  me,  timeframe="60T")
  pcm.register("fer", fer, timeframe="240T")
  ```
- [ ] 每个 symbol 的 feature_computer_factory 返回支持多 timeframe 的计算器
- [ ] 环境变量: `MLBOT_ME_BAR_MINUTES=60` (独立于 BPC_BAR_MINUTES)

### 1.1.7 本地验证

- [ ] 观察模式 (trade_size=0) 启动三策略
- [ ] 确认日志中 ME 使用 60T 特征、BPC/FER 使用 240T 特征
- [ ] 运行 24h+，确认无崩溃、特征计算无异常
- [ ] 对比: ME 实盘特征 vs ME 研究特征 (同时间戳偏差 < 1%)

---

## ✅ 1.2 实盘监控基础设施 (P1)

> **依赖**: 1.1 完成后才能产生有意义的监控数据
> **参考**: `实盘监控系统设计.md` Part B

### 1.2.1 15min 统计快照 (心理安抚核心)

- [x] 新增 `src/time_series_model/live/stats_collector.py`
  - 信号漏斗计数: direction → gate → entry_filter → evidence → pcm → order
  - 按策略分层统计 (bpc/me/fer)
  - 持仓状态快照
  - 系统健康指标 (CPU/内存 via psutil)
- [x] 写入 SQLite `live/highcap/data/db/live_monitor.db` 的 `stats_15min` 表
- [x] 自动清理 > 30 天数据
- [x] `GenericLiveStrategy.decide()` 添加 `_last_funnel` 漏斗跟踪
- [x] `LivePCM.decide()` 收集各策略漏斗数据 + `record_pcm_selected()`
- [x] `OrderFlowListener` 每 15min 触发 flush + 下单记录
- [x] `run_live.py` 创建 StatsCollector 并注入 PCM + listener

### 1.2.2 特征快照 retention

- [x] `Feature15MinStorage` / `Feature4HStorage` 添加 `cleanup_old_files(days=30)`
  - 保留最近 30 天，自动删除旧 parquet 文件
  - 每天触发一次 (在 `_flush_stats()` 中检查)

### 1.2.3 Telegram 告警通道 (可选，上线前非必须)

- [ ] 新增 `src/time_series_model/live/alerter.py`
  - CRITICAL: kill_switch 触发、数据源断开 → 即时通知
  - WARNING: 连续亏损、订单失败 → 15min 汇总
- [ ] Telegram Bot Token 配置到 `live/server.env`

### 1.2.4 本地验证

- [ ] 观察模式运行 → 确认 stats_15min 表每 15min 有新记录
- [ ] 确认信号漏斗数据合理 (gate reject rate 在 70-90% 范围)

---

## ✅ 1.3 配置部署验证 (P1) — 已完成

### 1.3.1 研究配置 → 生产配置

- [x] 运行 `python scripts/deploy_config_to_live.py --diff` 检查差异 → 全部同步，零差异
- [x] 确认 `live/highcap/config/strategies/` 下三策略配置是最新 ADOPT 版本
- [x] 确认 `live/highcap/config/constitution/constitution.yaml` 与研究一致
- [x] 确认 `live/highcap/config/pcm_regime.yaml` 与研究一致

### 1.3.2 启动命令统一

- [x] 更新 `z实验_006_统一实盘/实盘启动命令.md`:
  - 从 `run_three_strategies_live.py` 改为 `bash live/scripts/start_live.sh`
  - timeframe 从 meta.yaml 动态读取，无需硬编码环境变量
  - 更新 PCM 优先级说明 (LV > FER > ME > BPC)
- [x] 更新 `三策略实盘就绪检查清单.md` 启动命令对齐

---

# Phase 2: 腾讯云部署

> 前置: Phase 1 本地验证全部通过
> 方案: GitHub Actions 构建 Docker 镜像 → ghcr.io → 服务器 pull
> 服务器只需 Docker，无需 Python/pip/venv

---

## 2.1 GitHub 配置

### 2.1.1 创建 GitHub PAT (Personal Access Token)

- [ ] 进入 GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- [ ] 点击 "Generate new token (classic)"
- [ ] 勾选权限:
  - `write:packages` (推送镜像到 ghcr.io)
  - `read:packages` (服务器拉取镜像)
- [ ] 生成后复制 token（只显示一次！）

### 2.1.2 配置 GitHub Secrets

- [ ] 进入仓库 → Settings → Secrets and variables → Actions → New repository secret
- [x] 添加以下 6 个 Secrets:

| Secret 名称 | 值 | 获取方式 |
|---|---|---|
| `DEPLOY_HOST` | 服务器公网 IP | 腾讯云控制台 → 云服务器 → 实例列表 |
| `DEPLOY_USER` | `ubuntu` | 服务器 OS 默认用户 |
| `DEPLOY_SSH_KEY` | SSH 私钥完整内容 | 本地 `cat ~/.ssh/id_tencent_cloud_ssh` |
| `GHCR_TOKEN` | GitHub PAT | 需 `read:packages` + `write:packages` |
| `BINANCE_API_KEY` | Binance API Key | Binance API Management |
| `BINANCE_API_SECRET` | Binance API Secret | Binance API Management |

### 2.1.3 验证 SSH 连通性

- [ ] 本地测试: `ssh <DEPLOY_USER>@<DEPLOY_HOST> "echo ok"`
- [ ] 如果是新机器，先手动 SSH 一次接受 host key

### 2.1.4 CI/CD 文件确认

- [x] `.github/workflows/deploy.yml` — 构建 + 推送 + 部署流水线
- [x] `docker/Dockerfile.live` — 生产镜像定义（代码打包进镜像）
- [x] `.dockerignore` — 排除 data/密钥/实验文档（7 类规则）

---

## 2.2 远程服务器配置

### 2.2.1 运行 Bootstrap 脚本（一次性）

- [x] 从本地执行（自动安装 Docker + 创建目录 + 配置 systemd）:
  ```bash
  ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 'sudo bash -s' < scripts/server_bootstrap.sh
  ```
- [x] 执行后自动完成:
  - Docker 安装并启动
  - `/opt/quant-engine/live/highcap/data/{db,ticks,features_15min,features_4h}/` 目录创建
  - `quant-engine.service` systemd 服务配置并启用

### 2.2.2 Binance API 密钥

- [x] 已通过 GitHub Secrets 注入（`BINANCE_API_KEY` + `BINANCE_API_SECRET`）
- [x] 部署时 CI/CD 自动写入服务器 `/opt/quant-engine/live/binance_mainnet.env`
- [ ] 确认 API key 权限: 只允许合约交易 + 只允许服务器 IP

### 2.2.3 安全组检查

- [ ] SSH 端口限制为固定 IP（当前 0.0.0.0/0 不安全，腾讯云控制台修改）
- [ ] 确认无其他端口暴露（Grafana 3000 等后续按需开放）

---

## 2.3 首次部署

### 2.3.1 触发首次构建

- [ ] 方式 A（推荐）: GitHub 仓库 → Actions 页面 → "Build & Deploy" → Run workflow
- [ ] 方式 B: push 代码到 main 分支自动触发
- [ ] 确认 Actions 日志:
  - Build & Push Image: ✅ 镜像推送到 ghcr.io
  - Deploy to Server: ✅ 镜像拉取 + 服务重启

### 2.3.2 服务器上运行 Warmup 数据下载

> **不需要从本地上传** — 服务器（东京）直连 Binance 比本地 VPN 更快
> **不需要 Feature Store** — live 模式所有特征基于 ticks/bars 实时重算

- [ ] SSH 到服务器，通过 Docker 容器执行 warmup:
  ```bash
  ssh root@<SERVER_IP>
  docker run --rm \
    -v /opt/quant-engine/live/highcap/data:/app/live/highcap/data \
    quant-engine:latest \
    bash live/scripts/prepare_warmup_ticks.sh highcap 6
  ```
- [ ] 预期: 下载 6 个月 aggTrades → 转换为 1min ticks + bars
- [ ] 耗时: ~10-30 分钟（取决于网络和 symbol 数量）
- [ ] 验证: `ls /opt/quant-engine/live/highcap/data/ticks/BTCUSDT/ | wc -l` 应有 ~180 个文件

### 2.3.3 启动服务

- [ ] 启动:
  ```bash
  sudo systemctl start quant-engine
  ```
- [ ] 查看实时日志:
  ```bash
  sudo journalctl -u quant-engine -f
  ```
- [ ] 确认启动成功标志:
  - WS 连接成功 (`✅ WebSocket connected`)
  - warmup 阶段完成 (`WARMUP → NORMAL`，约 4h）
  - 三策略注册成功 (`Registered: bpc, me, fer`)

---

## 2.4 后续部署（日常迭代）

每次修改代码后:
```bash
# 本地 commit + push
git add . && git commit -m "fix: xxx" && git push origin main
# GitHub Actions 自动: build image → push ghcr.io → server pull → restart
# 无需 SSH 到服务器
```

手动重启（不重新构建）:
```bash
# GitHub Actions → Run workflow → 勾选 "skip_build"
# 或直接 SSH:
ssh root@<SERVER_IP> "sudo systemctl restart quant-engine"
```

### 关键安全保障

| 保护项 | 机制 |
|---|---|
| `order_management.db` | volume 挂载在服务器，镜像更新不影响 |
| `live/highcap/data/` | volume 挂载，持久化在 `/opt/quant-engine/live/highcap/data/` |
| API 密钥 | GitHub Secrets 注入 → CI/CD 自动写入服务器，不进镜像 |
| 回滚 | `docker pull ghcr.io/<repo>:sha-<旧commit>` + restart |

---

## 2.5 监控部署（Prometheus + Grafana）

> 前置: 2.3 首次部署完成，quant-engine 正常运行

### 2.5.1 同步监控配置到服务器

- [ ] 本地执行:
  ```bash
  rsync -avz -e "ssh -i ~/.ssh/id_tencent_cloud_ssh" \
    terraform/monitoring/ ubuntu@43.135.44.160:/opt/monitoring/
  ```
- [ ] 确认文件同步: prometheus.yml + docker-compose + 3 个 dashboard JSON

### 2.5.2 启动监控容器

- [ ] 本地执行:
  ```bash
  ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 \
    'sudo bash -s' < scripts/monitoring_bootstrap.sh
  ```
- [ ] 执行后自动完成:
  - Prometheus 容器启动（端口 9091）
  - Grafana 容器启动（端口 3000）
  - Dashboard 自动加载（quant.json + account_market.json + signal_pipeline.json）

### 2.5.3 安全组放行监控端口

- [ ] 腾讯云控制台 → 安全组:
  - 9091/tcp (Prometheus) — 限制为你的 IP
  - 3000/tcp (Grafana) — 限制为你的 IP

### 2.5.4 验证监控

- [ ] 访问 Grafana: `http://43.135.44.160:3000` (admin/admin)
- [ ] 确认 Prometheus target 状态: `http://43.135.44.160:9091/targets` → quant-engine UP
- [ ] 内存占用: `docker stats --no-stream` 确认 Prometheus ~150MB + Grafana ~80MB

---

# Phase 3: 上线流程

> 前置: Phase 1 + Phase 2 全部完成

---

## 3.1 观察模式 (trade_size=0)

- [ ] 启动: `MLBOT_LIVE_TRADE_SIZE=0.0` (只看不做)
- [ ] 运行 48h+
- [ ] 检查:
  - 信号漏斗数据合理
  - 三策略都有信号产生
  - ME 使用 1H 特征，BPC/FER 使用 4H 特征
  - 内存/CPU 在正常范围 (< 500MB / < 30%)
  - WS 连接稳定，无频繁重连
  - 宪法状态始终绿灯

## 3.2 微量实盘 (最小单位)

- [ ] 切换: `MLBOT_LIVE_TRADE_SIZE=0.001` (最小 BTC 交易量)
- [ ] 运行 1-2 周
- [ ] 检查:
  - 订单执行成功率 > 95%
  - 滑点在预期范围 (< 5bps)
  - 止损/止盈正确触发
  - PnL 与回测方向一致 (不要求绝对值一致)

## 3.3 正式运营

- [ ] 逐步增加 `trade_size` 到目标仓位
- [ ] 建立每日检查 SOP:
  - 查看 15min 统计快照
  - 确认三灯号状态 (Capital / Edge / Activity)
- [ ] 每周本地检查:
  ```bash
  # 下载数据
  bash live/scripts/download_monitor_data.sh --days 7
  # 跑周频检查
  python scripts/local_monitor_weekly.py --data data/live_latest.parquet --strategy me --baseline results/.../training_baseline.json
  ```

---

# Phase 4: 延后项 (上线运行稳定后)

> 这些不阻塞上线，但有助于提升系统健康度

| 优先级 | 项目 | 说明 | 何时做 |
|--------|------|------|--------|
| P2 | Phase 4.5.8 Archetype 降级 | 连亏自动暂停，当前有账户级 kill switch 兜底 | 上线 2 周后 |
| P2 | Phase 4.5.4 回测宪法模拟 | PCM 回测加 kill switch 模拟 | 上线后有数据时 |
| P3 | PCM Plateau 优化 | detection 阈值 + scale 因子，conflict_rate=3.42% 收益极低 | 收集 4 周实盘数据后 |
| P3 | LV 策略 | 15min timeframe，Feature Store 计算成本高 | 三策略稳定后 |
| P3 | 可视化 Dashboard | Streamlit/Flask 页面 | 有空时 |

---

# 快速参考

## 本地模拟实盘命令

```bash
cd /home/yin/trading/ml_trading_bot

# 观察模式 (不交易)
PYTHONPATH=. \
MLBOT_LIVE_SYMBOLS=BTCUSDT,ETHUSDT \
MLBOT_STRATEGIES_ROOT=live/highcap/config/strategies \
MLBOT_BPC_BAR_MINUTES=240 \
MLBOT_ME_BAR_MINUTES=60 \
MLBOT_BPC_WINDOW_MINUTES=15 \
MLBOT_LIVE_TRADE_SIZE=0.0 \
MLBOT_LIVE_USE_FUTURES=true \
MLBOT_LIVE_WARMUP_DAYS=30 \
MLBOT_ORDER_MODE=test \
MLBOT_PCM_REGIME_CONFIG=config/pcm_regime.yaml \
MLBOT_CONSTITUTION_YAML=live/highcap/config/constitution/constitution.yaml \
python scripts/run_live.py
```

# 快速参考

## 部署命令速查

```bash
# 首次部署
ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 'sudo bash -s' < scripts/server_bootstrap.sh  # 1. 初始化服务器
# 2. 配置 GitHub Secrets（6 个）
# 3. GitHub Actions → Run workflow（首次构建镜像）
# 4. warmup 数据已集成到启动流程，自动下载
sudo systemctl start quant-engine                             # 5. 启动服务

# 监控部署
rsync -avz -e "ssh -i ~/.ssh/id_tencent_cloud_ssh" terraform/monitoring/ ubuntu@43.135.44.160:/opt/monitoring/
ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 'sudo bash -s' < scripts/monitoring_bootstrap.sh

# 日常迭代
git push origin main  # 自动触发: build → push → deploy → restart
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/run_live.py` | 实盘入口 (多策略) |
| `docker/Dockerfile.live` | 生产 Docker 镜像定义 |
| `.github/workflows/deploy.yml` | CI/CD 流水线 |
| `scripts/server_bootstrap.sh` | 服务器初始化脚本 |
| `terraform/systemd/quant-engine.service` | systemd 服务定义 (Docker 模式) |
| `scripts/monitoring_bootstrap.sh` | 监控初始化脚本 (Prometheus + Grafana) |
| `terraform/monitoring/` | 监控配置 (prometheus.yml + docker-compose + dashboard) |
| `live/highcap/config/strategies/` | 生产策略配置 (53 个 YAML) |
| `scripts/deploy_config_to_live.py` | 研究→生产配置部署 |
