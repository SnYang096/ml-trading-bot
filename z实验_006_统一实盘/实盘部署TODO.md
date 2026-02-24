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
| **多时间框架实盘** | ❌ 阻塞 | run_live.py 只用单一 bar_minutes=240，ME(1H) 会被错误处理 |
| Terraform 基础设施 | ✅ 已搭建 | 腾讯云 ap-tokyo, 2vCPU/4GB, Docker + systemd |
| 配置部署工具 | ✅ 完成 | deploy_config_to_live.py (diff + deploy + git-commit) |
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

## 🟡 1.2 实盘监控基础设施 (P1)

> **依赖**: 1.1 完成后才能产生有意义的监控数据
> **参考**: `实盘监控系统设计.md` Part B

### 1.2.1 15min 统计快照 (心理安抚核心)

- [ ] 新增 `src/time_series_model/live/stats_collector.py`
  - 信号漏斗计数: direction → prefilter → gate → evidence → entry → pcm → order
  - 按策略分层统计 (bpc/me/fer)
  - 持仓状态快照
  - 系统健康指标 (CPU/内存/WS 状态)
- [ ] 写入 SQLite `data/live_monitor.db` 的 `stats_15min` 表
- [ ] 自动清理 > 30 天数据

### 1.2.2 特征快照 retention

- [ ] `Feature15MinStorage` / `Feature4HStorage` 添加 retention 清理
  - 保留最近 30 天，自动删除旧文件
  - 实现: 简单的 `cleanup_old_files(days=30)` 方法

### 1.2.3 Telegram 告警通道 (可选，上线前非必须)

- [ ] 新增 `src/time_series_model/live/alerter.py`
  - CRITICAL: kill_switch 触发、数据源断开 → 即时通知
  - WARNING: 连续亏损、订单失败 → 15min 汇总
- [ ] Telegram Bot Token 配置到 `live/server.env`

### 1.2.4 本地验证

- [ ] 观察模式运行 → 确认 stats_15min 表每 15min 有新记录
- [ ] 确认信号漏斗数据合理 (gate reject rate 在 70-90% 范围)

---

## 🟡 1.3 配置部署验证 (P1)

### 1.3.1 研究配置 → 生产配置

- [ ] 运行 `python scripts/deploy_config_to_live.py --diff` 检查差异
- [ ] 确认 `live/highcap/config/strategies/` 下三策略配置是最新 ADOPT 版本
- [ ] 确认 `live/highcap/config/constitution/constitution.yaml` 与研究一致

### 1.3.2 启动命令统一

- [ ] 更新 `z实验_006_统一实盘/实盘启动命令.md`:
  - 从 `run_three_strategies_live.py` 改为 `run_live.py`
  - 添加 `MLBOT_ME_BAR_MINUTES=60` 环境变量
  - 更新 PCM 优先级说明 (LV > FER > ME > BPC)

---

# Phase 2: 腾讯云部署

> 前置: Phase 1 本地验证全部通过
> 基础设施: Terraform 已搭建 (ap-tokyo, 2vCPU/4GB, Docker)

---

## 2.1 Terraform 配置更新

### 当前状态
- ✅ VPC + Subnet + Security Group
- ✅ CVM 实例 (Ubuntu 22.04, 2vCPU/4GB, 256GB 数据盘)
- ✅ Docker + systemd 服务
- ✅ Prometheus + Grafana + Filebeat 监控栈
- ⚠️ `quant-engine.service` 入口为 `main.py`，需改为 `run_live.py`

### 2.1.1 systemd 服务更新

- [ ] `terraform/systemd/quant-engine.service` 更新:
  - `ExecStart` 改为正确的启动命令
  - 添加多 timeframe 环境变量
  - 添加 constitution/pcm 配置路径
  ```ini
  Environment=MLBOT_LIVE_SYMBOLS=BTCUSDT,ETHUSDT
  Environment=MLBOT_STRATEGIES_ROOT=/opt/quant-engine/live/highcap/config/strategies
  Environment=MLBOT_BPC_BAR_MINUTES=240
  Environment=MLBOT_ME_BAR_MINUTES=60
  Environment=MLBOT_BPC_WINDOW_MINUTES=15
  Environment=MLBOT_LIVE_TRADE_SIZE=0.0
  Environment=MLBOT_LIVE_USE_FUTURES=true
  Environment=MLBOT_PCM_REGIME_CONFIG=/opt/quant-engine/config/pcm_regime.yaml
  Environment=MLBOT_CONSTITUTION_YAML=/opt/quant-engine/live/highcap/config/constitution/constitution.yaml
  ```

### 2.1.2 init.sh 补充

- [ ] Python 环境安装 (如果不用 Docker):
  - Python 3.10+ / pip / requirements.txt
- [ ] 或确认 Dockerfile.live 可正常构建

### 2.1.3 安全组检查

- [ ] SSH 端口限制为固定 IP (当前 0.0.0.0/0 不安全)
- [ ] 确认无其他端口暴露 (Grafana 3000 端口决策: 关闭 or VPN)

---

## 2.2 CI/CD 流程（本地→远程）

> ⚠️ 目前尚未搭建，需要优先完成

### 2.2.1 CI/CD 方案选型

- [ ] 确定 CI/CD 工具: GitHub Actions / GitLab CI / 手动脚本
- [ ] 确定制品形式: Docker image / git pull + pip install / rsync
- [ ] 确定触发方式: push to main 自动部署 / 手动触发 / tag 触发

### 2.2.2 构建流程

- [ ] 定义构建步骤:
  - 安装依赖
  - 运行测试 (`pytest tests/`)
  - 构建 Docker image (如果用 Docker)
- [ ] 确保 `order_management.db` 等运行时数据文件不会被打包/覆盖
  - `.gitignore` 已排除 `live/*/data/` 和 `*.db` ✅
  - Docker image / rsync 也需显式排除

### 2.2.3 部署流程

- [ ] 代码/镜像推送到远程服务器
- [ ] 远程服务器拉取最新代码/镜像
- [ ] 运行 `deploy_config_to_live.py --deploy` 同步配置（仅 yaml，不含 db）
- [ ] 重启服务 (`systemctl restart quant-engine`)
- [ ] 健康检查: 确认进程启动 + WS 连接成功

### 2.2.4 安全约束

- [ ] 远程 `live/highcap/data/` 目录下的运行时数据（db、日志）不得被部署覆盖
- [ ] API key / secret 通过 `server.env` 管理，不进 git
- [ ] 部署前自动备份远程 db 文件

---

## 2.3 代码部署（CI/CD 就绪后执行）

### 2.3.1 部署方式选择

- [ ] **方式 A (推荐): Git clone + requirements.txt**
  ```bash
  # 服务器上
  git clone <repo> /opt/quant-engine
  cd /opt/quant-engine
  pip install -r requirements.txt
  ```
- [ ] **方式 B: Docker image**
  - 构建 `Dockerfile.live` (只装依赖，代码 volume 挂载)
  - Push 到腾讯云 TCR 或 DockerHub

### 2.3.2 配置文件部署

- [ ] `config/` 目录完整上传
- [ ] `live/highcap/config/` 目录完整上传
- [ ] `live/server.env` 配置实际服务器信息

### 2.3.3 数据准备

- [ ] warmup 数据上传:
  ```bash
  # 本地准备
  bash live/scripts/prepare_warmup_ticks.sh highcap 6 --from-local
  # 上传到服务器
  rsync -avz live/highcap/data/ server:/opt/quant-engine/live/highcap/data/
  ```
- [ ] 或服务器直接下载:
  ```bash
  bash live/scripts/prepare_warmup_ticks.sh highcap 6
  ```

### 2.3.4 API 密钥配置

- [ ] `.env` 文件配置 Binance API key/secret
- [ ] 确认 API key 权限: 只允许合约交易 + 只允许服务器 IP
- [ ] 测试 API 连通性

---

## 2.4 Terraform 执行

```bash
# 1. 配置凭证
source config/local/qcloud.env

# 2. Plan
cd terraform
terraform plan

# 3. Apply
terraform apply

# 4. 验证
ssh ubuntu@<server_ip> "systemctl status quant-engine"
```

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

## Terraform 部署命令

```bash
source config/local/qcloud.env
cd terraform
terraform plan
terraform apply
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/run_live.py` | 实盘入口 (多策略) |
| `config/pcm_regime.yaml` | PCM Regime 仲裁配置 |
| `config/constitution/constitution.yaml` | 宪法硬约束 |
| `live/highcap/config/strategies/` | 生产策略配置 |
| `scripts/deploy_config_to_live.py` | 研究→生产配置部署 |
| `terraform/main.tf` | 云基础设施定义 |
| `terraform/systemd/quant-engine.service` | systemd 服务定义 |
