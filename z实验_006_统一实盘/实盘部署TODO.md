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
| **多时间框架本地验证** | ✅ 完成 | 观察模式验证通过，已部署到远程 |
| 实盘监控 (服务器) | ✅ 代码完成 | StatsCollector + metrics_exporter + PCM funnel |
| 配置路径隔离 | ✅ 完成 | _infer_base_dir 重写，live/ 自包含，persist_to=data/db/ |
| 配置部署工具 | ✅ 完成 | deploy_config_to_live.py 支持 GLOBAL_CONFIGS (constitution + pcm_regime) |
| .gitignore 安全 | ✅ 完成 | live/*/data/ + *.db 排除，运行时数据不进 git |
| Terraform 基础设施 | ✅ 已搭建 | 腾讯云 ap-tokyo, 2vCPU/4GB, Docker + systemd |
| CI/CD 流程 | ✅ 已搭建 | GitHub Actions → Docker image → ghcr.io → 服务器 pull |
| **首次部署** | ✅ 完成 | 2026-02-25 Build + Deploy 成功，quant-engine active (running) |
| 监控脚本 (本地) | ✅ 完成 | weekly/monthly monitor + feature drift + retrain trigger |
| 实盘监控 (服务器) | ✅ 代码完成 | StatsCollector + metrics_exporter + PCM funnel |

---

# Phase 1: 本地验证（上线前必须完成）

> 目标: 在本地模拟实盘环境，验证多时间框架系统正确性

---

## ✅ 1.1 多时间框架实盘支持 (P0 — 已完成)

> **已解决**: `_setup_three_strategies()` 从 meta.yaml 动态读取各策略 timeframe
> BPC/FER→240T, ME→60T, 各策略独立 IncrementalFeatureComputer
> **代码位置**: `scripts/run_live.py` `_setup_three_strategies()`

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

- [x] `IncrementalFeatureComputer` 支持多 timeframe 输出
  - 方案 A (已实现): 每个 symbol 创建多个 fc 实例 (4H + 1H)
  - `compute_features_batch(bars, ticks, primary_timeframe)` 支持指定 timeframe
  - `extra_feature_computers` 字典存放额外 timeframe 的 FC 实例
  - 验证: ME 特征值 = 研究时 `features_labeled.parquet` 中同时间戳值 (±1% 偏差)

### 1.1.3 OrderFlowListener 多 timeframe 特征计算

- [x] `_compute_and_save_15min_features()` 支持多组 timeframe 特征输出
  - 已实现: 主 FC 计算 primary_tf + `extra_feature_computers` 计算额外 timeframe
  - 输出: `features_by_timeframe = {primary_tf: features, tf_me: extra_features}`
  - 存储: 15min 快照按 timeframe 分别保存

### 1.1.4 GenericLiveStrategy timeframe 绑定

- [x] `GenericLiveStrategy.__init__()` 接受 `primary_timeframe` 参数
  - BPC: primary_timeframe="240T"
  - ME:  primary_timeframe="60T"
  - FER: primary_timeframe="240T"
- [x] `GenericLiveStrategy.decide()` 由 LivePCM 路由对应 timeframe 特征

### 1.1.5 LivePCM 多 timeframe 决策路由

- [x] `LivePCM.register(name, strategy, timeframe="240T")` 记录每策略的 timeframe
- [x] `LivePCM.decide(features_by_timeframe, symbol)`:
  - 对每个注册策略，取 `features_by_timeframe[strategy.timeframe]`
  - 传给对应 strategy.decide()
  - 仲裁逻辑不变 (优先级 + evidence score)

### 1.1.6 run_live.py 升级

- [x] `_setup_three_strategies()` 改为多 timeframe:
  - 从 meta.yaml 动态读取: `tf_bpc=240T, tf_me=60T, tf_fer=240T`
  - `pcm.register("bpc", bpc, timeframe=tf_bpc)` 等
- [x] 每个 symbol 的 feature_computer_factory 返回支持多 timeframe 的计算器
  - primary FC = 4H, `listener.extra_feature_computers = {tf_me: me_fc}`
- [x] timeframe 从 meta.yaml 读取，无需硬编码环境变量

### 1.1.7 本地验证

- [x] 观察模式 (trade_size=0) 启动三策略 24h+
- [x] 确认日志中 ME 使用 60T 特征、BPC/FER 使用 240T 特征
- [x] 对比: ME 实盘特征 vs ME 研究特征 (同时间戳偏差 < 1%)

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

### 1.2.3 Telegram 告警通道 (→ 移至线上部署阶段)

> 移至 Phase 4 延后项，在线上运行稳定后实现

- [ ] 新增 `src/time_series_model/live/alerter.py`
  - CRITICAL: kill_switch 触发、数据源断开 → 即时通知
  - WARNING: 连续亏损、订单失败 → 15min 汇总
- [ ] Telegram Bot Token 配置到 `live/server.env`

### 1.2.4 本地验证

- [x] 观察模式运行 → 确认 stats_15min 表每 15min 有新记录
- [x] 确认信号漏斗数据合理 (gate reject rate 在 70-90% 范围)

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

## ✅ 2.1 GitHub 配置 — 已完成

### 2.1.1 创建 GitHub PAT (Personal Access Token)

- [x] 进入 GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- [x] 点击 "Generate new token (classic)"
- [x] 勾选权限:
  - `write:packages` (推送镜像到 ghcr.io)
  - `read:packages` (服务器拉取镜像)
- [x] 生成后复制 token（只显示一次！）

### 2.1.2 配置 GitHub Secrets

- [x] 进入仓库 → Settings → Secrets and variables → Actions → New repository secret
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

- [x] 本地测试: `ssh <DEPLOY_USER>@<DEPLOY_HOST> "echo ok"`
- [x] 如果是新机器，先手动 SSH 一次接受 host key

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

  ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 \
  'sudo journalctl -u quant-engine -f'

- [x] 执行后自动完成:
  - Docker 安装并启动
  - `/opt/quant-engine/live/highcap/data/{db,ticks,features_15min,features_4h}/` 目录创建
  - `quant-engine.service` systemd 服务配置并启用

### 2.2.2 Binance API 密钥

- [x] 已通过 GitHub Secrets 注入（`BINANCE_API_KEY` + `BINANCE_API_SECRET`）
- [x] 部署时 CI/CD 自动写入服务器 `/opt/quant-engine/live/binance_mainnet.env`
- [x] 确认 API key 权限: 只允许合约交易 + 只允许服务器 IP

### 2.2.3 安全组检查

- [x] SSH 端口限制为固定 IP
- [x] 确认无其他端口暴露（Grafana 3000 等后续按需开放）

---

## 2.3 首次部署 — ✅ 已完成 (2026-02-25)

### 2.3.1 触发首次构建

- [x] 方式 A: GitHub Actions → Run workflow
- [x] Build & Push Image: ✅ 镜像推送到 ghcr.io (~3min)
- [x] Deploy to Server: ✅ 镜像拉取 + 服务重启

### 2.3.2 Warmup 数据

> ℹ️ **改为本地下载 + rsync 上传**，服务器直接下载太慢（服务器带宽有限）
> `start_live.sh` 不再自动下载，未找到 warmup 数据则启动中止

**本地下载 + 上传流程**：
```bash
# 1. 本地下载 warmup 数据
bash live/scripts/prepare_warmup_ticks.sh highcap 6

# 2. 打包 + scp 上传（比 rsync 快很多，parquet 小文件多）
tar cf /tmp/warmup_ticks.tar -C live/highcap/data/ticks .
scp -i ~/.ssh/id_tencent_cloud_ssh /tmp/warmup_ticks.tar ubuntu@43.135.44.160:/tmp/

# 3. 远程解压 + 重启服务
ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 \
  'tar xf /tmp/warmup_ticks.tar -C /opt/quant-engine/live/highcap/data/ticks/ && rm /tmp/warmup_ticks.tar && sudo systemctl restart quant-engine'
```

- [x] 本地下载完成
- [x] tar 打包 + scp 上传完成
- [x] 验证: `ls /opt/quant-engine/live/highcap/data/ticks/BTCUSDT/ | wc -l` 应有 ~180 个文件

### 2.3.3 服务状态

- [x] `quant-engine.service` active (running)
- [x] 确认 warmup 下载完成 (`WARMUP → NORMAL`)
- [x] 确认 WS 连接成功 (`✅ WebSocket connected`)
- [x] 确认三策略注册 (`Registered: bpc, me, fer`)

### 部署踩坑记录

| 问题 | 原因 | 修复 |
|------|------|------|
| Dockerfile 找不到 | `.gitignore` 排除了 `/docker/` | 加 `!/docker/Dockerfile.live` 例外 |
| pip 升级失败 | Ubuntu 24.04 PEP 668 + debian pip 无 RECORD | `rm EXTERNALLY-MANAGED` + `--ignore-installed` |
| PyWavelets 构建失败 | 1.4.1 无 Python 3.12 wheel | 放宽到 `>=1.5.0` |
| Docker 权限拒绝 | ubuntu 用户不在 docker 组 | `usermod -aG docker ubuntu` |
| Warmup input() 崩溃 | systemd 无终端，EOFError | `auto_confirm=True` + EOFError 兜底 |
| LightGBM/sklearn 多余 | 实盘不用，只在训练代码 | 从 Dockerfile.live 移除 |
| Warmup 服务器下载太慢 | 服务器带宽有限，36 文件 ~25GB | 改为本地下载 + tar 打包 scp 上传 |

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

## ✅ 2.6 Funding Rate / OI 数据管线 — 已完成

> 所有策略 (BPC/ME/FER/LV) 的特征都依赖 `funding_rate_features_f` 和 `oi_features_f`。
> `.dockerignore` 排除 `data/`，Docker 镜像内无此数据，需外部提供。

### 2.6.1 方案A: 历史数据上传 + Volume Mount

- [x] 本地 tar 打包 6 个 live symbols 的 funding_rate + OI parquet (~53MB)
- [x] scp 上传到服务器 `/opt/quant-engine/data/`
- [x] `quant-engine.service` 添加 `-v /opt/quant-engine/data:/app/data` volume mount
- [x] 验证容器内 `/app/data/funding_rate/parquet/` 和 `/app/data/open_interest/parquet/` 有文件

### 2.6.2 方案B: 自动刷新机制

- [x] 新建 `scripts/refresh_funding_oi_data.py`:
  - Funding Rate: `/fapi/v1/fundingRate` 每8h一条, 最多1000条≈333天
  - OI: `/futures/data/openInterestHist` 5m粒度, **分段查询** (29天/段, API限制~30天)
  - 不需要 API key, 使用 Binance 公开端点
  - 增量合并 + 去重, 与研究 pipeline parquet 格式完全兼容
- [x] `start_live.sh` 第4步: 启动时自动刷新最近 60 天
- [x] `run_live.py`: 每 12 小时定时增量刷新 (asyncio task)

### 2.6.3 deploy.yml 修复

- [x] env 文件写入加 `sudo` (解决 root 权限问题)
- [x] `rm -rf` 防止 Docker mount 创建的目录残留
- [x] `.github/workflows/deploy.yml` 自身加入触发 paths

---

> 前置: 2.3 首次部署完成，quant-engine 正常运行

## ✅ 2.5 监控部署（Prometheus + Grafana）— 已完成

### 2.5.1 同步监控配置到服务器

- [x] 本地执行:
  ```bash
  rsync -avz -e "ssh -i ~/.ssh/id_tencent_cloud_ssh" \
    terraform/monitoring/ ubuntu@43.135.44.160:/opt/monitoring/
  ```
- [x] 确认文件同步: prometheus.yml + docker-compose + 3 个 dashboard JSON

### 2.5.2 启动监控容器

- [x] 本地执行:
  ```bash
  ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 \
    'sudo bash -s' < scripts/monitoring_bootstrap.sh
  ```
- [x] 执行后自动完成:
  - Prometheus 容器启动（端口 9091）
  - Grafana 容器启动（端口 3000）
  - Dashboard 自动加载（quant.json + account_market.json + signal_pipeline.json）

### 2.5.3 安全组放行监控端口

- [x] 腾讯云控制台 → 安全组:
  - 9091/tcp (Prometheus) — 限制为你的 IP
  - 3000/tcp (Grafana) — 限制为你的 IP

### 2.5.4 验证监控

- [x] 访问 Grafana: `http://43.135.44.160:3000` (admin/admin)
- [x] 确认 Prometheus target 状态: `http://43.135.44.160:9091/targets` → quant-engine UP
- [x] 内存占用: `docker stats --no-stream` 确认 Prometheus ~150MB + Grafana ~80MB

---

# Phase 3: 上线流程

> 前置: Phase 1 + Phase 2 全部完成

---

## 🔄 3.1 观察模式 → NORMAL 运行中

> ✅ 系统已从观察模式升级到 NORMAL 模式，监控正常工作
> 当前状态: 还未开单，等待信号触发

### 3.1.1 系统健康

- [x] WS 连接稳定
- [x] 内存/CPU 正常
- [x] 无 Python 异常/crash
- [x] 三策略都已注册 (`Registered: bpc, me, fer`)
- [x] Grafana 监控正常工作

### 3.1.2 特征一致性验证

> 待开单后验证，当前无信号无法对比

- [ ] 抽取实盘 15min 特征快照 (3-5 个时间戳)
- [ ] 对比偏差: 数值特征 < 1%, 分类特征完全一致

### 3.1.3 信号一致性验证

> 待开单后验证

- [ ] 收集实盘信号日志 + 对比研究代码重跑结果

### 3.1.4 无未来函数验证

> 待开单后验证

- [ ] 实盘 T 时刻信号 vs 研究回测 T 时刻信号对比

### 3.1.5 数据完整性

- [x] 1min bars 正常采集
- [x] tick 数据正常聚合
- [x] 15min 特征快照按时生成

### 3.1.6 实盘侧 Bug 修复 (2026-03-01)

- [x] `live_pcm.py`: PCM 跳过 timeframe 不匹配策略时清空 `_last_funnel`，避免漏斗统计读到上一次的陈旧结果

---

## 3.2 微量实盘 ($50-100 仓位, 20 笔)

> 目标：验证订单执行正确性，积累首批交易样本
> 资金：$1000 本金，每笔 $50-100 (5-10%)

### 3.2.1 执行质量验证

- [ ] 订单执行成功率 > 95%
- [ ] 实际成交价 vs 信号价：滑点 < 5bps
- [ ] 止损/止盈正确触发 (对比预期触发价位)
- [ ] 持仓时间分布合理 (与回测一致，非秒级进出)

### 3.2.2 首批样本统计 (20 笔)

> **统计学说明**：20 笔样本量较小，置信度有限，重点看方向性
> 真实胜率 60% 时，20 笔观测到胜率的 95% 置信区间约为 [36%, 81%]
> 所以 20 笔主要排除"系统性错误"，不能精确验证胜率

- [ ] 收集 20 笔完整交易 (含开仓/平仓/PnL)
- [ ] 统计：
  - 胜率 (期望 > 50%，若 < 40% 需停机排查)
  - 盈亏比 (期望 > 1.0)
  - 平均持仓时间 (与回测比较)
  - 最大单笔亏损 (应在止损范围内)
- [ ] PnL 方向与回测一致 (允许幅度差异)

### 3.2.3 通过标准

| 指标 | 红线 (停机排查) | 黄线 (继续观察) | 绿灯 (可加仓) |
|------|----------------|----------------|----------------|
| 胜率 (20笔) | < 35% | 35-50% | > 50% |
| 盈亏比 | < 0.5 | 0.5-1.0 | > 1.0 |
| 最大单笔亏损 | > 10% 本金 | 5-10% | < 5% |
| 订单失败率 | > 10% | 5-10% | < 5% |
| 信号vs回测方向 | 完全相反 | 部分偏差 | 基本一致 |

---

## 3.3 阶梯加仓计划

> 核心原则：**用时间和样本量换信心，分阶段放大风险敞口**
> 本金 $1000 → 总亏损上限 $200 (20%) → 触发全局止损
> ℹ️ PCM 已实现 evidence slot 竞争 + slot 内加仓逻辑，风控底线: slot=3 × risk_per_slot=0.01 = 6%

### 阶段设计

| 阶段 | 笔数区间 | 单笔仓位 | 累计最大亏损 | 进入条件 |
|------|---------|---------|-------------|----------|
| S1 试水 | 1-20 | $50 (5%) | $100 (10%) | 3.1 观察模式通过 |
| S2 验证 | 21-50 | $100 (10%) | $150 (15%) | S1 胜率 > 50% + 盈亏比 > 1.0 |
| S3 正常 | 51+ | $150-200 (15-20%) | $200 (20%) | S2 累计盈利 > 0 |

### 加仓决策规则

- **升级条件** (S1→S2)：
  - 20 笔完成
  - 胜率 > 50%
  - 累计 PnL ≥ 0 (不亏即可)
  - 无系统性异常

- **降级/停机条件**：
  - 连续 5 笔亏损 → 暂停 24h，检查市场环境
  - 累计亏损达到阶段上限 → 回退到上一阶段仓位
  - 累计亏损达 $200 (20%) → 全局停机，重新评估

- **永远不做的事**：
  - 亏损后加大仓位 "追回来"
  - 跳过阶段直接上大仓位
  - 在没搞清亏损原因时继续交易

### 每周复盘 SOP

```bash
# 下载数据
bash live/scripts/download_monitor_data.sh --days 7
# 跑周频检查
python scripts/local_monitor_weekly.py --data data/live_latest.parquet --strategy me --baseline results/.../training_baseline.json
```

复盘内容：
- 本周胜率 / 盈亏比 / PnL
- 信号漏斗通过率 (gate reject rate 70-90% 正常)
- 最大回撤
- 是否需要调整阶段

---

## ✅ B.4 实盘假设监控 (Phase 8) — 已完成

> 依据: `系统每层假设与实盘监控.md`
> 已实现脚本 (本地可用):
>   - `scripts/local_monitor_feature_drift.py` — PSI/KS 特征漂移
>   - `scripts/local_monitor_weekly.py` — 周频快速检查
>   - `scripts/local_monitor_monthly.py` — 月频全层报告
>   - `scripts/export_training_baseline.py` — 训练基线导出
>   - `scripts/monitor_retrain.py` — 重训触发器

| 层 | 假设 | 关键指标 | 失效阈值 |
|----|------|----------|----------|
| L1 特征 | 统计规律保持 | feature_drift_zscore | > 3.0 连续 3 天 |
| L2 预筛选 | 有效过滤噪声 | prefilter_pass_rate | 偏离训练期 ±50% |
| L3 Gate | 正向 lift | gate_lift | < 1.2 |
| L4 Evidence | score↔R 相关 | evidence_r_correlation | Spearman < 0.05 |
| L5 Direction | 增加胜率 | direction_accuracy | < 55% (30日) |
| L6 Entry Filter | 提升质量 | entry_filter_lift | < 1.0 |
| L7 Execution Tier | 高tier优于低tier | per_tier_mean_r | T1 ≤ T3 连续 2 周 |
| L8 PCM | 被选 > 被拒 | counterfactual_r | 被拒 > 被选持续 1 周 |
| L9 宪法 | 安全不过限 | kill_switch_count | 月 > 3 次 |

### 实施清单

- [x] 接入实盘数据后验证监控脚本
- [x] 假设失效告警通道 (Grafana + Prometheus)
- [ ] 假设失效归因 SOP (待有实盘数据后完善)

---

## ✅ B.5 实盘性能监控 (Phase 9) — 已完成

### 延迟目标

| 环节 | 目标值 | 告警阈值 |
|------|--------|----------|
| Tick → 特征计算 | < 200ms | > 500ms |
| 推理 | < 100ms | > 300ms |
| PCM 仲裁 | < 50ms | > 150ms |
| 下单 | < 100ms | > 300ms |
| 端到端 | < 500ms | > 1000ms |

### 实施清单

- [x] 延迟打点埋入
- [x] 轻量监控看板 (Prometheus/Grafana)
- [x] 告警通道接入
- [x] 性能基线建立 (上线首周)
- [ ] 降级策略: 延迟过高时暂停非核心 archetype (延后)

---

# Phase 4: 延后项 (上线运行稳定后)

> 这些不阻塞上线，但有助于提升系统健康度

| 优先级 | 项目 | 说明 | 何时做 |
|--------|------|------|--------|
| P2 | Phase 4.5.8 Archetype 降级 | 连亏自动暂停，当前有账户级 kill switch 兜底 | 上线 2 周后 |
| P2 | Phase 4.5.4 回测宪法模拟 + 一致性检查 | 🔨 进行中：kill switch + 加仓 + 事件vs研究回测对比 | 当前任务 |
| P2 | Phase 4.5.4 回测一致性对齐 | ✅ pre-slot 已对齐(1895≈1879), slot 精度修复中 | 当前任务 |
| P3 | PCM Plateau 优化 | detection 阈值 + scale 因子，conflict_rate=3.42% 收益极低 | 收集 4 周实盘数据后 |
| P3 | LV 策略 | 15min timeframe，Feature Store 计算成本高 | 三策略稳定后 |
| P2 | Telegram 告警通道 | alerter.py + Telegram Bot (从 Phase 1.2.3 移入) | 线上运行稳定后 |
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
# 4. 本地下载 warmup + 打包上传
bash live/scripts/prepare_warmup_ticks.sh highcap 6
tar cf /tmp/warmup_ticks.tar -C live/highcap/data/ticks .
scp -i ~/.ssh/id_tencent_cloud_ssh /tmp/warmup_ticks.tar ubuntu@43.135.44.160:/tmp/
ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 \
  'tar xf /tmp/warmup_ticks.tar -C /opt/quant-engine/live/highcap/data/ticks/ && rm /tmp/warmup_ticks.tar'
# 5. 重启服务
ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 'sudo systemctl restart quant-engine'

# 监控部署
rsync -avz -e "ssh -i ~/.ssh/id_tencent_cloud_ssh" terraform/monitoring/ ubuntu@43.135.44.160:/opt/monitoring/
ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 'sudo bash -s' < scripts/monitoring_bootstrap.sh

# 日常迭代
git push origin main  # 自动触发: build → push → deploy → restart
```

## 架构决策记录

### ADR-001: 单进程架构（不拆分 WebSocket / 交易 / 监控）

> 决策时间: 2026-02-26

**决策**: 当前阶段不拆分模块，WebSocket 采集 + 特征计算 + 决策 + 下单保持单进程。

**原因**:
1. **重启影响可忽略**: 特征通过 `compute_features_batch()` 从磁盘批量计算（150天 bars + 8天 ticks），不依赖流式内存状态。重启丢失 1-2 分钟 tick 对 15 分钟决策周期无影响。
2. **启动恢复完善**: `start_live.sh` 自动执行 `prepare_warmup_ticks.sh --fill-gap` 补缺失数据，`_restore_state()` 从磁盘恢复 memory_window。
3. **监控已独立**: Prometheus / Grafana 是独立 Docker 容器，不受 quant-engine 重启影响。
4. **规模不需要**: 2核4G 服务器、十几个 token、15分钟决策周期，拆分增加 IPC 复杂度，收益为零。

**重新评估条件**: 当策略升级到 1 分钟级决策、或需要 tick 级实时信号时，考虑拆分 WebSocket 采集层。

---

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
| `scripts/refresh_funding_oi_data.py` | Funding Rate / OI 增量刷新 (Binance 公开 API) |
