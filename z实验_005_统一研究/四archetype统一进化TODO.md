# 四 Archetype 统一进化计划

> 创建时间: 2026-02-17
> 最后更新: 2026-02-22
> 目标: BPC / ME / FER / LV 四策略完整进化 + PCM 动态分配 + 全量训练

---

## 📌 当前状态总览

| 领域 | 状态 | 说明 |
|------|------|------|
| 特征体系 (Phase 0-2) | ✅ 完成 | 7 基础 + 3 交叉 + OI 体系 |
| LV 配置 (Phase 3) | ✅ 完成 | 15min archetype 全套配置 |
| PCM 重构 (Phase 4) | ✅ 完成 | v2 严格性排序 |
| 数据 (Phase 5) | ✅ 完成 | highcap symbols 数据齐全 |
| 语义预筛选 (Phase 5.5) | ✅ 完成 | BPC/ME/FER 均有 prefilter |
| 训练 (Phase 6) | ✅ BPC/ME/FER | 手动训练完成; LV 暂缓 (15min FS 太慢) |
| 多时间框架-研究 (Phase 7-R) | ✅ 完成 | ME→1H 配置完成 |
| **本地研究 pipeline** | 🔨 完善中 | auto_research_pipeline.py + 实验隔离 |
| **实盘部署** | 📋 规划中 | 多时间框架实盘 + 监控 + 部署脚本 |

---

# Part A: 本地研究 Pipeline TODO

> 目标: 完善本地研究全自动 pipeline，确保可重复、可比较、可追溯
> 完成后即可进入实盘部署

---

## ✅ Phase 0: 语义特征审查 (已完成)

### 0.1 已加入的 7 个高价值特征 (全策略)

| 特征节点 | 输出列 | BPC 语义 | ME 语义 | FER 语义 |
|----------|--------|----------|---------|----------|
| `funding_rate_features_f` | funding_rate, funding_rate_zscore_50 等 | 拥挤度/假突破风险 | 方向确认 | 单边过度=反转机会 |
| `funding_scene_semantic_scores_f` | funding_{compression,ignition,absorption,exhaustion}_score | compression→压缩确认 | ignition→点火确认 | exhaustion→力竭确认 |
| `garch_features_f` | garch_volatility, persistence, leverage_gamma, alpha, beta | persistence→压缩延续 | volatility→扩张环境 | leverage_gamma→不对称波动 |
| `fp_imbalance_scene_semantic_scores_f` | fp_imbalance_{compression,ignition,absorption,exhaustion}_score | compression→结构确认 | ignition→订单流确认 | exhaustion→力竭确认 |
| `vpin_scene_semantic_scores_f` | vpin_{compression,ignition,absorption,exhaustion}_score | compression→VPIN压缩 | ignition→知情交易确认 | exhaustion→信息力竭 |
| `vwap_position_f` | price_to_vwap_pct, price_to_vwap_ratio | 回踩锚点 | 远离VWAP=动能强 | 远离=均值回归压力 |
| `exhaustion_at_liquidity_void_f` | exhaustion_at_liquidity_void | guardrail(反向) | ⚠️ 反向指标但tree自学 | ✅✅ 完美匹配 |

**审查结论**: 所有 7 个特征对 3 个 archetype 都合理。`exhaustion_at_liquidity_void_f` 对 ME 是反向信号，但 tree model 会自动学到"高值→deny"。无需修改。

---

## ✅ Phase 1: 组合语义特征 (乘法交叉) — 已完成

### 1.1 设计原理

两个独立信息源同时指向同一场景 → 确认度更高。只做明确有语义意义的组合。

### 1.2 已实现交叉特征

| 交叉特征 | 公式 | 语义 | 主要 Archetype |
|----------|------|------|---------------|
| `dual_compression_f` | `funding_compression_score × vpin_compression_score` | 资金+VPIN双源压缩确认 | BPC |
| `dual_ignition_f` | `funding_ignition_score × fp_imbalance_ignition_score` | 资金+Footprint双源点火确认 | ME |
| `dual_exhaustion_f` | `funding_exhaustion_scene_score × vpin_exhaustion_scene_score` | 资金+VPIN双源力竭确认 | FER |

### 1.3 实现清单

- [x] 在 `utils_interaction_features.py` 添加 3 个 compute 函数
- [x] 在 `feature_dependencies.yaml` 注册 3 个特征节点
- [x] 编写测试: 功能 + 无未来函数 + 流式一致性
- [x] 加入所有策略的 `features_gate.yaml` 和 `features_evidence.yaml`
- [x] 验证 pipeline 可运行

---

## ✅ Phase 2: OI (Open Interest) 特征体系 — 已完成

### 2.1 OI 下载模块

参考 `src/data_tools/download_funding_rate.py` 模式。

- [x] 创建 `src/data_tools/download_open_interest.py`
  - 数据源: Binance `/futures/data/openInterestHist` (支持 5m/15m/1h/4h/1d)
  - 输出: `data/open_interest/parquet/<SYMBOL>_YYYY-MM_open_interest.parquet`
  - 支持: 增量下载 / 断点续传 / force 重下
- [x] 在 CLI (`src/cli/main.py`) 注册 `mlbot data download-open-interest` 命令
- [x] 编写测试: mock API + 基本功能验证

### 2.2 OI 特征计算模块

- [x] 创建 `src/features/time_series/open_interest_features.py`
  - `compute_open_interest_features_from_df`:
    - `oi_value` (原始 OI，张→USD: `oi * mark_price`)
    - `oi_change_pct` (OI 变化百分比)
    - `oi_zscore_50` (50 周期 z-score)
    - `oi_price_divergence` (OI 增 + 价格不动 = 危险)
  - asof join 到 kline bars (无未来函数)
  - 支持流计算 (IncrementalFeatureComputer 兼容)
- [x] 注册到 `feature_dependencies.yaml`
- [x] 编写测试: 功能 + look-ahead bias 检测 + 流式一致性

### 2.3 OI 场景语义特征

- [x] `compute_oi_scene_semantic_scores_from_df`:
  - `oi_compression_score`: OI平稳 + 价格压缩 → 蓄力
  - `oi_ignition_score`: OI快速增 + 价格移动 → 方向确认
  - `oi_absorption_score`: OI增 + 价格不动 → 吸收/横盘
  - `oi_exhaustion_score`: OI快速降 + 单边极端 → 平仓清算
- [x] 注册到 `feature_dependencies.yaml`
- [x] 编写测试

### 2.4 OI × Funding 交叉特征 (LV 核心)

- [x] `oi_stress_x_funding_extreme_f`: `oi_zscore_50 × funding_rate_abs_zscore_50`
  - 语义: OI高 + 资金费率极端 = 清算风险最高
- [x] `oi_divergence_x_garch_leverage_f`: `oi_price_divergence × garch_leverage_gamma`
  - 语义: OI-价格背离 + 杠杆效应 = 系统脆弱

---

## ✅ Phase 3: LV (Liquidation Vulnerability) Archetype — 已完成

### 3.1 LV 语义定义 (来自 lv.md)

```
Liquidation Risk ∝ 杠杆集中度 × 单边持仓比例 × 订单簿深度薄弱度
```

**核心特征轴** (不同于 BPC/ME/FER):
- OI 异常
- Funding 偏离
- Long/Short 比例 (暂缺数据)
- Orderbook 深度 (需 L2 数据)

**时间粒度**: 15min (比 4H 更短，捕捉快速清算事件)

### 3.2 配置创建清单

- [x] `config/strategies/lv/meta.yaml` — LV 元信息 (timeframe: 15T)
- [x] `config/strategies/lv/model.yaml` — 模型配置
- [x] `config/strategies/lv/labels.yaml` — 标签定义 (清算驱动的大幅移动)
- [x] `config/strategies/lv/labels_return_tree.yaml`
- [x] `config/strategies/lv/labels_rr_extreme.yaml`
- [x] `config/strategies/lv/backtest.yaml` — 回测配置 (15min bar)
- [x] `config/strategies/lv/features.yaml` — 全量特征
- [x] `config/strategies/lv/features_gate.yaml` — Gate 训练输入 (OI + FR + GARCH 为主)
- [x] `config/strategies/lv/features_evidence.yaml` — Evidence 训练输入
- [x] `config/strategies/lv/archetypes/gate.yaml` — Gate 规则 (待训练)
- [x] `config/strategies/lv/archetypes/evidence.yaml` — Evidence 规则 (待训练)
- [x] `config/strategies/lv/archetypes/entry_filters.yaml` — Entry Filter (待训练)
- [x] `config/strategies/lv/archetypes/execution.yaml` — Execution 参数
- [x] `config/strategies/lv/archetypes/direction.yaml` — 方向判断
- [x] `config/strategies/lv/archetypes/holding.yaml` — 持仓管理 (快进快出)

### 3.3 LV 独特设计

| 维度 | BPC/ME/FER (4H) | LV (15min) |
|------|-----------------|------------|
| 时间粒度 | 240T (4小时) | 15T (15分钟) |
| 核心因果轴 | 结构/能量/均衡偏离 | 杠杆脆弱性 |
| 信号频率 | 低-中 | 可能较高 |
| 持仓时间 | 数小时-数天 | 数分钟-数小时 |
| 风险特征 | 可预测 | 非线性/尾部 |
| PCM 角色 | 常规 slot | override 型 (清算事件可覆盖其他) |

---

## ✅ Phase 4: PCM 分配模块重构 — 已完成

### 4.1 重构前状态

- 固定优先级: FER > ME > BPC
- 固定 max_slots=2
- 无 regime 动态调整
- 无 KPI 评估

### 4.2 已实现

基于 `一个 "Archetype Slot 分配与覆盖逻辑".md` 的设计:

**阶段一: 只做优先级动态 (不做 budget 动态)**

- [x] 实现 RegimeDetector (3 个状态 + 防抖):
  - `NORMAL`: BPC > ME > FER > LV (常态)
  - `HIGH_VOL`: ME > BPC > FER > LV (高波动扩张)
  - `HIGH_LEVERAGE`: LV > FER > ME > BPC (高杠杆脆弱)
- [x] LV Override Logic:
  - LV 作为独立 15min 不参与常规 slot 竞争
  - 当清算信号触发时可 override
- [x] Regime 判断条件:
  - HIGH_VOL: `atr_percentile > 0.7`
  - HIGH_LEVERAGE: `oi_zscore > 1.5 AND funding_rate_abs_zscore > 2.0`
  - NORMAL: 默认
- [x] 防抖机制: `min_bars_in_regime=3`，防止频繁切换
- [x] YAML 配置: `config/pcm_regime.yaml`

### 4.3 KPI 评估模块

- [x] PCM 评估指标设计 (`scripts/evaluate_pcm_allocation.py`):
  - `conflict_rate`: 冲突信号占比 (低=策略互补)
  - `regime_switch_frequency`: Regime 切换频率 (不宜过高)
  - `per_archetype_contribution`: 各策略对总 Sharpe 的贡献
  - `counterfactual_loss`: 被拒信号的事后表现 (反事实分析)
  - `regime_stats`: 分 regime 统计各策略表现
- [x] 评估脚本: `scripts/evaluate_pcm_allocation.py` (429 lines, CLI 支持)

### 4.4 代码修改清单

- [x] `src/time_series_model/portfolio/live_pcm.py` — 添加 RegimeDetector + 动态优先级
- [x] `scripts/backtest_execution_layer.py` — 更新默认优先级 `["BPC", "ME", "FER", "LV"]`
- [x] `scripts/run_live.py` + `demo_three_strategies.py` — regime-aware PCM
- [x] `config/pcm_regime.yaml` — Regime 配置
- [x] 测试: 39/39 通过 (含 15 个新 regime 测试) + 4/4 smoke

---

## ✅ Phase 5: 数据准备 — 完成

> highcap symbols (BTC/ETH/BNB/SOL/XRP/ADA) 数据已齐全，无需全量 59 symbols

- [x] OI 下载器: `scripts/download_oi_from_data_vision.py` (308 lines)
- [x] 一键下载: `scripts/download_all_data.sh`
- [x] highcap symbols OI + Funding Rate 数据齐全 (2023-01 ~ 2026-02)

---

## ✅ Phase 5.5: 语义预筛选 (Semantic Pre-filter) — 完成

> 设计文档: `gate_v3_semantic_prefilter_TODO.md`
> 实验报告: `docs/architecture/gate_semantic_prefilter_design.md`

### 核心思想

同一组预筛选条件同时解决两个问题：
1. **训练噪声** → 剥离不相关样本，专属特征重获区分力
2. **archetype 分配** → 定义"哪些 bar 属于此 archetype"的语义边界

### 已实现的预筛选规则

| 策略 | 规则文件 | 实现状态 |
|------|----------|----------|
| BPC | `archetypes/gate.yaml` guardrails: `bpc_volume_compression_pct ≥ 0.3 AND price_position ≤ 0.9` | ✅ |
| ME | `archetypes/prefilter.yaml`: `atr_percentile ≥ 0.922` (P90, CV=0.70) | ✅ |
| FER | `archetypes/prefilter.yaml`: `trapped_longs ≥ 4.48 OR trapped_shorts ≥ 3.77` (any_of) | ✅ |
| LV | 待训练后确定 | ⛏️ 暂缓 |

### 关键实现

- [x] `train_strategy_pipeline.py` 支持 `any_of` OR 逻辑 prefilter
- [x] `loader.py` 支持 `any_of → De Morgan AND deny` guardrail 转换
- [x] `analyze_archetype_feature_stratification.py` 支持 OR 对检测 + AND 累积模拟
- [x] BPC/ME/FER 三策略 prefilter 均已配置并训练验证

---

## ✅ Phase 6: 训练 — BPC/ME/FER 完成

> BPC、ME、FER 均已手动训练完成
> LV 暂缓: 15min Feature Store 计算成本太高，上线后再处理

### 已完成

- [x] BPC (4H): 全流程训练 + Gate/Evidence/EntryFilter/Execution 优化
- [x] ME (1H): 全流程训练 + 优化
- [x] FER (4H): 全流程训练 + 优化 (trapped OR prefilter v3.0)
- [x] 自动化脚本: `scripts/train_all_archetypes.sh`
- [x] 自动研究流水线: `scripts/auto_research_pipeline.py` (实验隔离 + ADOPT/KEEP/ALERT 决策)

### 暂缓

- [ ] LV (15min): Feature Store 计算太慢，延后到上线阶段

---

## ✅ Phase 7-R: 多时间框架 — 研究路径完成

```
时间频谱 ↑

L3 (4H)  ───────────────  BPC     FER
          （结构突破）   （结构失败）

L2 (1H)  ───────────────  ME
          （动能推进）

L1 (15m) ───────────────  LV
          （流动性挤压）
```

- [x] ME `meta.yaml`: timeframe "240T" → "60T"
- [x] 构建 1H Feature Store layer (`unified_1h_2023_2025`)
- [x] ME 独立训练: `mlbot train final --strategy me --timeframe 60T`
- [x] `train_all_archetypes.sh` 添加 `TIMEFRAME_1H="60T"`

---

### A.1 研究 Pipeline 自动化

> 已实现: `scripts/auto_research_pipeline.py` (实验隔离 + ADOPT/KEEP/ALERT 决策)
> 配置: `config/research_pipeline.yaml`
> 命令文档: `z实验_006_统一实盘/本地研究pipeline命令.md`

- [x] 一键自动化: `python scripts/auto_research_pipeline.py --strategy fer`
- [x] 实验目录隔离: config 随实验走，不覆盖生产
- [x] 实验管理 CLI: `--list` / `--adopt` / `--diff` / `--no-adopt`
- [x] 确定性决策规则: Sharpe 比值驱动 ADOPT/KEEP/ALERT
- [ ] Pipeline 端到端验证: 跑一次完整 pipeline 确认无报错 → 见 `z实验_005_统一研究/快速启动命令.md` 第一节
- [x] DEPLOY 脚本: `scripts/deploy_config_to_live.py` (diff + deploy + git-commit + rollback)

### A.2 研究待验证项

- [ ] ME@1H vs ME@4H 对比回测 (RR / Sharpe / 与 BPC 正交性)
- [ ] ME labels 适配 1H: forward_bars / max_holding_bars 是否需要重算
- [ ] PCM 联合回测: BPC + ME + FER 三策略联合 Sharpe / 冲突率

### A.3 LV (暂缓)

> 15min Feature Store 计算成本过高，上线后再推进

- [ ] LV Feature Store 构建 (15min)
- [ ] LV 全流程训练
- [ ] LV prefilter 阈值确定

---

### A.4 研究 Pipeline 目录结构

```
config/strategies/{strategy}/          ← 研究模板 (git 管理)
  ├── archetypes/                      ← ADOPT 时更新
  │   ├── gate.yaml
  │   ├── evidence.yaml
  │   ├── entry_filters.yaml
  │   ├── execution.yaml
  │   ├── direction.yaml
  │   ├── prefilter.yaml
  │   └── holding.yaml
  ├── features.yaml / features_gate.yaml / features_evidence.yaml
  ├── labels*.yaml / model.yaml / meta.yaml / backtest.yaml
  └── prefilter.yaml (候选特征声明)

results/research_history/{strategy}/{YYYYMMDD_HHMMSS}/   ← 实验快照 (不进 git)
  ├── strategies/{strategy}/           ← 隔离的 config 副本
  │   └── archetypes/                  ← 所有 --promote 写这里
  ├── archetypes/                      ← 快照副本 (方便查看)
  ├── report.json                      ← metrics + 决策
  ├── comparison.json                  ← 与上次对比
  └── pipeline.log                     ← 运行日志
```

---

### A.5 研究 Pipeline 11 步训练链

```
Step 0:  Data Download + Convert (增量, 容错)
Step 1:  Feature Store Build
Step 2:  Prepare Only (features_labeled.parquet)
Step 3:  Prefilter Analyze (--promote → 实验目录)
Step 4:  Direction Validation (--promote → 实验目录)
Step 5:  Gate Optimize (--promote → 实验目录)
Step 6:  Evidence Optimize (--promote → 实验目录)
Step 7:  Entry Filter Optimize (--promote → 实验目录)
Step 8:  Execution Grid Optimize (--promote → 实验目录)
Step 9:  Backtest
Step 10: Export Training Baseline (容错)
  ↓
决策: ADOPT → 实验 archetypes/ → config/strategies/{strategy}/archetypes/
      KEEP  → 保留实验, 不更新生产
      ALERT → 保留实验 + 告警
```

---

# Part B: 实盘部署 TODO

> 目标: 研究 pipeline 验证通过后，部署到实盘环境
> 前置条件: Part A 完成 (至少 BPC + ME + FER pipeline 端到端通过)

---

## B.1 配置部署 (config → live)

> 当前: `run_live.py` 读取 `live/highcap/config/strategies/`
> 研究确认后需要复制: `config/strategies/ → live/highcap/config/strategies/`

- [x] DEPLOY 脚本: `scripts/deploy_config_to_live.py` (diff + deploy + git-commit + rollback)
- [ ] 部署时自动 git commit live/ 目录变更 → `--git-commit` 已支持
- [ ] 回滚机制: `--rollback` 指引 + `git revert` 可快速恢复 live/ 配置

```
研究实验 (results/research_history/) 
  → ADOPT → config/strategies/ (研究确认, git)
  → DEPLOY → live/highcap/config/strategies/ (生产部署, git)
```

---

## B.2 多时间框架实盘路径 (Phase 7-L)

> 研究路径已完成: BPC/FER→4H, ME→1H, LV→15min
> 实盘需要升级以支持多 timeframe 并行

### B.2.1 IncrementalFeatureComputer 多 timeframe 输出

- [ ] 方案 A（推荐）: 多次调用，复用现有代码

```python
features_4h = fc.compute_features_batch(bars, ticks, "240T")
features_1h = fc.compute_features_batch(bars, ticks, "60T")
features_15m = fc.compute_features_batch(bars, ticks, "15T")
```

### B.2.2 OrderFlowListener 多 timeframe 特征计算

- [ ] `_compute_and_save_15min_features()` 支持多组 timeframe 特征
- [ ] 按 archetype 的 primary_timeframe 路由对应特征

### B.2.3 PCM 多 timeframe 决策

- [ ] `LivePCM.decide()` 接收 `features_by_timeframe`
- [ ] 每个 strategy 用其对应 timeframe 的 features

```python
pcm.register("bpc", bpc, timeframe="240T")
pcm.register("me", me, timeframe="60T")
pcm.register("fer", fer, timeframe="240T")
pcm.register("lv", lv, timeframe="15T")
```

### B.2.4 run_live.py 升级

- [ ] `_setup_four_strategies()`: 每个策略独立 primary_timeframe
- [ ] 环境变量: `MLBOT_ME_BAR_MINUTES=60`, `MLBOT_LV_BAR_MINUTES=15`

### B.2.5 架构设计原则

```
时间频谱 ↑

L3 (4H)  ───────────────  BPC     FER
          （结构突破）   （结构失败）

L2 (1H)  ───────────────  ME
          （动能推进）

L1 (15m) ───────────────  LV
          （流动性挤压）

统一 15min 决策节奏 → 高周期只作 slow state feature / regime filter
```

---

## B.3 PCM 优先级验证 (Phase 4.5)

> 目标: 验证 v2 优先级 (LV>FER>ME>BPC) 相比 v1 有优势
> 设计文档: `PCM优先级简化设计.md`

- [ ] 历史 predictions 重跑 PCM 回测，v1 vs v2 冲突解决
- [ ] 反事实分析: v2 被拒信号事后 R 和胜率
- [ ] Regime 分层验证: HIGH_VOL 下 ME>FER 是否更优
- [ ] 实盘后每周检查 PCM 冲突日志

---

## B.4 实盘假设监控 (Phase 8)

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

- [ ] 接入实盘数据后验证监控脚本
- [ ] 假设失效告警通道
- [ ] 假设失效归因 SOP

---

## B.5 实盘性能监控 (Phase 9)

### 延迟目标

| 环节 | 目标值 | 告警阈值 |
|------|--------|----------|
| Tick → 特征计算 | < 200ms | > 500ms |
| 推理 | < 100ms | > 300ms |
| PCM 仲裁 | < 50ms | > 150ms |
| 下单 | < 100ms | > 300ms |
| 端到端 | < 500ms | > 1000ms |

### 实施清单

- [ ] 延迟打点埋入
- [ ] 轻量监控看板 (Prometheus/Grafana 或替代)
- [ ] 告警通道接入
- [ ] 性能基线建立 (上线首周)
- [ ] 降级策略: 延迟过高时暂停非核心 archetype

---

## 📊 进度追踪

| Phase | 状态 | 备注 |
|-------|------|------|
| Phase 0: 审查 | ✅ 完成 | 7 特征全部合理 |
| Phase 1: 组合特征 | ✅ 完成 | 3 个乘法交叉特征 |
| Phase 2: OI 体系 | ✅ 完成 | 下载器 + 特征 + 场景语义 + 交叉 |
| Phase 3: LV 配置 | ✅ 完成 | 15min archetype 全套配置 |
| Phase 4: PCM 重构 | ✅ 完成 | v2 严格性排序 |
| Phase 5: 数据 | ✅ 完成 | highcap symbols 数据齐全 |
| Phase 5.5: 预筛选 | ✅ 完成 | BPC/ME/FER 均已配置 |
| Phase 6: 训练 | ✅ BPC/ME/FER | LV 暂缓 |
| Phase 7-R: 多TF研究 | ✅ 完成 | ME→1H 配置完成 |
| **Part A: 研究 pipeline** | 🔨 完善中 | 实验隔离已完成，待端到端验证 |
| **Part B: 实盘部署** | 📋 规划中 | 等 Part A 完成后推进 |
