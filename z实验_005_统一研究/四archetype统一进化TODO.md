# 四 Archetype 统一进化计划

> 创建时间: 2026-02-17
> 最后更新: 2026-02-24
> 目标: BPC / ME / FER / LV 四策略完整进化 + PCM 动态分配 + 全量训练

---

## 📌 当前状态总览

| 领域 | 状态 | 说明 |
|------|------|------|
| 特征体系 (Phase 0-2) | ✅ 完成 | 7 基础 + 3 交叉 + OI 体系 |
| LV 配置 (Phase 3) | ✅ 完成 | 15min archetype 全套配置 |
| PCM 重构 (Phase 4) | ✅ 完成 | v2 严格性排序 |
| **PCM-宪法统一 (Phase 4.5)** | 🔨 部分完成 | 配置 ✅ + PCM回测 ✅ + 宪法模拟 📋延后 |
| 数据 (Phase 5) | ✅ 完成 | highcap symbols 数据齐全 |
| 语义预筛选 (Phase 5.5) | ✅ 完成 | BPC/ME/FER 均有 prefilter |
| 训练 (Phase 6) | ✅ BPC/ME/FER | 手动训练完成; LV 暂缓 (15min FS 太慢) |
| 多时间框架-研究 (Phase 7-R) | ✅ 完成 | ME→1H 配置完成 |
| **本地研究 pipeline** | ✅ 完成 | auto_research_pipeline.py --all + PCM 联合回测 (Step 9.5) |
| **实盘部署** | 🔨 进行中 | P0: 多时间框架实盘 → P1: 监控 → 腾讯云部署 |

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

## 🔨 Phase 4.5: PCM-宪法统一集成 — 设计完成，待实现

> 设计文档: `z实验_005_统一研究/PCM宪法统一架构完整设计文档.md`
> 目标: 解决 Constitution 与 PCM 配置重复/执行路径断裂/统计不记录等 6 个矛盾

### 4.5.1 配置统一 (Phase 1: 纯配置变更)

- [ ] `config/constitution/constitution.yaml` 添加 `resource_allocation` 段
  - `per_strategy_limits`: 每策略 max_slots + allow_add_position
  - `pcm_config_ref`: 指向 pcm_regime.yaml
- [ ] `config/pcm_regime.yaml` 升级 v3:
  - 移除 `max_slots` (从 constitution 读取)
  - 添加 `constitution_ref` 字段
- [ ] `live/highcap/config/constitution/constitution.yaml` 同步更新
- [ ] 验证: config/ 和 live/highcap/config/ 的 constitution.yaml 关键字段一致

### 4.5.2 ConstitutionExecutor 增强 (Phase 2)

- [ ] `constitution_executor.py` 新增 `validate_resource_allocation()` 方法
  - 检查 per_strategy slot 使用是否超限
  - 检查 archetype risk ≤ constitution risk_per_slot
- [ ] `enforcement.py`: `enforce_before_order()` 调用新增校验
- [ ] 测试: 单元测试验证 per_strategy_limits 约束

### 4.5.3 PCM 统一加载 (Phase 3)

- [ ] `live_pcm.py`: `LivePCM.__init__()` 接受 constitution_config
  - `max_slots` 从 constitution 读取
  - 启动时校验 `archetype_risk ≤ risk_per_slot`
- [ ] `run_live.py`: 加载顺序统一 (constitution → pcm)
- [ ] 测试: 校验一致性的单元测试

### 4.5.4 回测宪法模拟 (Phase 4)

- [ ] `backtest_execution_layer.py` `_run_pcm_mode()` 增强:
  - 新增 `--constitution <path>` 参数
  - 逐 bar 跟踪 equity curve → 计算 drawdown
  - 模拟 kill switch 逻辑 (dd > max_dd → 停止新入场)
  - 模拟 per_strategy slot 限制
- [ ] 测试: 构造 drawdown > 20% 场景验证 kill switch 模拟

### 4.5.5 PCM 统计输出 (Phase 5)

- [ ] `_run_pcm_mode()` 收集完整 PCM 统计到 dict
- [ ] 新增打印段: "PCM DECISION STATISTICS"
  - 仲裁摘要: 总信号/冲突数/Slot拒绝/宪法拒绝
  - Per-Archetype 仲裁: 信号数/获胜数/冲突胜率
  - Regime 分布: 各 regime 的 bar 数/入场数
  - 宪法模拟: 最大回撤/kill switch/损失突破次数
- [ ] 新增 `--pcm-stats-json <path>` 参数，输出结构化 JSON
- [ ] `parse_pcm_stats_stdout()` 解析器供 pipeline 使用

### 4.5.6 研究 Pipeline 集成 (Phase 6)

- [ ] `auto_research_pipeline.py` 新增 Step 9.5 (PCM 联合回测)
  - 扫描其他策略最新 predictions
  - 调用 `--pcm ... --pcm-stats-json` 联合回测
  - 解析 pcm_stats.json
- [ ] 扩展 `save_report()`: 包含 pcm_stats
- [ ] 扩展对比决策: 加入 PCM 联合 Sharpe / conflict_rate / constitution_sim
  - `conflict_rate > 0.15` → ALERT
  - `constitution_sim.kill_switch = true` → ERROR
  - `pcm_sharpe_daily < 1.0` → ALERT
- [ ] 快照: 保存 `pcm_regime_snapshot.yaml` 到实验目录
- [ ] 测试: dry-run 验证 Step 9.5 命令正确

### 4.5.7 配置一致性验证脚本 (Phase 7)

- [ ] 新增 `scripts/validate_constitution_pcm_consistency.py`
  - 检查 6 项一致性规则（见设计文档 §6.1）
  - 可纳入 CI
- [ ] 端到端验证: BPC + FER PCM 联合回测 → pcm_stats.json → 正确解析
- [ ] 运行所有现有测试确保无回归

### 4.5.8 Archetype 降级与恢复机制 (Phase 8)

> **目的**: 单个策略连续亏损时自动暂停该策略，不影响其他策略运行
> **设计来源**: `z实验_006_统一实盘/实盘监控系统设计.md` §B1 archetype_health

**两层风控共存**:
```
第一层: Archetype 级降级 (本 Phase 实现)
  ├─ 作用范围: 单个策略
  ├─ 触发: 连亏 N 笔 → pause_archetype (该策略不再产生新信号)
  ├─ 已有持仓: 正常管理 (不强平)
  └─ 其他策略: 不受影响

第二层: 账户级 Kill Switch (已实现, constitution.yaml)
  ├─ 作用范围: 全账户
  ├─ 触发: daily -4% / weekly -8% / max_dd -20%
  └─ 不可违反的底线
```

#### 8.1 配置: `constitution.yaml` 新增 `archetype_degradation` 段

- [ ] `config/constitution/constitution.yaml` 新增:
  ```yaml
  archetype_degradation:
    enabled: true
    default:
      max_consecutive_losses: 5    # 连亏 5 笔 → 暂停
      review_after_hours: 24       # 24h 后允许恢复
    per_strategy:
      lv:  { max_consecutive_losses: 3 }  # LV 波动大，3 笔就停
      fer: { max_consecutive_losses: 4 }  # FER 中等
      me:  { max_consecutive_losses: 5 }  # ME 盈亏比型，容忍更多
      bpc: { max_consecutive_losses: 5 }  # BPC 骨架策略
  ```
- [ ] `live/highcap/config/constitution/constitution.yaml` 同步

#### 8.2 Runtime 追踪: SQLite `archetype_loss_tracker` 表

- [ ] 复用现有 safety_state SQLite (constitution_executor 已有 persist_to)
- [ ] 新增表:
  ```sql
  CREATE TABLE archetype_loss_tracker (
    strategy TEXT PRIMARY KEY,        -- bpc / me / fer / lv
    consecutive_losses INTEGER DEFAULT 0,
    consecutive_wins INTEGER DEFAULT 0,
    last_trade_time TEXT,
    last_trade_pnl REAL,
    paused INTEGER DEFAULT 0,         -- 0=active, 1=paused
    paused_at TEXT,
    paused_reason TEXT,
    total_trades INTEGER DEFAULT 0
  );
  ```
- [ ] 每笔交易完成后更新: pnl < 0 → losses++, 否则 reset
- [ ] consecutive_losses >= threshold → SET paused = 1

#### 8.3 Enforcement: 接入 LivePCM 决策链

- [ ] `live_pcm.py`: `decide()` 前检查 `archetype_loss_tracker.paused`
  - paused=1 → 跳过该策略信号 (log WARNING)
  - paused=0 → 正常走仲裁
- [ ] `order_flow_listener.py`: 交易完成回调 → 更新 tracker
- [ ] Telegram 通知: 暂停时发送 WARNING + 恢复命令提示

#### 8.4 恢复机制: CLI 命令 (非 CI, 非 API)

> **设计决策**: 不走 CI 重新提交代码 (太慢, 且恢复交易不是配置变更)
> **设计决策**: 不建 REST API (当前无 API 层, 安全面太大)
> **方案**: CLI 命令直接修改 SQLite 状态, 秒级生效
> **扩展路径**: 未来 Telegram 机器人调用同一 CLI = 免费获得"在线恢复"

- [ ] 新增 `scripts/manage_archetype_health.py`:
  ```bash
  # 查看所有策略降级状态
  python scripts/manage_archetype_health.py --status
  
  # 恢复某个策略 (清零 consecutive_losses, paused=0)
  python scripts/manage_archetype_health.py --resume fer
  
  # 手动暂停某个策略
  python scripts/manage_archetype_health.py --pause fer --reason "manual review"
  
  # 查看某策略的交易历史摘要
  python scripts/manage_archetype_health.py --history fer --last 20
  ```
- [ ] 恢复前安全检查:
  - `review_after_hours` 是否已满足 (不满足则 WARNING 但允许 --force)
  - 可选: 自动跑 `check_need_retrain.py --strategy X` 快检
- [ ] 所有操作写日志 (`logs/archetype_health.log`)

#### 8.5 测试

- [ ] 单元测试: 构造连亏 N 笔场景 → 验证 paused=1
- [ ] 单元测试: CLI --resume → 验证 paused=0, losses 清零
- [ ] 集成测试: LivePCM.decide() 跳过 paused 策略
- [ ] 边界: 同时 2 个策略 paused + 1 个 active → 系统仍运行

### 4.5 实施计划

| 周 | Phase | 内容 | 验证 |
|----|-------|------|------|
| W1 | 1-2 | 配置统一 + ConstitutionExecutor 增强 | 运行现有测试无回归 |
| W2 | 3-4 | PCM 统一加载 + 回测宪法模拟 | BPC+FER 联合回测验证 |
| W3 | 5-6 | 统计输出 + Pipeline 集成 | 端到端 dry-run + 实际运行 |
| W4 | 7 | 验证 + 文档 | 全量测试通过 |
| W5 | 8 | Archetype 降级 + 恢复 CLI | 连亏暂停 + CLI 恢复测试 |

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
- [x] Pipeline 端到端验证: `--all` 三策略 ADOPT + PCM PASS (2026-02-24)
- [x] DEPLOY 脚本: `scripts/deploy_config_to_live.py` (diff + deploy + git-commit + rollback)
- [x] **Step 9.5 PCM 联合回测**: 已实现 (conflict_rate=3.42%, sharpe_daily=28.79)
  - 自动扫描其他策略最新 predictions
  - 调用 `backtest_execution_layer.py --pcm --pcm-stats-json --constitution`
  - PCM 决策统计输出到 `pcm_stats.json`
  - 宪法模拟结果纳入决策
  - 详见 Phase 4.5.5 / 4.5.6

### A.2 研究待验证项

- [ ] ME@1H vs ME@4H 对比回测 (RR / Sharpe / 与 BPC 正交性)
- [ ] ME labels 适配 1H: forward_bars / max_holding_bars 是否需要重算
- [ ] PCM 联合回测: BPC + ME + FER 三策略联合 Sharpe / 冲突率 → **纳入 Step 9.5 自动化 (Phase 4.5)**

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
  ├── pcm_stats.json                   ← NEW: PCM 联合回测统计
  ├── pcm_regime_snapshot.yaml          ← NEW: PCM 配置快照
  ├── comparison.json                  ← 与上次对比
  └── pipeline.log                     ← 运行日志
```

---

### A.5 研究 Pipeline 12 步训练链

```
Step 0:    Data Download + Convert (增量, 容错)
Step 1:    Feature Store Build
Step 2:    Prepare Only (features_labeled.parquet)
Step 3:    Prefilter Analyze (--promote → 实验目录)
Step 4:    Direction Validation (--promote → 实验目录)
Step 5:    Gate Optimize (--promote → 实验目录)
Step 6:    Evidence Optimize (--promote → 实验目录)
Step 7:    Entry Filter Optimize (--promote → 实验目录)
Step 8:    Execution Grid Optimize (--promote → 实验目录)
Step 9:    Single-Strategy Backtest
Step 9.5:  PCM Joint Backtest (NEW) ← 多策略联合 + pcm_stats.json + 宪法模拟
Step 10:   Export Training Baseline (容错)
  ↓
单策略决策: ADOPT → 实验 archetypes/ → config/strategies/{strategy}/archetypes/
              KEEP  → 保留实验, 不更新生产
              ALERT → 保留实验 + 告警
PCM 联合决策 (NEW):
  conflict_rate > 0.15          → ALERT (冲突率过高)
  constitution_sim.kill_switch  → ERROR (回测触发熔断)
  pcm_sharpe_daily < 1.0        → ALERT (PCM 组合 Sharpe 低)
  上述均通过                   → PASS
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
| **Phase 4.5: PCM-宪法统一** | 🔨 部分完成 | 配置统一 ✅ + PCM 联合回测 ✅ + 回测宪法模拟 📋延后 + 降级机制 📋延后 |
| Phase 5: 数据 | ✅ 完成 | highcap symbols 数据齐全 |
| Phase 5.5: 预筛选 | ✅ 完成 | BPC/ME/FER 均已配置 |
| Phase 6: 训练 | ✅ BPC/ME/FER | LV 暂缓 |
| Phase 7-R: 多TF研究 | ✅ 完成 | ME→1H 配置完成 |
| **Part A: 研究 pipeline** | 🔨 完善中 | 实验隔离已完成，待 PCM 联合回测集成 |
| **Part B: 实盘部署** | 🔨 进行中 | 详见 `z实验_006_统一实盘/实盘部署TODO.md` |
