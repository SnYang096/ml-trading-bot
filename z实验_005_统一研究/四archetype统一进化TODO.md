# 四 Archetype 统一进化计划

> 创建时间: 2026-02-17
> 目标: BPC / ME / FER / LV 四策略完整进化 + PCM 动态分配 + 全量训练

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

## 🔄 Phase 5: 数据准备 — 进行中

### 5.0 数据源选择

> **注意**: Binance REST API (`/futures/data/openInterestHist`) 的 `startTime` 参数仅支持最近 ~30 天，
> 超过 40 天即返回 400 错误。因此改用 **Binance Data Vision (S3)** 作为 OI 历史数据源。

- 数据源: `https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{YYYY-MM-DD}.zip`
- 每个 ZIP 包含 5 分钟精度 CSV (288 rows/day)
- 列: `create_time, symbol, sum_open_interest, sum_open_interest_value, ...`

### 5.1 OI 数据下载

- [x] 创建 `scripts/download_oi_from_data_vision.py` (308 lines) — 从 Data Vision S3 下载
- [x] 创建 `scripts/download_all_data.sh` — 整合 OI + Funding Rate 一键下载
- [ ] 执行全量下载 (59 symbols × 37 months, 后台运行中)

```bash
# 一键下载 (OI + Funding Rate)
nohup bash scripts/download_all_data.sh > /tmp/download_all.log 2>&1 &

# 或仅下载 OI:
python scripts/download_oi_from_data_vision.py \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a \
  --start-date 2023-01-01 \
  --parquet-dir data/open_interest/parquet
```

输出格式: `{SYMBOL}_{YYYY}-{MM}_oi_5m.parquet` (DatetimeIndex, columns: oi_contracts, oi_usd, _symbol)

### 5.2 Funding Rate 补全

- [x] 已有 2741 个文件覆盖 58 个 symbol (到 2025-11)
- [ ] 补全到最新 (download_all_data.sh Step 2 自动执行)

```bash
python src/data_tools/download_funding_rate.py \
  --symbols <all_symbols> \
  --start-year 2023 --start-month 1 \
  --parquet-dir data/funding_rate/parquet
```

---

## 🔨 Phase 5.5: 语义预筛选 (Semantic Pre-filter) — Gate v3 核心改进

> 详细设计: `gate_v3_semantic_prefilter_TODO.md`
> 实验报告: `docs/architecture/gate_semantic_prefilter_design.md`

### 5.5.0 问题根因

当前 Gate v1/v2 训练在**全量 bar** 上（ME: 26K bars），大部分 bar 与 archetype 语义无关：
- ME 专属特征 (me_accel_5k 等) 在非扩张 bar 上是纯噪声 → importance=0
- 模型退化为"通用市场 vs 踩坑"分类器，失去 archetype 语义

### 5.5.1 核心思想：从"拍脑袋"到"因果语义驱动"

**旧方案 (v1/v2)**：`labels_rr_extreme.yaml` 的 `filters` 仅做 `notna: true`（全量 bar）

**新方案 (v3)**：在 `filters` 段加 `min`/`max` 条件，只保留 archetype 语义匹配的 bar

> 关键发现：`train_strategy_pipeline.py` 的 `apply_filters()` (L616-L632) **已原生支持** `min`/`max` 操作符 → **零代码修改**，只改 YAML

### 5.5.2 一举两得

同一组预筛选条件同时解决两个问题：
1. **训练噪声** → 剥离不相关样本，专属特征重获区分力
2. **archetype 分配** → 定义"哪些 bar 属于此 archetype"的语义边界

### 5.5.3 预筛选条件来源链路

```
Step 1: 启发式因果设计 → 基于 archetype 语义写特征计算代码
  ↓  脚本: momentum_expansion_features.py / fer_features.py / bpc_features.py
Step 2: 分位数分层验证 → 用百分位阈值切数据, 对比 bad rate
  ↓  脚本: scripts/analyze_archetype_feature_stratification.py
  ↓  报告: z实验_005_统一研究/gate_semantic_prefilter_design.md
Step 3: Plateau 稳健性搜索 → 用 Lift + Stable Plateau 找稳定阈值区间
  ↓  脚本: scripts/optimize_gate_unified.py
输出: guardrail/hard_gate 规则 → 复用为 pre_filter 条件
```

### 5.5.4 四策略预筛选条件 (加入 labels_rr_extreme.yaml filters)

| 策略 | 预筛选条件 | 预期样本量 |
|---|---|---|
| ME | `me_atr_pct ≥ 0.40 AND me_cvd_alignment ≥ 0.40 AND me_volume_surge ≥ 0.30` | ~5-8K (从26K) |
| FER | `fer_trapped_longs_score ≥ 2.75 OR fer_trapped_shorts_score ≥ 3.75` | 待评估 |
| LV | `oi_zscore ≥ 0.5 AND funding_rate_abs_zscore_50 ≥ 0.5` | 待评估 |
| BPC | `bpc_volume_compression_pct ≥ 0.30 AND price_position ≤ 0.90` | 待评估 |

> 注意: FER 的 OR 逻辑当前 `apply_filters()` 不支持，需新增 `any_of` 支持或拆为两次训练

### 5.5.5 Plateau 优化 pre_filter 阈值

当前阈值来自 guardrail（人工设定），可用 Plateau 算法自动搜索最优语义边界：

```bash
# 复用现有优化器搜索 pre_filter 阈值
python scripts/optimize_gate_unified.py \
  --logs results/train_final_xxx/me/predictions.parquet \
  --strategy me --label-col success_no_rr_extreme
```

### 5.5.6 执行清单

- [ ] ME `labels_rr_extreme.yaml` 的 filters 段加 min 条件
- [ ] FER `labels_rr_extreme.yaml` 加 min 条件 (OR 逻辑待解决)
- [ ] LV `labels_rr_extreme.yaml` 加 min 条件
- [ ] BPC `labels_rr_extreme.yaml` 加 min + max 条件
- [ ] (可选) 用 Plateau 搜索优化 pre_filter 阈值
- [ ] 重建 FS + 重训 Gate v3 → 对比 CV 和 feature importance

---

## ✅ Phase 6: 全量训练

### 6.0 训练自动化脚本

- [x] 创建 `scripts/train_all_archetypes.sh`
  - Step 0: Feature Store 构建 (4H / 1H / 15min)
  - **Step 0.5: 语义预筛选 — 修改 labels_rr_extreme.yaml filters** (Phase 5.5)
  - Step 1-4: 各策略 train → gate → optimize
  - Step 5: PCM 联合回测 + KPI 评估
- [x] 报告输出路径: `z实验_005_统一研究/reports/train_<timestamp>/`
  - 每个策略子目录: `bpc/`, `me/`, `fer/`, `lv/`
  - 每个子目录含: train.log, gate.log, gate_optimized.json, evidence_optimized.json
- [ ] 执行全量训练 (依赖 Phase 5 数据下载完成 + Phase 5.5 pre_filter 配置)

```bash
# 一键训练 (后台运行)
nohup bash scripts/train_all_archetypes.sh > /tmp/train_all.log 2>&1 &
```

### 6.1 构建 Feature Store (含新特征)

```bash
# 4H (BPC/FER)
mlbot feature-store build --config config/strategies/bpc \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-12-31 --warmup-months 3 --no-docker

# 1H (ME)
mlbot feature-store build --config config/strategies/me \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a --timeframe 60T \
  --start-date 2023-01-01 --end-date 2025-12-31 --warmup-months 3 --no-docker

# 15min (LV)
mlbot feature-store build --config config/strategies/lv \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a --timeframe 15T \
  --start-date 2023-01-01 --end-date 2025-12-31 --warmup-months 3 --no-docker
```

### 6.2 训练四个 Archetype

每个 archetype 完整流程: **语义预筛选 → Gate 训练 → Apply → Optimize → Evidence → Entry → Backtest**

> 与旧流程的区别: 在 `mlbot train final` **之前**，先确保 `labels_rr_extreme.yaml` 的 `filters` 段包含语义预筛选条件（Phase 5.5），让模型只在 archetype 相关 bar 上训练。

```bash
# ──── BPC (4H) ────
# Step 0: 确认 labels_rr_extreme.yaml 已加入 pre_filter:
#   - column: bpc_volume_compression_pct
#     min: 0.30
#   - column: price_position
#     max: 0.90
mlbot train final --strategy bpc --label-config labels_rr_extreme.yaml
python scripts/apply_archetype_gate.py --strategy bpc
python scripts/optimize_gate_unified.py --strategy bpc
python scripts/optimize_evidence_plateau.py --logs <bpc_predictions> --strategy bpc
python scripts/optimize_entry_filter_plateau.py --logs <bpc_predictions> --strategy bpc
python scripts/backtest_execution_layer.py --logs <bpc_predictions> --strategy bpc

# ──── ME (1H) ────
# Step 0: 确认 labels_rr_extreme.yaml 已加入 pre_filter:
#   - column: me_atr_pct
#     min: 0.40
#   - column: me_cvd_alignment
#     min: 0.40
#   - column: me_volume_surge
#     min: 0.30
mlbot train final --strategy me --timeframe 60T --label-config labels_rr_extreme.yaml
python scripts/apply_archetype_gate.py --strategy me
python scripts/optimize_gate_unified.py --strategy me
python scripts/optimize_evidence_plateau.py --logs <me_predictions> --strategy me
python scripts/optimize_entry_filter_plateau.py --logs <me_predictions> --strategy me
python scripts/backtest_execution_layer.py --logs <me_predictions> --strategy me

# ──── FER (4H) ────
# Step 0: 确认 labels_rr_extreme.yaml 已加入 pre_filter:
#   - column: fer_trapped_longs_score
#     min: 2.75
#   注意: OR 逻辑 (trapped_shorts ≥ 3.75) 需 any_of 支持
mlbot train final --strategy fer --label-config labels_rr_extreme.yaml
python scripts/apply_archetype_gate.py --strategy fer
python scripts/optimize_gate_unified.py --strategy fer
python scripts/optimize_evidence_plateau.py --logs <fer_predictions> --strategy fer
python scripts/optimize_entry_filter_plateau.py --logs <fer_predictions> --strategy fer
python scripts/backtest_execution_layer.py --logs <fer_predictions> --strategy fer

# ──── LV (15min) ────
# Step 0: 确认 labels_rr_extreme.yaml 已加入 pre_filter:
#   - column: oi_zscore
#     min: 0.5
#   - column: funding_rate_abs_zscore_50
#     min: 0.5
mlbot train final --strategy lv --label-config labels_rr_extreme.yaml
python scripts/apply_archetype_gate.py --strategy lv
python scripts/optimize_gate_unified.py --strategy lv
python scripts/optimize_evidence_plateau.py --logs <lv_predictions> --strategy lv
python scripts/optimize_entry_filter_plateau.py --logs <lv_predictions> --strategy lv
python scripts/backtest_execution_layer.py --logs <lv_predictions> --strategy lv
```

### 6.3 PCM 联合回测

```bash
python scripts/backtest_execution_layer.py \
  --pcm bpc:<bpc_predictions> \
        me:<me_predictions> \
        fer:<fer_predictions> \
        lv:<lv_predictions>

python scripts/evaluate_pcm_allocation.py \
  --pcm-report <pcm_report>
```


---

## 🔀 Phase 7: 多时间框架架构升级 (Multi-Timeframe)

> **目标频谱**:
> - L3 结构层 (4H): BPC + FER
> - L2 推进层 (1H): ME
> - L1 微观层 (15min): LV
> - 统一 15min 决策节奏

### 7.0 当前架构评估

**已支持 ✅**:
- GenericLiveStrategy 每个 instance 有独立 `primary_timeframe` + `bar_minutes`
- Feature Store 支持多 timeframe layer (4H + 15min 已有)
- `mlbot train final --timeframe X` 可指定每个策略的 timeframe
- `train_all_archetypes.sh` 已分 `TIMEFRAME_4H` / `TIMEFRAME_15M`
- IncrementalFeatureComputer.compute_features_batch(primary_timeframe=) 支持不同 timeframe 聚合
- 15min 统一决策节奏已在 OrderFlowListener 中

**需要升级 ❌**:
- `_setup_three_strategies()` 使用单一 `bar_minutes=240` 给所有策略
- IncrementalFeatureComputer 每次只产出一个 timeframe 的特征
- OrderFlowListener._compute_and_save_15min_features() 只调一次 compute_features_batch
- PCM.decide() 只收到一组 features，无法按 archetype timeframe 区分
- ME meta.yaml timeframe 仍是 "240T"（需改 "60T"）
- 无 1H Feature Store layer

### 7.1 研究路径升级 (训练/回测)

- [x] ME `config/strategies/me/meta.yaml`: timeframe "240T" → "60T"
- [ ] ME labels 适配 1H: forward_bars / max_holding_bars 重新计算
- [x] 构建 1H Feature Store layer (`unified_1h_2023_2025`)
- [x] ME 独立训练: `mlbot train final --strategy me --timeframe 60T`
- [x] 更新 `train_all_archetypes.sh`: 添加 `TIMEFRAME_1H="60T"`, ME 使用 1H
- [ ] 回测验证: ME@1H vs ME@4H 对比（RR / Sharpe / 与 BPC 正交性）

```bash
# ME (1H) - 独立训练
mlbot train final --strategy me --timeframe 60T --label-config labels_rr_extreme.yaml
python scripts/apply_archetype_gate.py --strategy me --timeframe 60T
python scripts/optimize_gate_unified.py --strategy me
python scripts/optimize_evidence_plateau.py --logs <me_1h_predictions> --strategy me
```

### 7.2 实盘路径升级 (Live)

#### 7.2.1 IncrementalFeatureComputer 多 timeframe 输出

- [ ] 新增 `compute_features_multi_timeframe()` 方法
  - 从同一份 1min bars，分别聚合到 4H/1H/15min
  - 返回 `Dict[str, Dict[str, float]]`（key = timeframe, value = features）
  - 或合并为一组 features，用前缀区分（如 `4h_atr_percentile`, `1h_atr_percentile`）

```python
# ✅ 方案 A（推荐）: 多次调用，复用现有已验证代码
# 优点: 零新代码、独立故障隔离、调试简单
# 性能: 多花 ~5s（VPIN/TC 重复计算），15min 周期内可忽略
features_4h = fc.compute_features_batch(bars, ticks, "240T")
features_1h = fc.compute_features_batch(bars, ticks, "60T")
features_15m = fc.compute_features_batch(bars, ticks, "15T")

# ❌ 方案 B: 一次调用，共享 tick 处理（暂不实现）
# 理由: 过早优化，需要大量新代码+测试，收益不值得复杂度成本
# 如果将来 15min 周期内计算时间不够，再考虑
```

#### 7.2.2 OrderFlowListener 多 timeframe 特征计算

- [ ] `_compute_and_save_15min_features()` 支持按策略 timeframe 计算多组特征
- [ ] 特征传递: 按 archetype 的 primary_timeframe 路由对应特征

#### 7.2.3 PCM 多 timeframe 决策

- [ ] `LivePCM.decide()` 接收多组 timeframe features
- [ ] 每个 registered strategy 用其对应 timeframe 的 features 调用 decide()
- [ ] 方案: PCM 内部维护 `archetype → timeframe` 映射

```python
# PCM 知道每个策略的 timeframe
pcm.register("bpc", bpc, timeframe="240T")
pcm.register("me", me, timeframe="60T")
pcm.register("fer", fer, timeframe="240T")
pcm.register("lv", lv, timeframe="15T")

# decide 时按 timeframe 路由
def decide(self, *, symbol, features_by_timeframe, ...):
    for name, strategy in self._strategies.items():
        tf = self._timeframes[name]
        intents = strategy.decide(features=features_by_timeframe[tf], ...)
```

#### 7.2.4 run_live.py 升级

- [ ] `_setup_four_strategies()`: 每个策略独立 primary_timeframe
- [ ] `IncrementalFeatureComputer` 配置多个 timeframe 输出
- [ ] 环境变量: `MLBOT_ME_BAR_MINUTES=60`, `MLBOT_LV_BAR_MINUTES=15`

### 7.3 架构设计原则

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

**关键约束**:
1. 触发节奏统一 15min，高周期只是 context（不是触发周期）
2. 每层最多 1-2 个主 archetype，避免频谱冲突
3. ME@1H 和 BPC@4H 正交（一个做"速度"，一个做"结构"）
4. FER@4H 和 BPC@4H 构成 continuation ↔ reversal 对
5. 特征计算从 1min bars 起始，不同 timeframe 只是聚合粒度

### 7.4 验证清单

- [ ] ME@1H 训练: Sharpe > 0.5, 与 BPC@4H 相关性 < 0.3
- [ ] 实盘多 timeframe 特征计算: 15min/1h/4h 三组特征正确
- [ ] PCM 多 timeframe 仲裁: 不同 timeframe 策略可正常竞争
- [ ] 回归测试: 原 4H-only 模式行为不变（向后兼容）

---

## 📋 执行顺序与依赖

```
Phase 1 (组合特征)  ──┐
Phase 2 (OI 特征)   ──┤──→ Phase 5 (数据) ──→ Phase 5.5 (语义预筛选) ──→ Phase 6 (训练)
Phase 3 (LV 配置)   ──┤                                                        │
Phase 4 (PCM 重构)  ──┘                                                        ▼
                                                                      Phase 7 (多时间框架)
```

- Phase 1-4 可并行开发
- Phase 5 依赖 Phase 2 (OI 下载器)
- Phase 6 依赖 Phase 1-5 全部完成

---

## 📊 进度追踪

| Phase | 状态 | 备注 |
|-------|------|------|
| Phase 0: 审查 | ✅ 完成 | 7 特征全部合理 |
| Phase 1: 组合特征 | ✅ 完成 | 3 个乘法交叉特征 (dual_compression/ignition/exhaustion) |
| Phase 2: OI 体系 | ✅ 完成 | 下载器 + 特征 + 场景语义 + 交叉特征 |
| Phase 3: LV 配置 | ✅ 完成 | 15min LV archetype 全套配置 |
| Phase 4: PCM 重构 | ✅ 完成 | v2: 单一优先级制 (LV>FER>ME>BPC, 按条件严格性), 移除 Override 层 |
| Phase 5: 数据 | ✅ 完成 | OI 59 symbols 1947 files (2023-01~2026-02), FR 59 symbols 2860 files |
| Phase 5.5: 语义预筛选 | 🔨 设计完成 | 详见 gate_v3_semantic_prefilter_TODO.md |
| Phase 6: 训练 | 🔨 待执行 | 依赖 Phase 5.5 pre_filter 配置, 报告输出到 reports/ |
| Phase 7: 多时间框架 | ✅ 研究路径完成 | ME→1H, BPC/FER→4H, LV→15min, 实盘路径待升级 |

---

## 🔍 Phase 4.5: PCM 优先级验证 (v2 严格性排序)

> 目标: 验证 v2 优先级 (LV>FER>ME>BPC) 相比 v1 (BPC>ME>FER>LV) 有优势
> 设计文档: `PCM优先级简化设计.md`
> 假设监控: `系统每层假设与实盘监控.md`

### 验证方法

- [ ] 用历史 predictions.parquet 重跑 PCM 回测，对比 v1 vs v2 的冲突解决效果
  - 指标: Sharpe, 胜率, 平均 R, max DD
  - 期望: v2 在冲突场景下 Sharpe ≥ v1（冲突率低，预期影响小）
- [ ] 反事实分析: v2 被拒信号的事后 R 和胜率
  - 关注: v2 被拒的 BPC 信号（之前是最高优先级）事后表现
- [ ] Regime 分层验证: HIGH_VOL 下 ME>FER 是否给出更好的冲突解决

### 实盘监控要点

- [ ] 每周检查 PCM 冲突日志，对比被选 vs 被拒信号事后 R
- [ ] 若被拒信号事后 R 系统性高于被选 → 优先级需要调整
- [ ] 监控各层假设是否成立（见 `系统每层假设与实盘监控.md`）

---

## 📡 Phase 8: 实盘假设监控

> 依据: `系统每层假设与实盘监控.md`
> 目标: 验证系统 9 层假设在实盘环境中是否持续成立，失效时触发调整

### 监控范围

| 层 | 假设 | 关键指标 | 失效阈值 |
|----|------|----------|----------|
| L1 特征 | 历史统计规律在新数据上保持 | feature_drift_zscore | > 3.0 连续 3 天 |
| L2 预筛选 | 语义场景条件有效过滤噪声 | prefilter_pass_rate | 偏离训练期 ±50% |
| L3 Gate | 通过 Gate 的信号有显著正向 lift | gate_pass_rate, gate_lift | lift < 1.2 |
| L4 Evidence | 高 evidence score → 高 R | evidence_r_correlation | Spearman < 0.05 |
| L5 Direction | 方向判断增加胜率 | direction_accuracy | < 55% (30日滚动) |
| L6 Entry Filter | 入场过滤器提升信号质量 | entry_filter_lift | lift < 1.0 |
| L7 Execution Tier | 高 tier 的 R 和胜率优于低 tier | per_tier_mean_r | T1 ≤ T3 连续 2 周 |
| L8 PCM | 被选信号 > 被拒信号 | counterfactual_r | 被拒 avg R > 被选持续 1 周 |
| L9 宪法 | 安全边界合理且不过度限制 | kill_switch_trigger_count | 月触发 > 3 次需审查 |

### 实施清单

- [ ] 日频监控脚本: 特征漂移 / Gate 通过率 / Direction 准确率 / 当日 PnL
- [ ] 周频监控脚本: Evidence 相关性 / PCM 冲突分析 / Tier 分层表现
- [ ] 月频全层验证报告: 9 层假设逐一检查 + 正交性验证
- [ ] 假设失效告警: 任何层指标越过阈值 → 自动通知
- [ ] 假设失效归因 SOP: 失效 → 定位层 → 分析原因 → 制定调整方案

---

## ⚡ Phase 9: 实盘性能监控

> 目标: 确保系统在生产环境运行的延迟、吞吐、稳定性满足交易需求

### 延迟监控

| 环节 | 指标 | 目标值 | 告警阈值 |
|------|------|--------|----------|
| Tick 接收 → 特征计算完成 | feature_latency_ms | < 200ms | > 500ms |
| 特征计算 → Gate/Evidence 推理 | inference_latency_ms | < 100ms | > 300ms |
| 信号生成 → PCM 仲裁完成 | pcm_latency_ms | < 50ms | > 150ms |
| PCM 决策 → 下单提交 | order_submit_latency_ms | < 100ms | > 300ms |
| 端到端 (Tick → Order) | e2e_latency_ms | < 500ms | > 1000ms |

### 稳定性监控

| 指标 | 说明 | 告警条件 |
|------|------|----------|
| WebSocket 断线次数 | 每日断线统计 | > 5 次/天 |
| Tick 缺失率 | 预期 tick 数 vs 实际接收数 | > 2% |
| 特征计算异常率 | NaN/Inf 产出比例 | > 0.1% |
| 内存使用 | 进程 RSS | > 4GB 或持续增长 |
| CPU 使用率 | 进程 CPU 占比 | > 80% 持续 5 分钟 |
| 磁盘 I/O | 日志/数据写入速率 | 磁盘使用 > 90% |

### 交易执行监控

| 指标 | 说明 | 告警条件 |
|------|------|----------|
| 下单成功率 | 提交 vs 成交 | < 95% |
| 滑点 | 信号价 vs 成交价 | avg slippage > 0.1% |
| 持仓偏离 | 目标仓位 vs 实际仓位 | 偏离 > 5% |
| API 错误率 | 交易所 API 调用失败率 | > 1% |

### 实施清单

- [ ] 延迟打点埋入: 各环节 start/end timestamp 记录
- [ ] Prometheus/Grafana 看板搭建 (或轻量替代方案)
- [ ] 日频性能报告: 延迟 P50/P95/P99 + 异常统计
- [ ] 告警通道接入: 关键指标越限 → 即时通知
- [ ] 性能基线建立: 上线首周记录各指标基线，后续对比
- [ ] 降级策略: 延迟过高时自动降低交易频率或暂停非核心 archetype
