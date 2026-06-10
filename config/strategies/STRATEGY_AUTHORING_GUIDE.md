# 策略编写规范 (Strategy Authoring Guide)

> 目标：让未来的 AI/开发者能独立创建一个新策略，遵循项目约定。

## 1. 策略分类

| 类型 | 示例 | 引擎 |
|------|------|------|
| **Per-trade 趋势策略** | TPC, BPC, ME, SRB | `src/time_series_model/live/generic_live_strategy.py` |
| **Portfolio 级策略** | spot_accum_simple, rolling_trend | 自定义 engine（`src/time_series_model/live/`） |
| **Multi-leg 策略** | chop_grid, trend_scalp | `scripts/chop_grid_backtest.py` |

## 2. 文件结构

```
config/strategies/<name>/
├── meta.yaml              # [必需] 策略元信息
├── features.yaml          # [必需] 特征管线配置
├── README.md              # [推荐] 策略说明
├── model.yaml             # [可选] ML 模型配置
├── labels*.yaml           # [可选] 训练标签
├── semantic_polarity.yaml # [可选] 语义极性
├── research/              # [推荐] 研究笔记
└── archetypes/            # [必需]
    ├── prefilter.yaml     # 信号预过滤
    ├── gate.yaml          # 硬门控
    ├── direction.yaml     # 方向判断
    ├── entry_filters.yaml # 入场过滤器
    ├── regime.yaml        # Regime 过滤
    └── execution.yaml     # 执行参数 (SL/TP/trailing/add_position)
```

## 3. meta.yaml 规范

```yaml
strategy:
  name: <strategy-name>-<timeframe>   # 例: tpc-long-120T
  description: |
    策略简要描述。
  owner: quant_team
  archetype: <ArchetypeName>          # PascalCase
  timeframe: "120T"                   # 120T=2H, 60T=1H, 240T=4H
  symbol_include: []                  # 空=全部
  symbol_exclude: []
```

## 4. archetypes/*.yaml 规范

### 4.1 prefilter.yaml — 信号预过滤

必须包含 `rules` 列表，每条规则含 `feature`, `operator`, `value`：

```yaml
rules:
  - feature: tpc_pullback_depth
    operator: <=
    value: 0.85
    rationale: 回踩深度控制
    locked: true
```

### 4.2 gate.yaml — 硬门控

```yaml
schema:
  phases: [hard_gate, guardrail]
  evaluation_order: hard_gate -> guardrail
hard_gates:
  - id: gate_xxx
    priority: 6
    when:
      feature_name:
        value_gt: 0.4    # 支持 value_gt / value_lt / value_eq
    then:
      action: deny       # deny 或 warn
    locked: true
```

### 4.3 direction.yaml — 方向判断

```yaml
direction_rules:
  - id: rule_id
    method: signal_match_position_band  # 或 dual_position_agree_deadband
    signal_rules:
      - feature: macd_atr
        transform: sign
    position_band:
      feature: ema_1200_position
      inner_abs: 0.10
```

### 4.4 entry_filters.yaml — 入场过滤

```yaml
filters:
  - id: filter_id
    enabled: true
    conditions:
      - feature: vol_confirm
        operator: '>='
        value: 0.45
combination_mode: or   # or 或 and
```

### 4.5 regime.yaml — 市场环境分类

**regime.yaml 是牛/熊/震荡分类的唯一权威来源。** 其他层（execution、gate、prefilter）
只能引用 regime 分类结果，不得重复判断。

```yaml
# ── 分类定义（唯一权威）──
allowed_regimes:
  bull:
    description: "强趋势市 — 适合结构退出"
    match: all                              # all=所有规则满足, any=任一满足
    rules:
      - feature: ema_1200_position
        operator: '>='
        value: 0.15
      - feature: adx
        operator: '>='
        value: 25
  bear:
    description: "弱势/下跌趋势 — 适合移动止损"
    match: any
    rules:
      - feature: ema_1200_position
        operator: '<='
        value: -0.10
      - feature: adx
        operator: '<='
        value: 20
  neutral:
    description: "震荡市 — 适合移动止损"
    match: any
    rules:
      - feature: adx
        operator: '<='
        value: 25

# ── 方向限制（各层引用）──
allowed_sides: [long, short]
```

### 4.5.1 各层引用 regime 的方式

**execution.yaml** 根据 regime 切换退出策略：

```yaml
stop_loss:
  exit_by_regime:
    bull:               # ← 引用 regime.yaml 的 bull 分类
      structural_exit: ema1200
      trailing:
        enabled: false
    bear:               # ← 引用 regime.yaml 的 bear 分类
      trailing:
        enabled: true
    neutral:            # ← 默认
      trailing:
        enabled: true
```

**gate.yaml** 也可以引用：

```yaml
hard_gates:
  - id: gate_in_bull_only
    regime_allow: [bull]      # 只在牛市启用此 gate
    ...
```

**设计原则：**
- `regime.yaml` 定义"什么是牛/熊/震荡"（特征+阈值）
- 其他层只需声明"在 X 类市场中怎么做"
- 禁止在其他层重复判断 regime（如 execution 里再写 ema_1200_position > 0.15）

### 4.6 execution.yaml — 执行参数

```yaml
stop_loss:
  initial_r: 4.0
  structural_exit: ema1200
  trailing:
    enabled: true
    activation_r: 3.5
    trail_r: 6.0
  breakeven:
    enabled: true
    trigger_r: 6.0
    lock_level_r: 2
    measure: atr
add_position:
  add_size_multipliers: [0.25, 0.5, 1.0]
  trigger:
    type: float_r_ladder_only
  min_current_r_by_add: [0.5, 1, 1.5]
  min_current_r_unit: atr
holding:
  max_holding_bars: 0
  time_stop_bars: 0
```

## 5. R&D 标准流程 — 禁止跳步

> 详细完整版见 `config/experiments/LAYER_PROMOTION_CRITERIA.md`

| Phase | 名称 | 做什么 | 工具 | 能 promote？ |
|:-----:|------|--------|------|:-----------:|
| **0** | 特征可算 | 特征进 `features*.yaml`，能产出 `features_labeled.parquet` | `train_strategy_pipeline --prepare-only` | ❌ |
| **1** | 假设扫描 | IC、label plateau、condition-set → 生成假设 | `mlbot research scan/ic/plateau` + `rd_loop_*.yaml` | ❌ |
| **2** | 定参 | 人从 Phase 1 报告选 τ/lookback/阈值，写 `DECISION.md` | 人读 scan 报告 | ❌ |
| **3** | 因果复验 | `segment_matrix` + `market_segment.yaml` 三段 event_backtest | `variant_grid` + `scripts.event_backtest` | ❌ |
| **4** | 人审 | trading map 核对入场语义 | `run_trading_maps.sh` | ❌ |
| **5** | Promote | 三段 Total R↑ + maxDD 不恶化 + 逻辑可解释 → `locked: true` | 人工判定 | ✅ |

**核心原则：IC/label scan 只能生成假设，不能直接决定生产配置。只有三段 variant-grid 回测 + 三条杠达标才能 promote。**

### 5.1 Phase 0 操作细节

#### 特征添加到 `features.yaml` 后，必须用增量 FeatureStore

**错误做法**：
```bash
# ❌ 不指定 --layer → config hash 变化，自动生成新 layer → 全量重算（数十分钟）
mlbot feature-store build --no-docker --config config/strategies/tpc \
  --symbols BTCUSDT,ETHUSDT --timeframe 120T
```

**正确做法**：
```bash
# 1. 找到已有 layer
ls feature_store/features_tpc_120T_*
# → features_tpc_120T_9506bdec50（6 币种 × 52+ 月）

# 2. --layer 指定已有 layer → 增量添加新特征 → ~30s
mlbot feature-store build --no-docker \
  --config config/strategies/tpc \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --timeframe 120T \
  --start-date 2022-01-01 --end-date 2026-04-30 \
  --root feature_store \
  --layer features_tpc_120T_9506bdec50 \
  --warmup-months 12
```

**原理**：`--layer` 指定已有名称 → 系统检测已有月份已存在 → 只计算缺失的新列（增量）；不指定则 config hash 触发全新 layer → 全部从零计算。

**重要**：
- `--start-date` 应和已有 layer 的数据范围对齐，避免算不需要的早期月份
- 新特征必须出现在策略 `features.yaml` 的 `requested_features` 里（`_shared/features.yaml` 只是注册表，不触发计算）
- 用 `--no-reuse` 可禁用跨 layer 复制，纯增量；默认开启 reuse（自动从其他 layer 拷贝已有月份）

## 6. 引擎代码位置

| 组件 | 路径 |
|------|------|
| 通用策略引擎 | `src/time_series_model/live/generic_live_strategy.py` |
| 持仓管理 | `src/time_series_model/live/position_logic.py` |
| PCM 投资组合 | `src/time_series_model/portfolio/live_pcm.py` |
| 特征计算 | `src/time_series_model/live/live_feature_plan.py` |
| 事件回测 | `scripts/event_backtest/backtester.py` |
| Variant grid | `scripts/event_backtest/variant_grid.py` |

## 7. 回测运行

```bash
# 单次回测
python -m scripts.event_backtest \
  --strategy tpc \
  --strategies-root config/strategies \
  --start-date 2023-01-01 --end-date 2025-01-01 \
  --symbols BTCUSDT,ETHUSDT --fast

# Variant grid (多变体对比)
python -m scripts.event_backtest \
  --variant-grid config_experiments/<name>/grid.yaml
```

## 8. 实盘注册

在 `config/constitution/constitution.yaml` 的 `resource_allocation.enabled_archetypes` 中加入策略名。

## 9. 实验目录规范

所有 R&D 实验文件**自包含在一个目录**中，放在 `config/experiments/` 下：

```
config/experiments/<YYYYMMDD>_<strategy>_<topic>/
├── README.md                       # [必需] 复现步骤 + 结论
├── DECISION.md                     # [必需] 定参决策 + promote/delete 建议
├── rd_loop_<topic>.yaml            # [可选] Phase 1 扫描配置
├── phase1_scan.json                # [可选] Phase 1 产物
├── phase2_grid.yaml                # [可选] Phase 3 variant grid
└── phase3_results.md               # [可选] grid 结果汇总
```

**规则**：
- grid.yaml **在实验目录内**，不在全局 `config_experiments/` 下
- grid 的 `strategies_root` 指向 `config_experiments/<variant>/`（静态策略树快照）
- 不要创建自定义脚本（如 `augment_adx.py`）——用 `mlbot feature-store build` + `mlbot research`
- 一个实验一个目录，所有文件（rd_loop、grid、DECISION）在一起

**示例**（TPC ADX regime）：
```
config/experiments/20260610_tpc_regime_adx_phase1/
├── README.md                       # 完整复现命令
├── DECISION.md                     # ADX(50)>25 作为 bull
├── rd_loop_tpc_regime_adx.yaml     # Phase 1 扫描
├── phase1_scan.json                # IC/plateau 结果
└── phase2_grid.yaml                # E9 vs E21 vs E22
```

## 10. Portfolio 级策略特殊说明

Portfolio 级策略（如 rolling_trend）不完全遵循 per-trade archetype 模式：

- 仍需 `meta.yaml` + `features.yaml` 定义信号源
- 执行逻辑在专用 engine 中（`src/time_series_model/live/<name>_strategy.py`）
- 回测逻辑在专用 backtest 脚本中（`scripts/<name>_backtest.py`）
- 不在 PCM slot 池中竞争，使用独立账户

## 11. 检查清单

- [ ] `meta.yaml` 有正确的 name/timeframe/archetype
- [ ] `features.yaml` 声明所有需要的特征节点
- [ ] 6 个 archetypes yaml 完整且语法正确
- [ ] constitution 中已注册（如需要）
- [ ] 回测可以正常运行
- [ ] 有 README.md 说明策略逻辑
