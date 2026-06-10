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

### 4.5 regime.yaml — 市场环境

```yaml
allowed_regimes: [bull, bear, neutral]
allowed_sides: [long, short]
rules:
  - any_of:
      - feature: ema_1200_position
        operator: '>='
        value: 0.1
```

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

## 5. 引擎代码位置

| 组件 | 路径 |
|------|------|
| 通用策略引擎 | `src/time_series_model/live/generic_live_strategy.py` |
| 持仓管理 | `src/time_series_model/live/position_logic.py` |
| PCM 投资组合 | `src/time_series_model/portfolio/live_pcm.py` |
| 特征计算 | `src/time_series_model/live/live_feature_plan.py` |
| 事件回测 | `scripts/event_backtest/backtester.py` |
| Variant grid | `scripts/event_backtest/variant_grid.py` |

## 6. 回测运行

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

## 7. 实盘注册

在 `config/constitution/constitution.yaml` 的 `resource_allocation.enabled_archetypes` 中加入策略名。

## 8. 实验目录规范

```
config_experiments/<experiment_name>_strategies/
├── grid.yaml                      # variant grid 定义
├── DECISION.md                    # 实验结论
├── <variant_name>/
│   └── <strategy>/archetypes/
│       └── <changed_file>.yaml    # 只放改动的文件
└── constitution/
    └── <variant>.yaml             # constitution override
```

## 9. Portfolio 级策略特殊说明

Portfolio 级策略（如 rolling_trend）不完全遵循 per-trade archetype 模式：

- 仍需 `meta.yaml` + `features.yaml` 定义信号源
- 执行逻辑在专用 engine 中（`src/time_series_model/live/<name>_strategy.py`）
- 回测逻辑在专用 backtest 脚本中（`scripts/<name>_backtest.py`）
- 不在 PCM slot 池中竞争，使用独立账户

## 10. 检查清单

- [ ] `meta.yaml` 有正确的 name/timeframe/archetype
- [ ] `features.yaml` 声明所有需要的特征节点
- [ ] 6 个 archetypes yaml 完整且语法正确
- [ ] constitution 中已注册（如需要）
- [ ] 回测可以正常运行
- [ ] 有 README.md 说明策略逻辑
