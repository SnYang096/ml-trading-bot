# ts-factor-eval 功能增强说明

## 概述

本次更新为 `ts-factor-eval` 添加了两个重要功能：
1. **相关性去冗余**：自动移除高度相关的冗余特征
2. **Best Lag 过滤**：根据特征的 best lag（IC 峰值时间）过滤特征

## 1. 相关性去冗余功能

### 功能说明

自动识别并移除高度相关的特征，减少特征冗余，提高模型训练效率。

### 使用方法

```bash
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_REMOVE_CORRELATED=1 \
  TS_FACTOR_CORRELATION_THRESHOLD=0.9
```

### 参数说明

- `TS_FACTOR_REMOVE_CORRELATED=1`：启用相关性去冗余
- `TS_FACTOR_CORRELATION_THRESHOLD=0.9`：相关性阈值（默认 0.9），相关系数 >= 此值的特征将被视为冗余

### 工作原理

1. 计算所有特征之间的相关性矩阵
2. 按 IC IR（或 IC Mean）对特征排序
3. 依次检查每个特征：
   - 如果与已保留的特征相关性 >= 阈值
   - 且当前特征分数 <= 已保留特征的分数
   - 则移除当前特征，保留已保留的特征
4. 输出筛选后的特征列表

### 输出

- 在报告中显示移除的特征及原因
- 在 diagnostics 中保存详细的移除信息

## 2. Best Lag 过滤功能

### 功能说明

根据特征的 best lag（IC 峰值对应的时间滞后）过滤特征，只保留与目标 lag 匹配的特征。

### Best Lag 含义

- **Best Lag**：在不同 forward lag（1, 10, 20 bars）中，该特征与未来收益的 IC 最高的那个 lag
- 表示该特征最擅长预测未来 N 个 bar 的收益

### 使用方法

```bash
# 自动推断 target lag（从策略配置的 max_holding_bars）
# 需要显式启用 TS_FACTOR_FILTER_BY_BEST_LAG=1，因为未指定 target_lag
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_FILTER_BY_BEST_LAG=1 \
  TS_FACTOR_LAG_TOLERANCE=5

# 指定 target lag（会自动启用 best lag 过滤，无需 TS_FACTOR_FILTER_BY_BEST_LAG=1）
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_TARGET_LAG=20 \
  TS_FACTOR_LAG_TOLERANCE=5
```

### 参数说明

- `TS_FACTOR_FILTER_BY_BEST_LAG=1`：启用 Best Lag 过滤（如果不指定 `TS_FACTOR_TARGET_LAG`，需要此参数来启用过滤）
- `TS_FACTOR_TARGET_LAG=20`：目标 lag（如果指定，会自动启用 best lag 过滤，无需同时指定 `TS_FACTOR_FILTER_BY_BEST_LAG=1`）
- `TS_FACTOR_LAG_TOLERANCE=5`：容差（默认 5），保留 `|best_lag - target_lag| <= tolerance` 的特征

**注意**：如果指定了 `TS_FACTOR_TARGET_LAG`，best lag 过滤会自动启用，无需再设置 `TS_FACTOR_FILTER_BY_BEST_LAG=1`。

### Target Lag 推断规则

如果不指定 `TS_FACTOR_TARGET_LAG`，系统会：
1. 从策略配置的 `labels.yaml` 中读取 `label_generator.params.max_holding_bars`
2. 使用 `max(10, min(max_holding_bars * 0.4, max_holding_bars // 2))` 作为 target lag
3. 如果无法推断，使用 IC decay lags 的中间值

**说明**：
- `max_holding_bars` 是标签生成时的**最大扫描周期**（上限），用于确定标签的超时阈值
- 实际的持仓周期通常小于 `max_holding_bars`，因为一旦触达止盈/止损就会提前平仓
- `target_lag` 应该反映**典型的持仓周期**，而不是最大持仓周期
- 对于 SR Reversal 策略，典型持仓周期通常是 `max_holding_bars` 的 30-50%
- 系统使用 40% 作为典型持仓周期的估计

**示例**：
- `max_holding_bars = 50` → `target_lag = max(10, min(20, 25)) = 20`
- `max_holding_bars = 30` → `target_lag = max(10, min(12, 15)) = 12`
- `max_holding_bars = 20` → `target_lag = max(10, min(8, 10)) = 10`

**建议**：如果需要精确控制 target lag，直接使用 `TS_FACTOR_TARGET_LAG` 参数，而不是依赖自动推断。

### max_holding_bars 配置位置

`max_holding_bars` 在策略配置目录下的 `labels.yaml` 文件中配置：

**配置文件路径**：`config/strategies/{strategy_name}/labels.yaml`

**配置结构**：
```yaml
name: sr_reversal
target_column: label

label_generator:
  module: src.time_series_model.strategies.labels.sr_reversal_label
  function: compute_sr_reversal_label_full_scan
  params:
    max_holding_bars: 50  # ← 这里配置最大持仓周期
    take_profit_r: 2.0
    stop_loss_r: 1.0
    combine_mode: any_success
```

**说明**：
- `max_holding_bars` 表示策略的最大持仓周期（bars）
- 对于 SR Reversal 策略，通常设置为 50
- 这个值用于标签生成（确定超时时间），也用于推断 target lag

### 使用建议

- **SR Reversal 策略**：通常 `max_holding_bars=50`，推荐 target lag = 20
- **短期策略**：如果 holding period 较短（< 20 bars），使用较小的 target lag
- **长期策略**：如果 holding period 较长（> 50 bars），可以使用更大的 target lag

### 输出

- 在报告中显示保留和移除的特征
- 显示每个特征的 best lag 和与 target lag 的差值
- 在 diagnostics 中保存详细的过滤信息

## 3. 组合使用

两个功能可以组合使用，实现更精细的特征筛选：

```bash
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_REMOVE_CORRELATED=1 \
  TS_FACTOR_CORRELATION_THRESHOLD=0.9 \
  TS_FACTOR_TARGET_LAG=20 \
  TS_FACTOR_LAG_TOLERANCE=5
```

**执行顺序**：
1. 先计算所有特征的 IC/IR 和 best lag
2. 然后应用相关性去冗余
3. 最后应用 best lag 过滤

## 4. 典型工作流

### 阶段 1：初步筛选

```bash
# 使用相关性去冗余快速去除冗余特征
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_REMOVE_CORRELATED=1
```

### 阶段 2：精细化筛选

```bash
# 根据策略的 holding period 过滤特征（指定 target lag 会自动启用过滤）
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_TARGET_LAG=20
```

### 阶段 3：综合筛选

```bash
# 同时应用两个筛选条件（如果需要自动推断 target lag，需要显式启用 TS_FACTOR_FILTER_BY_BEST_LAG=1）
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_REMOVE_CORRELATED=1 \
  TS_FACTOR_FILTER_BY_BEST_LAG=1

# 或者指定 target lag（自动启用 best lag 过滤）
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_REMOVE_CORRELATED=1 \
  TS_FACTOR_TARGET_LAG=20
```

## 5. 关于 Best Lag 的选择建议

### Best Lag = 1 的特征

- **特点**：擅长预测未来 1 个 bar 的收益
- **适用场景**：高频交易、短期策略
- **SR Reversal 策略**：通常不推荐，因为：
  - 交易成本高（频繁交易）
  - 信号稳定性差（短期噪声大）
  - 与策略逻辑不匹配（SR Reversal 是中期反转策略）

### Best Lag = 10-20 的特征

- **特点**：擅长预测未来 10-20 个 bars 的收益
- **适用场景**：中期策略（如 SR Reversal）
- **推荐**：这类特征通常更适合 SR Reversal 策略

### Best Lag > 20 的特征

- **特点**：擅长预测长期收益
- **适用场景**：长期策略、趋势跟踪
- **使用建议**：可以保留，但要注意与 `max_holding_bars` 的匹配

## 6. 与 ts-dim-compare 的关系

本次更新将 `ts-dim-compare` 中的相关性去冗余功能迁移到了 `ts-factor-eval`，使得：

- ✅ 在因子评估阶段就可以去除冗余特征
- ✅ 工作流更清晰：`ts-factor-eval` → `ts-strategy-feature-compare`
- ✅ `ts-dim-compare` 可以专注于批量统计筛选（如果需要）

## 7. 修复的问题

### 修复缺失输出列问题

- 修复了 `atr_f` 配置问题：从 `compute_atr` 改为 `compute_atr_from_series`
- `compute_atr` 返回 Series，而 narrow-IO 架构需要 DataFrame
- `compute_atr_from_series` 返回 DataFrame，符合架构要求

## 8. 报告中的新信息

HTML 报告现在包含：

1. **Correlation-Based Feature Removal** 部分
   - 显示移除的特征数量和原因
   - 列出被移除的特征

2. **Best Lag Filtering** 部分
   - 显示 target lag 和容差
   - 列出保留和移除的特征
   - 显示每个特征的 best lag 信息

## 相关文档

- [架构文档](ARCHITECTURE.md) - 了解特征筛选流程
- [研发流程指南](DEVELOPMENT_WORKFLOW.md) - 了解完整的工作流

