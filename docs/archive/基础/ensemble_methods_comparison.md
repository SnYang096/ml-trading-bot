# 多时间框架集成方法对比

## 问题分析

原始的 `average` 方法存在以下问题：

### 原始方法的缺陷

```python
# 对所有时间框架信号取平均
ensemble_signal = mean(signal_1m, signal_5m, signal_15m)

# 只有平均值 > 0.1 或 < -0.1 才开仓
if ensemble_signal > 0.1: BUY
elif ensemble_signal < -0.1: SELL
else: HOLD
```

**问题示例**（假设3个时间框架）：

| 1m | 5m | 15m | 平均值 | 结果 | 问题 |
|----|----|-----|--------|------|------|
| +1 | +1 | +1  | 1.0    | 做多 ✓ | 完美但罕见 |
| +1 | +1 | 0   | 0.67   | 做多 ✓ | 合理 |
| +1 | 0  | 0   | 0.33   | 做多 ✓ | 只有一个信号就开仓？|
| +1 | -1 | 0   | 0.0    | 不开仓 ✗ | **错失机会** |
| +1 | +1 | -1  | 0.33   | 做多 ✓ | 有冲突信号还开仓？|

**核心问题**：
1. **大幅减少开仓机会** - 需要多数时间框架同向
2. **信号稀释** - 强信号被弱信号中和
3. **时间框架平等不合理** - 15分钟和1分钟权重相同

---

## 改进方案对比

### 1. `weighted` - 加权投票法（推荐默认）

**理念**：大周期更重要，但不完全忽略小周期

**权重计算**：
```python
# 使用平方根避免大周期过度主导
weights = sqrt([1, 5, 15]) / sum(sqrt([1, 5, 15]))
# 结果：1m=0.21, 5m=0.47, 15m=0.81 (归一化后)
```

**开仓条件**：加权信号 > 0.3 或 < -0.3

**示例**：
| 1m | 5m | 15m | 加权信号 | 结果 |
|----|----|-----|----------|------|
| +1 | +1 | -1  | 0.12     | 不开仓（冲突太大）|
| +1 | +1 | 0   | 0.68     | 做多 ✓ |
| +1 | 0  | +1  | 0.85     | 做多 ✓ |
| +1 | -1 | 0   | -0.05    | 不开仓（冲突）|

**优点**：
- ✅ 平衡了各时间框架的重要性
- ✅ 增加开仓机会（阈值0.3 vs 0.1）
- ✅ 大周期有更高话语权

**适用场景**：通用场景，日常交易

---

### 2. `hierarchical` - 分层决策法（稳健型）

**理念**：大周期定方向，小周期找入场时机

**逻辑**：
```python
trend_direction = signal_15m  # 大周期定趋势
entry_timing = signal_1m      # 小周期找时机

# 只有当小周期与大周期同向时才开仓
if trend_direction > 0.5 AND entry_timing > 0.5: BUY
if trend_direction < -0.5 AND entry_timing < -0.5: SELL
```

**示例**：
| 1m | 5m | 15m | 结果 | 说明 |
|----|----|-----|------|------|
| +1 | +1 | +1  | 做多 ✓ | 完美对齐 |
| +1 | 0  | +1  | 做多 ✓ | 趋势向上，入场时机对 |
| +1 | -1 | +1  | 做多 ✓ | 以大周期和小周期为主 |
| +1 | +1 | 0   | 不开仓 | 无明确趋势 |
| +1 | +1 | -1  | 不开仓 | 逆势不做 |

**优点**：
- ✅ 遵循"顺势交易"原则
- ✅ 减少逆势交易风险
- ✅ 逻辑清晰，易于理解

**缺点**：
- ⚠️ 开仓机会较少（需要大小周期对齐）
- ⚠️ 可能错过短期机会

**适用场景**：趋势跟随策略，风险厌恶型交易者

---

### 3. `independent` - 独立信号法（激进型）

**理念**：任何时间框架的强信号都可独立触发

**逻辑**：
```python
# 只要任何一个时间框架给出强信号就开仓
if ANY(signal_1m, signal_5m, signal_15m) > 0.5: BUY
if ANY(signal_1m, signal_5m, signal_15m) < -0.5: SELL
```

**示例**：
| 1m | 5m | 15m | 结果 | 说明 |
|----|----|-----|------|------|
| +1 | 0  | 0   | 做多 ✓ | 1m强信号即可 |
| +1 | -1 | 0   | 做多 ✓ | 有强信号就做 |
| 0  | 0  | +1  | 做多 ✓ | 15m强信号即可 |
| 0  | 0  | 0   | 不开仓 | 无强信号 |

**优点**：
- ✅ 最大化开仓机会
- ✅ 不会错过任何时间框架的强信号
- ✅ 适合捕捉短期机会

**缺点**：
- ⚠️ 可能产生冲突交易
- ⚠️ 风险较高，假信号多
- ⚠️ 需要更严格的风控

**适用场景**：日内短线交易，高频交易，接受高风险

---

### 4. `majority` - 多数投票法（中庸型）

**理念**：超过半数时间框架同意才开仓

**逻辑**：
```python
count_long = sum(signal > 0.5 for all signals)
count_short = sum(signal < -0.5 for all signals)

if count_long > n_timeframes/2: BUY
if count_short > n_timeframes/2: SELL
```

**示例**（3个时间框架）：
| 1m | 5m | 15m | 做多票数 | 做空票数 | 结果 |
|----|----|-----|----------|----------|------|
| +1 | +1 | 0   | 2        | 0        | 做多 ✓ |
| +1 | +1 | -1  | 2        | 1        | 做多 ✓ |
| +1 | 0  | 0   | 1        | 0        | 不开仓 |
| +1 | -1 | 0   | 1        | 1        | 不开仓 |

**优点**：
- ✅ 民主决策，相对公平
- ✅ 减少单一时间框架的误导
- ✅ 开仓机会适中

**缺点**：
- ⚠️ 所有时间框架权重相同（不合理）
- ⚠️ 可能忽略大周期的重要信息

**适用场景**：不确定各时间框架重要性时的备选方案

---

### 5. `average` - 原始平均法（最保守）

**理念**：所有时间框架平均后需超过低阈值

**逻辑**：
```python
avg_signal = mean(signal_1m, signal_5m, signal_15m)
if avg_signal > 0.1: BUY
if avg_signal < -0.1: SELL
```

**优点**：
- ✅ 最保守，假信号最少
- ✅ 适合高胜率策略

**缺点**：
- ⚠️ 开仓机会极少
- ⚠️ 可能错过大量有效信号
- ⚠️ 信号稀释严重

**适用场景**：极端风险厌恶，或作为对比基准

---

## 开仓机会对比

假设1000个时间点，各方法的开仓次数估算：

| 方法 | 预计开仓次数 | 胜率预期 | 盈亏比预期 | 风险等级 |
|------|-------------|----------|------------|----------|
| `average` | 50-100 | 65-70% | 1.8-2.2 | ⭐ 极低 |
| `hierarchical` | 100-200 | 60-65% | 1.5-2.0 | ⭐⭐ 低 |
| `majority` | 150-250 | 58-62% | 1.4-1.8 | ⭐⭐⭐ 中 |
| `weighted` | 200-300 | 55-60% | 1.3-1.7 | ⭐⭐⭐ 中高 |
| `independent` | 300-450 | 50-55% | 1.2-1.5 | ⭐⭐⭐⭐ 高 |

---

## 使用建议

### 使用方法

```python
from ml_trading.strategies.ml_strategy import MLTradingStrategy

# 1. 默认加权方法（推荐）
strategy = MLTradingStrategy(ensemble_method='weighted')

# 2. 保守的分层方法
strategy = MLTradingStrategy(ensemble_method='hierarchical')

# 3. 激进的独立信号法
strategy = MLTradingStrategy(ensemble_method='independent')

# 4. 多数投票法
strategy = MLTradingStrategy(ensemble_method='majority')

# 5. 原始平均法（对比基准）
strategy = MLTradingStrategy(ensemble_method='average')
```

### 选择建议

**如果你是...**

1. **趋势交易者** → 使用 `hierarchical`
   - 大周期定方向，小周期找时机
   - 遵循"顺势而为"

2. **波段交易者** → 使用 `weighted`（默认）
   - 平衡各时间框架
   - 适中的开仓频率

3. **日内交易者** → 使用 `independent`
   - 捕捉短期机会
   - 需要更好的风控

4. **新手** → 使用 `hierarchical` 或 `average`
   - 减少交易次数
   - 降低风险

5. **回测对比** → 同时测试所有方法
   - 找出最适合你数据的方法

### 组合建议

可以针对不同市场状态使用不同方法：

```python
# 趋势市：使用 hierarchical
if market_regime == 'trending':
    strategy = MLTradingStrategy(ensemble_method='hierarchical')

# 震荡市：使用 independent（捕捉短期波动）
elif market_regime == 'ranging':
    strategy = MLTradingStrategy(ensemble_method='independent')

# 不确定：使用 weighted
else:
    strategy = MLTradingStrategy(ensemble_method='weighted')
```

---

## 性能对比实验

### 实验设计

```python
import pandas as pd
from ml_trading.strategies.ml_strategy import MLTradingStrategy

methods = ['average', 'weighted', 'hierarchical', 'independent', 'majority']
results = {}

for method in methods:
    print(f"\n{'='*60}")
    print(f"Testing method: {method}")
    print(f"{'='*60}")
    
    strategy = MLTradingStrategy(ensemble_method=method)
    strategy.train_strategy()
    signals = strategy.generate_signals()
    
    # 统计指标
    total_signals = (signals['discrete_signal'] != 0).sum()
    long_signals = (signals['discrete_signal'] == 1).sum()
    short_signals = (signals['discrete_signal'] == -1).sum()
    
    results[method] = {
        'total_trades': total_signals,
        'long_trades': long_signals,
        'short_trades': short_signals,
        'trade_ratio': total_signals / len(signals) * 100
    }

# 对比结果
comparison_df = pd.DataFrame(results).T
print("\n" + "="*80)
print("ENSEMBLE METHODS COMPARISON")
print("="*80)
print(comparison_df)
```

---

## 总结

**核心观点**：

1. ✅ **原始 `average` 方法确实存在问题** - 会大大减少开仓机会
   
2. ✅ **不同方法适合不同交易风格**：
   - 保守型 → `average` 或 `hierarchical`
   - 平衡型 → `weighted`（推荐默认）
   - 激进型 → `independent`

3. ✅ **建议先用 `weighted` 测试**，然后根据回测结果调整

4. ✅ **记得同时优化风控参数**，激进的集成方法需要更严格的风控

---

## 后续改进方向

1. **动态集成**：根据市场状态自动切换方法
2. **信号强度过滤**：只取高置信度信号
3. **时间框架动态权重**：根据近期表现调整权重
4. **机器学习元模型**：学习何时该用哪种集成方法

