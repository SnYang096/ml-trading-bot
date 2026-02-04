# CVD Divergence V2: 连续化背离特征

## 概述

CVD Divergence V2 是对 V1（Bool 类型）的工业级改进，解决了原版特征对树模型不友好的问题。

### V1 问题

```python
# V1 输出 Bool 类型
cvd_bullish_divergence: 0 / 1  # 只有 2 个分裂点
cvd_bearish_divergence: 0 / 1  # 触发稀疏
cvd_divergence_strength: 0~1   # 仅在 divergence=1 时有值
```

### V2 改进

1. **连续化输出**：每 bar 都有值，模型可精细分裂
2. **Rank 替代 min-max**：避免极值抖动
3. **3 个工业级复合特征**：区分"健康背离" vs "反转背离"

---

## 输出特征

| 特征名 | 范围 | 语义 |
|--------|------|------|
| `cvd_divergence_score` | [-1, 1] | 背离得分：正=看涨背离（吸筹），负=看跌背离（派发） |
| `cvd_divergence_score_pct` | [0, 1] | 背离得分的历史百分位 |
| `price_position` | [0, 1] | 价格在窗口内的相对位置 |
| `trend_div_alignment` | [-1, 1] | 趋势-背离对齐度（顺趋势 vs 逆趋势） |
| `trend_div_tension` | [0, 1] | 趋势-背离张力（冲突强度） |
| `div_location_pressure` | [0, 1] | 背离位置压力（极端位置反转潜力） |

---

## 核心算法

### 1. 滚动百分位排名（处理重复值）

```python
def _rolling_percentile_rank(series, window):
    def _pct_rank(x):
        if len(x) < 2:
            return 0.5
        # 使用 <= 而非 <，处理重复值更稳定
        rank = (x <= x.iloc[-1]).sum() - 1
        return rank / (len(x) - 1)
    return series.rolling(window).apply(_pct_rank, raw=False)
```

**为什么用 `<=`？**
- 最小值 → 0
- 最大值 → 1
- 重复值（如 CVD plateau）→ 中性分布，不会系统性偏低

### 2. 背离得分

```python
price_position = _rolling_percentile_rank(close, w)
cvd_position = _rolling_percentile_rank(cvd, w)

# 正值：CVD 相对强，价格相对弱 → 看涨背离（吸筹）
# 负值：CVD 相对弱，价格相对强 → 看跌背离（派发）
divergence_score = (cvd_position - price_position).clip(-1, 1)
```

### 3. 三个工业级复合特征

```python
D = divergence_score
T = trend_strength      # [-1, 1]，正=上行趋势
P = price_position      # [0, 1]

# Feature 1: Trend–Divergence Alignment（方向轴）
# 正 = 顺趋势背离（健康），负 = 逆趋势背离（反转风险）
trend_div_alignment = D * T

# Feature 2: Trend–Divergence Tension（冲突强度）
# 使用 sqrt() 非线性：小冲突更敏感，大冲突不过饱和
trend_div_tension = sqrt(|D| * |T|)

# Feature 3: Divergence Location Pressure（位置压力）
# 高值 = 背离 + 极端位置（反转潜力高）
div_location_pressure = |D| * |P - 0.5| * 2
```

---

## 三个复合特征的交易语义

### Feature 1: `trend_div_alignment`（方向轴）

| 值 | 含义 | 策略 |
|----|------|------|
| 正值 | 顺趋势背离 | 健康回调/回踩，可加仓 |
| 负值 | 逆趋势背离 | 反转风险，减仓/观望 |
| ~0 | 无背离或无趋势 | 中性 |

**例子**：
- 上涨趋势（T > 0）+ 看涨背离（D > 0）→ alignment > 0 → 健康吸筹
- 上涨趋势（T > 0）+ 看跌背离（D < 0）→ alignment < 0 → 派发风险

### Feature 2: `trend_div_tension`（冲突强度）

| 值 | 含义 |
|----|------|
| 高 | 趋势与背离冲突强烈，市场在"拉锯" |
| 低 | 趋势与背离协调，或两者都弱 |

**sqrt 非线性的作用**：
- 小冲突（0.1 × 0.2 = 0.02）→ sqrt = 0.14（更敏感）
- 大冲突（0.8 × 0.8 = 0.64）→ sqrt = 0.8（不过饱和）

### Feature 3: `div_location_pressure`（位置压力）

| 值 | 含义 |
|----|------|
| 高 | 价格处于极端位置 + 有背离 → 反转潜力高 |
| 低 | 价格在中间区域，或无背离 |

---

## 配置示例

```yaml
# config/feature_dependencies.yaml
cvd_divergence_v2_f:
  module: interaction
  compute_func: compute_cvd_divergence_v2_from_series
  dependencies:
    - trend_r2_20_f
  required_columns:
    - close
    - cvd
  output_columns:
    - cvd_divergence_score
    - cvd_divergence_score_pct
    - price_position
    - trend_div_alignment
    - trend_div_tension
    - div_location_pressure
  compute_params:
    position_window: 50
    percentile_window: 540
  column_mappings:
    trend_strength: trend_r2_20
```

---

## 测试覆盖

测试文件：`tests/features/test_cvd_divergence_v2.py`

| 测试维度 | 测试内容 |
|----------|----------|
| 功能正确性 | 输出列完整、范围正确、语义正确 |
| 未来函数 | 修改未来数据不影响历史值 |
| 流式一致性 | 流式计算与批量计算结果一致 |

---

## 迁移指南

### 从 V1 迁移到 V2

| V1 特征 | V2 替代 | 说明 |
|---------|---------|------|
| `cvd_bullish_divergence` | `cvd_divergence_score > 0` | 连续值，可设阈值 |
| `cvd_bearish_divergence` | `cvd_divergence_score < 0` | 连续值，可设阈值 |
| `cvd_divergence_strength` | `abs(cvd_divergence_score)` | 每 bar 都有值 |
| - | `trend_div_alignment` | 新增：方向轴 |
| - | `trend_div_tension` | 新增：冲突强度 |
| - | `div_location_pressure` | 新增：位置压力 |

### 推荐策略

1. **树模型**：直接使用 6 个输出列，模型自动学习分裂点
2. **规则系统**：用 `cvd_divergence_score_pct` 设置阈值（如 > 0.8 触发）
3. **Router**：用 `trend_div_alignment` 区分健康背离 vs 反转背离
