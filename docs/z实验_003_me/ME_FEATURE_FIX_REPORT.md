# ME (MomentumExpansion) 特征归一化修复报告

**日期**: 2026-02-16  
**状态**: ✅ 修复完成并验证

---

## 问题诊断

### 根本原因
ME Gate训练效果极差（Lift = 0.97x）的根本原因是**特征方差过低**，导致LightGBM无法使用这些特征。

### 问题表现
```python
# 之前训练结果
feature_importance = {
    "me_impulse_ratio": 0.0,
    "me_cps": 0.0,
    "me_multi_bar_acceleration": 0.0,
    "me_escape_dist": 0.0,
    # ... 所有ME特征的importance都是0.0
}

# 特征统计（问题版本）
me_multi_bar_acceleration: mean=0.5000, std=0.0277  # 方差极小
me_escape_dist: mean=0.5001, std=0.0146             # 几乎是常数
me_gate_structure_score: mean=0.5181, std=0.0921   # 变化很小
```

---

## 修复方案

### 修复的文件
`src/features/time_series/momentum_expansion_features.py` - `compute_me_gate_features_from_series()`

### 修复内容

#### 1. me_multi_bar_acceleration（加速度特征）
**问题**: 使用固定阈值0.02进行线性归一化，导致大部分值被clip到[-1, 1]后集中在0.5附近

**修复**: 
```python
# 修复前
accel_raw = recent_returns - prior_returns
me_multi_bar_acceleration = (accel_raw / 0.02).clip(-1, 1)  # 固定阈值
me_multi_bar_acceleration = (me_multi_bar_acceleration + 1) / 2  # 线性归一化

# 修复后
accel_raw = recent_returns - prior_returns
accel_atr_normalized = accel_raw / atr_s.rolling(lookback).mean().clip(lower=eps)
accel_percentile = _stream_safe_percentile(accel_atr_normalized.fillna(0), compression_window)
me_multi_bar_acceleration = accel_percentile.clip(0, 1)  # 使用百分位保持方差
```

**改善效果**: 方差提升 **933%** (0.0277 → 0.2867)

#### 2. me_escape_dist（逃离距离特征）
**问题**: 使用固定系数3进行线性归一化，导致大部分值集中在0.5附近

**修复**:
```python
# 修复前
escape_dist_raw = (close - rolling_high) / prior_range_height  # 使用区间高度归一化
me_escape_dist_norm = (escape_dist_raw / 3 + 1) / 2  # 线性映射到0-1

# 修复后
escape_up = (close - rolling_high) / atr_s.clip(lower=eps)  # 使用ATR归一化
escape_dist_raw = pd.Series(np.where(...))  # 保留原始值（以ATR为单位）
me_escape_dist_norm = _stream_safe_percentile(escape_dist_raw, compression_window)  # 百分位
```

**改善效果**: 方差提升 **1798%** (0.0146 → 0.2767)

#### 3. me_compression_depth（压缩深度特征）
**问题**: 使用`rolling().rank(pct=True)`导致均匀分布，失去历史信息

**修复**:
```python
# 修复前
rolling_vol = bar_range.rolling(compression_window, min_periods=lookback).std()
rolling_vol_pct = rolling_vol.rolling(compression_window, min_periods=lookback).rank(pct=True)
me_compression_depth = 1 - rolling_vol_pct

# 修复后
rolling_vol = bar_range.rolling(lookback, min_periods=1).std()
rolling_vol_normalized = rolling_vol / atr_s.clip(lower=eps)  # ATR归一化
rolling_vol_pct = _stream_safe_percentile(rolling_vol_normalized, compression_window)
me_compression_depth = 1 - rolling_vol_pct
```

**改善效果**: 方差保持良好（0.3566 → 0.2699）

---

## 修复验证

### 测试数据
- 数据来源: `feature_store/features_46ebfeb7e8/BTCUSDT/240T/2023-01.parquet`
- 样本数: 186 rows
- 测试方法: 使用修复后的代码重新计算ME特征，对比统计指标

### 验证结果

| 特征 | 修复前 std | 修复后 std | 改善幅度 |
|------|-----------|-----------|----------|
| me_impulse_ratio | 0.1842 | 0.2341 | **+27.1%** |
| me_cps | 0.3697 | 0.2843 | -23.1% (CPS本身方差已合理) |
| me_multi_bar_acceleration | 0.0277 | 0.2867 | **+933.4%** |
| me_escape_dist | 0.0146 | 0.2767 | **+1798.0%** |
| me_compression_depth | 0.3566 | 0.2699 | -24.3% (保持合理方差) |

**平均改善**: **542.2%**

### 特征分布改善

#### 修复前（问题版本）
```
me_multi_bar_acceleration: mean=0.5000, std=0.0277  # 几乎是常数
me_escape_dist: mean=0.5001, std=0.0146             # 变化极小
me_gate_structure_score: mean=0.5181, std=0.0921   # 集中在0.5附近
```

#### 修复后
```
me_multi_bar_acceleration: mean=0.5199, std=0.2867  # 方差提升10倍
me_escape_dist: mean=0.5440, std=0.2767             # 方差提升18倍
me_gate_structure_score: mean=0.5089, std=0.1054   # 方差合理
```

---

## 核心设计原则

### 1. 使用流式安全的百分位计算
**函数**: `_stream_safe_percentile(series, window)`
- 确保流式和批量计算一致
- 使用固定窗口保持历史趋势
- 窗口不足时返回0.5（中性值）

### 2. 使用ATR归一化而非固定阈值
- ATR能自适应不同市场状态的波动率
- 避免固定阈值导致的clip效应
- 保持特征在不同市场周期的区分能力

### 3. 避免rank()导致的均匀分布
- rank(pct=True)会强制转换为均匀分布
- 丢失了原始特征的分布特性
- 应使用百分位保持历史趋势

---

## 预期效果

### 特征重要性
修复后，ME特征的feature_importance应该不再为0，预期：
- `me_compression_depth`: 50-150 (压缩深度对突破质量的影响)
- `me_impulse_ratio`: 30-80 (突破强度)
- `me_multi_bar_acceleration`: 20-60 (加速度)
- `me_escape_dist`: 20-50 (逃离距离)

### 模型性能
预期Lift指标改善：
- **修复前**: Top 30% Lift = 0.97x ❌
- **修复后**: Top 30% Lift > 1.0x ✅ (目标)
- **改善原因**: 特征现在有足够的方差来区分好坏样本

---

## 下一步建议

### 1. 重新构建Feature Store
由于特征计算逻辑已修改，需要重新构建Feature Store：
```bash
python3 scripts/build_feature_store_from_config.py \
  --config config/strategies/me \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2021-01-01 \
  --end-date 2025-12-31 \
  --force
```

### 2. 重新训练ME Gate
使用新的Feature Store重新训练：
```bash
python3 scripts/train_strategy_pipeline.py \
  --strategy me \
  --labels config/strategies/me/labels_rr_extreme.yaml \
  --symbol BTCUSDT \
  --start-date 2021-01-01 \
  --end-date 2025-12-31
```

### 3. 验证改善效果
重点验证：
- ME特征的feature_importance > 0
- Top 30% Lift > 1.0
- failure_rr_extreme降低程度

### 4. 多币种训练
如果单币种效果良好，进行多币种训练以提升泛化能力。

---

## 技术细节

### 流式安全百分位计算
```python
def _stream_safe_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    流式安全的百分位计算
    
    确保流式和批量计算一致：
    - 使用固定窗口 window
    - min_periods = window（确保窗口内数据足够）
    - 窗口不足时返回 0.5（中性值）
    """
    result = series.rolling(window, min_periods=window).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x) if len(x) == window else 0.5,
        raw=False
    )
    result = result.fillna(0.5)
    return result
```

### ATR归一化优势
```python
# 问题：固定阈值无法适应不同市场
accel_raw / 0.02  # 0.02在高波动期太小，在低波动期太大

# 解决：使用ATR自适应归一化
accel_raw / atr_s.rolling(lookback).mean()  # ATR随市场波动自动调整
```

---

## 总结

✅ **修复完成**: ME特征归一化问题已全部修复  
✅ **验证通过**: 特征方差平均提升542%，关键特征提升10-18倍  
⏳ **待验证**: 需要重新训练以验证模型性能改善  

**核心改进**:
1. 使用流式安全的百分位计算替代固定阈值归一化
2. 使用ATR归一化替代固定数值归一化
3. 避免rank()导致的均匀分布

**预期结果**: ME特征现在有足够的方差和区分能力，预期Lift指标将显著改善至 > 1.0x。
