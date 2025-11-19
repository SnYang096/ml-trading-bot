# 数据泄漏排查报告

## 测试结果

### Random Walk 测试
- **Simple 模式（OLS）**: IC = -0.0026（接近 0，✅ 无泄漏）
- **Full 模式（LightGBM）**: IC = 0.0404（> 0.03，⚠️ 有泄漏）

### Feature-Future Correlation 测试
- 50% 的特征有可疑相关性（|corr| > 0.1）
- 最大相关性：0.1672（`atr_zscore_w288`）

## 已修复的问题

### ✅ 1. 标签包含当前 bar 的 close
**问题**: `future_return[t] = (close[t+horizon] - close[t]) / close[t]` 依赖当前 bar 的 close

**修复**: 使用 `close[t+1]` 作为起始价格（假设在 t+1 开盘价成交）
```python
close_next = df[price_col].shift(-1)
df["future_return"] = close_next.pct_change(hold_period).shift(-hold_period)
```

**修复文件**:
- `rank_ic_trainer.py`
- `train.py`
- `rolling.py`
- `safe_multi_asset_preprocessing.py`
- `data_leakage_detector.py`
- `multi_tf_pipeline.py`
- `auto_rolling_update.py`

### ✅ 2. rolling_vol 索引对齐
**问题**: `rolling_vol` 计算后索引与原始数据不对齐

**修复**: 使用 `reindex` 保持索引对齐
```python
rolling_vol = rets.rolling(window=lookback_window, min_periods=min_samples).std()
return rolling_vol.reindex(series.index)
```

## 已验证安全的部分

### ✅ 1. 全局标准化
- `volatility_normalized_target` 使用滚动波动率，不是全局标准化
- ✅ 安全

### ✅ 2. 分类标签
- `historical_quantile_label` 使用滚动窗口
- ✅ 安全

### ✅ 3. 权重和 tradable mask
- Random walk 测试中已设置为统一值
- ✅ 不影响测试结果

## 待解决的问题

### ⚠️ 1. LightGBM 在随机数据上的过拟合
**现象**: Simple 模式（OLS）IC ≈ 0，但 Full 模式（LightGBM）IC = 0.0404

**可能原因**:
1. 树模型在随机数据上仍可能找到微弱模式（即使有正则化）
2. 样本/特征比：20 features / 1892 samples ≈ 94.6 samples/feature（可能偏少）
3. 统计上的偶然性

**当前正则化参数**:
```python
num_leaves: min(15, max(7, int(n_samples / 100)))  # 约 15
learning_rate: 0.02
feature_fraction: 0.7
bagging_fraction: 0.7
min_data_in_leaf: max(20, int(n_samples / 50))  # 约 38
min_gain_to_split: 0.1
lambda_l1: 0.1
lambda_l2: 0.1
max_depth: 5
```

**建议**:
1. 进一步增加正则化强度（更小的 `num_leaves`，更大的 `min_data_in_leaf`）
2. 或者接受这个结果（IC = 0.04 可能是 LightGBM 的统计噪声）

### ⚠️ 2. Feature-Future Correlation 警告
**现象**: 50% 的特征有可疑相关性（|corr| > 0.1）

**可能原因**:
1. 这些特征确实有预测能力（非泄漏）
2. 特征计算存在边界问题
3. 或者这是正常的（技术指标与未来收益的相关性）

**建议**:
- 检查高相关性特征的计算逻辑
- 确认这些特征是否真的只使用历史数据

## 结论

1. **标签计算已修复**: `future_return` 不再依赖当前 bar 的 close
2. **rolling_vol 索引对齐已修复**: 确保数据对齐正确
3. **LightGBM 过拟合**: IC = 0.04 虽然 > 0.03，但相对较小，可能是树模型在随机数据上的正常行为
4. **Feature-Future Correlation**: 需要进一步检查高相关性特征的计算逻辑

## 下一步行动

1. 进一步增加 LightGBM 正则化强度，测试是否能降低 IC
2. 检查高相关性特征（如 `atr_zscore_w288`）的计算逻辑
3. 或者接受当前结果，在实际训练中监控 OOS 性能

