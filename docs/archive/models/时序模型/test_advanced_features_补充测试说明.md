# test_advanced_features.py 补充测试说明

## 已添加的四种核心测试

### ✅ 1. 无未来函数测试（⭐⭐⭐⭐⭐ 必须）

为 GARCH、DTW、EVT 三个特征类各添加了 `test_*_features_no_future_leak` 测试：

**测试逻辑**：
- 计算第一次特征
- 修改未来数据（t=150 或 t=70 之后）
- 重新计算特征
- 验证前 N 个时间点的特征值不受未来数据影响（差异 < 1e-6）

**测试方法**：
```python
def test_garch_features_no_future_leak(self, sample_data_single_asset):
    # 计算第一次
    result1 = extract_garch_features(df, ...)
    
    # 修改未来数据
    df_future_modified = df.copy()
    df_future_modified.loc[df_future_modified.index[150]:, "close"] *= 2.0
    
    # 重新计算
    result2 = extract_garch_features(df_future_modified, ...)
    
    # 验证前 100 个时间点不受影响
    assert (result1.loc[check_idx] - result2.loc[check_idx]).abs().max() < 1e-6
```

### ✅ 2. 多资产归一化测试（⭐⭐⭐⭐ 必须，已加强）

原有的 `test_*_features_normalization_multi_asset` 测试已加强：

**加强内容**：
- 添加了均值差异检查
- 验证不同资产的特征分布是否对齐
- 添加了更详细的错误信息

**GARCH 加强**：
```python
# 检查不同资产的均值差异不应该太大（归一化后应该对齐）
mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
assert mean_range < 0.5, f"{col} 在不同资产间的均值差异过大"
```

**DTW 加强**：
```python
assert mean_range < 10.0, (
    f"{col} 在不同资产间的均值差异过大: {mean_range:.4f}，"
    f"归一化可能有问题。各资产均值: {by_symbol['mean'].to_dict()}"
)
```

**EVT 加强**：
```python
mean_range = by_symbol["mean"].max() - by_symbol["mean"].min()
assert mean_range < 1.0, f"{col} 在不同资产间的均值差异过大"
```

### ✅ 3. 流式 vs 批量一致性测试（⭐⭐⭐⭐ 强烈推荐）

为 GARCH、DTW、EVT 三个特征类各添加了 `test_*_features_streaming_vs_batch` 测试：

**测试逻辑**：
- 批量计算：一次性处理所有数据
- 流式计算：分块处理（模拟在线推理）
- 验证两种方式的结果一致（差异 < 1e-5）

**测试方法**：
```python
def test_garch_features_streaming_vs_batch(self, sample_data_single_asset):
    # 批量计算
    batch_result = extract_garch_features(df, ...)
    
    # 流式计算（分块处理）
    chunk_size = 50
    streaming_results = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size].copy()
        chunk_result = extract_garch_features(chunk, ...)
        streaming_results.append(chunk_result)
    
    streaming_result = pd.concat(streaming_results, axis=0)
    
    # 比较关键特征（跳过前 window 行）
    assert (batch_result.iloc[skip_rows:] - streaming_result.iloc[skip_rows:]).abs().max() < 1e-5
```

### ✅ 4. 特征相关性衰减测试（⭐⭐⭐ 可选但高价值）

为 GARCH、DTW、EVT 三个特征类各添加了 `test_*_features_correlation_decay` 测试：

**测试逻辑**：
- 计算不同窗口大小的特征
- 验证不同窗口特征之间的相关性
- 确保相关性平滑，无断崖式下降

**GARCH 测试**：
```python
# 不同窗口的 persistence 应该高度相关（>0.5）
windows = [30, 60, 90]
# 验证 window_30 与 window_60 的相关性 > 0.5
# 验证 window_60 与 window_90 的相关性 > 0.5
```

**DTW 测试**：
```python
# 不同窗口的 DTW 距离应该有一定相关性（>0.3 或 <-0.3）
windows = [10, 20, 30]
# 注意：DTW 距离可能受窗口大小影响较大，所以阈值较低
```

**EVT 测试**：
```python
# 不同窗口的 tail_shape 应该有一定相关性（>0.3 或 <-0.3）
windows = [60, 120, 180]
```

## 测试覆盖情况总结

| 测试类型 | GARCH | DTW | EVT | 状态 |
|---------|-------|-----|-----|------|
| 无未来函数 | ✅ | ✅ | ✅ | 已添加 |
| 多资产归一化 | ✅ 加强 | ✅ 加强 | ✅ 加强 | 已加强 |
| 流式 vs 批量 | ✅ | ✅ | ✅ | 已添加 |
| 相关性衰减 | ✅ | ✅ | ✅ | 已添加 |

## 运行测试

```bash
# 运行所有测试
pytest tests/test_advanced_features.py -v

# 运行特定测试
pytest tests/test_advanced_features.py::TestGARCHFeatures::test_garch_features_no_future_leak -v
pytest tests/test_advanced_features.py::TestGARCHFeatures::test_garch_features_streaming_vs_batch -v
pytest tests/test_advanced_features.py::TestGARCHFeatures::test_garch_features_correlation_decay -v
```

## 注意事项

1. **流式 vs 批量测试**：由于分块计算可能导致边界处理略有不同，允许一定误差（1e-5）
2. **相关性衰减测试**：不同特征类型的相关性阈值不同（GARCH >0.5，DTW/EVT >0.3）
3. **多资产归一化测试**：不同特征类型的均值差异阈值不同（GARCH <0.5，DTW <10.0，EVT <1.0）

## 下一步建议

1. 运行测试确保所有测试通过
2. 根据实际数据调整阈值（如果需要）
3. 考虑为其他特征提取函数（如 extended_volatility_features）添加类似测试

