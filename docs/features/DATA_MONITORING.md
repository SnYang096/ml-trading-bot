# 数据监控系统

## 目的

监控数据质量，追踪 inf 值的来源，区分是**源数据错误**还是**计算错误**。

## 监控点

### 1. 源数据加载后 (`data_utils.py`)

在 `load_raw_data` 函数中，重采样后立即检查：
- 检查列：`open`, `high`, `low`, `close`, `volume`
- 如果发现 inf，会打印：
  - 数据路径
  - 包含 inf 的列名
  - inf 的数量和百分比
  - 前几个 inf 的位置和值
  - 上下文数据（前后各3行）

**如果在这里发现 inf，说明是源数据问题！**

### 2. 训练流程中 (`train_strategy_pipeline.py`)

在 `train_strategy` 函数中，数据加载后立即检查：
- 调用 `check_source_data_quality` 进行全面检查
- 打印详细的质量报告

### 3. SR 强度计算前后 (`baseline_features.py`)

在 `_compute_boundary_strengths` 调用前后：
- **计算前**：检查 `open`, `high`, `low`, `close`, `volume`, `atr` 列
- **计算后**：检查所有边界强度特征列

如果计算前没有 inf，但计算后有 inf，说明是计算过程产生的问题。

### 4. Hurst 特征计算前 (`utils_hurst_features.py`)

在计算价格收益率前：
- 检查输入价格列是否有 inf
- 如果发现 inf，会打印详细信息

### 5. RSI 计算前 (`baseline_features.py`)

在 `compute_rsi` 函数中：
- 检查输入价格序列是否有 inf
- 如果发现 inf，会打印详细信息

## 监控输出示例

### 正常情况（无 inf）

```
   🔍 [DATA MONITOR] Checking source data quality...
      Data path: data/parquet_data
      Shape: (1000, 6)
      Columns: ['open', 'high', 'low', 'close', 'volume', 'atr']
   ✅ [DATA MONITOR] Source data quality check passed (no inf values)
```

### 发现 inf 值

```
   ⚠️  [DATA MONITOR] SOURCE_DATA @ after_load
      Column 'volume' contains 5 inf values (0.50%)
      First few inf indices: [Timestamp('2024-01-15 10:00:00'), ...]
      First few inf values: [inf, inf, ...]
      Context around first inf (row 100):
         volume
         2024-01-15 09:00:00  1000.0
         2024-01-15 10:00:00  inf
         2024-01-15 11:00:00  1000.0
      ⚠️  WARNING: Inf found in source column 'volume'!
         This suggests a problem with the raw data, not feature calculation.
         Adjacent column 'close' is OK: 100.0
```

### 计算过程中产生的 inf

```
   ⚠️  [FEATURE MONITOR] BASELINE_FEATURES @ after_sr_strength_calc
      Column 'sqs_poc' gained 10 new inf values
```

## 如何判断问题来源

### 源数据问题

**特征**：
- 在 `after_load` 或 `after_resample_*` 阶段发现 inf
- 出现在基础列：`open`, `high`, `low`, `close`, `volume`
- 相邻列可能也有 inf

**可能原因**：
1. 数据文件损坏
2. 数据加载逻辑错误
3. 重采样聚合函数产生 inf（如 `sum()` 在特定情况下）

**解决方法**：
1. 检查原始数据文件
2. 检查数据加载代码
3. 检查重采样逻辑

### 计算问题

**特征**：
- 源数据检查通过（无 inf）
- 在特征计算后出现 inf
- 出现在衍生特征列

**可能原因**：
1. 除法操作（分母为 0 或极小值）
2. 对数操作（输入为 0 或负数）
3. 滚动统计（窗口内数据异常）

**解决方法**：
1. 检查计算逻辑
2. 添加输入验证
3. 添加除零保护

## 监控代码位置

- `src/features/utils/data_monitor.py`: 监控工具函数
- `src/data_tools/data_utils.py`: 数据加载监控
- `scripts/train_strategy_pipeline.py`: 训练流程监控
- `src/features/time_series/baseline_features.py`: 特征计算监控
- `src/features/time_series/utils_hurst_features.py`: Hurst 特征监控

## 使用建议

1. **首次运行**：观察监控输出，了解数据质量
2. **发现问题**：根据监控输出判断是源数据问题还是计算问题
3. **修复后**：重新运行，确认问题已解决
4. **生产环境**：可以设置 `raise_on_inf=True` 来阻止训练继续（如果发现源数据 inf）

## 注意事项

- 监控代码使用 `try-except` 包裹，不会影响主流程
- 监控输出会打印到控制台，便于实时查看
- NaN 值不会被报告为错误（因为可能是正常的，如 warmup 期间）
- 只监控 inf 值，因为 NaN 可能是预期的（如数据不足时）

