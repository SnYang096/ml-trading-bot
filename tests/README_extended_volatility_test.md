# 扩展波动率特征测试

## 概述

`test_extended_volatility_features.py` 测试 `extract_extended_volatility_features` 函数是否正确生成所有41个波动率特征。

## 测试内容

### 1. 基本功能测试
- ✅ 验证所有41个特征都正确生成
- ✅ 验证特征名称与配置一致
- ✅ 验证返回的DataFrame格式正确

### 2. 数据质量测试
- ✅ 验证特征值非NaN、非Inf
- ✅ 验证特征值在合理范围内
- ✅ 验证特征之间的逻辑关系

### 3. 边界情况测试
- ✅ 测试没有ATR列的情况
- ✅ 测试自定义lag_periods
- ✅ 测试自定义window参数
- ✅ 测试小数据集和极端值

## 运行方式

### 方式1: 使用Makefile（推荐）

```bash
make test-extended-volatility-features
```

这会使用Makefile中定义的 `DOCKER_RUN_NO_TTY` 在Docker容器中运行测试。

### 方式2: 手动运行Docker命令

如果需要手动运行（与Makefile中的方式一致）：

```bash
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  --user $(id -u):$(id -g) \
  -e PYTHONPATH=/workspace/src \
  -e PYTHONUNBUFFERED=1 \
  -v $(pwd):/workspace \
  -v $(pwd)/data/parquet_data:/workspace/data/parquet_data \
  -w /workspace \
  --shm-size=8gb \
  hansenlovefiona017/lightgbm-runtime:v0.0.5 \
  python3 -m pytest tests/test_extended_volatility_features.py -v --tb=short
```

### 方式3: 在Docker容器内直接运行pytest

```bash
# 在Docker容器内
python3 -m pytest tests/test_extended_volatility_features.py -v
```

## 期望的特征列表（41个）

### 1. 多尺度历史波动率 (4个)
- `vol_raw_5`, `vol_raw_10`, `vol_raw_20`, `vol_raw_60`

### 2. ATR衍生特征 (15个)
- `vol_atr_norm`
- `vol_atr_ma_5/10/20`
- `vol_atr_std_5/10/20`
- `vol_atr_max_5/10/20`
- `vol_atr_min_5/10/20`
- `vol_atr_ratio_20`
- `vol_atr_change`, `vol_atr_change_abs`

### 3. 滞后特征 (3个)
- `vol_lag_1`, `vol_lag_2`, `vol_lag_3`

### 4. 趋势特征 (4个)
- `vol_slope_5/10/20`
- `vol_accel`

### 5. 移动平均特征 (6个)
- `vol_ma_5/10/20`
- `vol_ema_5/10/20`

### 6. Regime特征 (2个)
- `vol_zscore`
- `vol_percentile_approx`

### 7. 范围特征 (4个)
- `vol_range_10/20`
- `vol_range_pos_10/20`

### 8. 动量特征 (3个)
- `vol_mom_3/5/10`

## 测试验证点

1. **特征完整性**: 所有41个特征都必须存在
2. **数据有效性**: 特征值必须是非NaN、非Inf的有限值
3. **数值范围**: 
   - `vol_raw_*` 应该在 [0, 1.0) 范围内
   - `vol_atr_norm` 应该在 [0, 0.1) 范围内
   - `vol_percentile_approx` 应该在 [0, 1] 范围内
   - `vol_range_pos_*` 应该在 [0, 1] 范围内
4. **逻辑关系**: 
   - `vol_range_*` 应该 >= 0
   - `vol_atr_ratio_20` 应该 > 0

## 故障排查

如果测试失败：

1. **检查依赖**: 确保在Docker环境中运行
2. **检查特征名称**: 确保与 `config/feature_dependencies.yaml` 一致
3. **检查数据**: 确保模拟数据生成正确
4. **查看详细输出**: 使用 `-v` 参数查看详细测试输出

## 示例输出

```
tests/test_extended_volatility_features.py::test_extract_extended_volatility_features_basic PASSED
tests/test_extended_volatility_features.py::test_extract_extended_volatility_features_values PASSED
tests/test_extended_volatility_features.py::test_extract_extended_volatility_features_ranges PASSED
tests/test_extended_volatility_features.py::test_extract_extended_volatility_features_without_atr PASSED
tests/test_extended_volatility_features.py::test_extract_extended_volatility_features_custom_lag_periods PASSED
tests/test_extended_volatility_features.py::test_extract_extended_volatility_features_custom_window PASSED
tests/test_extended_volatility_features.py::test_extract_extended_volatility_features_edge_cases PASSED
tests/test_extended_volatility_features.py::test_extract_extended_volatility_features_feature_relationships PASSED

======================== 8 passed in 2.34s ========================
```

