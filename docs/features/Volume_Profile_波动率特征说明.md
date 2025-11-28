# Volume Profile 波动率特征说明

## 概述

从 WPT 降噪后的 Volume Profile 中提取 6 个波动率预测相关特征，用于提升波动率模型的预测准确性。

## 特征列表

| 特征名 | 说明 | 经济逻辑 | 预测意义 |
|--------|------|----------|----------|
| `vp_width_ratio` | Value Area Width / Full Range（市场共识强度） | 越小 → 市场在窄区间达成共识 | 波动可能低（压缩后爆发前） |
| `vp_poc_deviation` | 当前价格 vs POC 的标准化偏离 | 绝对值大 → 远离价值中枢 | 回归动力强 或 趋势加速 → 波动↑ |
| `vp_skewness` | 成交量分布偏度（趋势倾向） | 绝对值大 → 趋势强 | 波动可能持续 |
| `vp_entropy` | 信息熵（不确定性） | 越大 → 成交分布分散，无明确方向 | 多空博弈激烈 → 波动↑ |
| `vp_lv_ratio` | 低成交量区域比例（LVN） | 越大 → 存在多个低成交量"真空带" | 价格一旦进入，会快速穿越 → 短期波动↑ |
| `vp_hv_ratio` | 高成交量区域比例（HVN） | 越大 → 支撑/阻力密集 | 波动可能受限 |

## 实现位置

### 1. 核心函数

**文件**: `src/features/time_series/utils_volatility_features.py`

- `extract_volatility_features_from_vp()`: 从 `VolumeProfileResult` 提取 6 个标量特征
- `extract_volume_profile_volatility_features()`: 滚动窗口计算，将特征添加到 DataFrame

### 2. 配置文件

**文件**: `config/feature_dependencies.yaml`

```yaml
volume_profile_volatility_features:
  module: enhanced
  compute_func: extract_volume_profile_volatility_features
  dependencies: []
  required_columns: ["close", "volume"]
  output_columns: ["vp_width_ratio", "vp_poc_deviation", "vp_skewness", "vp_entropy", "vp_lv_ratio", "vp_hv_ratio"]
  category: volatility
  description: "Volume Profile 波动率特征（从 WPT 降噪后的 Volume Profile 提取）"
  compute_params:
    window: 100
    wavelet: "db4"
    level: 4
```

**文件**: `config/volatility_model.yaml`

```yaml
- name: volume_profile_volatility
  feature_name: volume_profile_volatility_features
  required: false
  columns:
    - vp_width_ratio
    - vp_poc_deviation
    - vp_skewness
    - vp_entropy
    - vp_lv_ratio
    - vp_hv_ratio
```

### 3. 函数映射

**文件**: `src/features/loader/feature_function_mapping.py`

已添加 `extract_volume_profile_volatility_features` 到 `FEATURE_FUNCTION_MAP`。

### 4. 自动集成

**文件**: `src/time_series_model/pipeline/training/volatility_model_config.py`

在 `prepare_volatility_model_data()` 中自动检测并计算 Volume Profile 波动率特征（如果尚未存在）。

## 使用方式

### 手动调用

```python
from src.features.time_series.utils_volatility_features import extract_volume_profile_volatility_features

# 在 DataFrame 上计算特征
df_with_vp_features = extract_volume_profile_volatility_features(
    df,
    price_col="close",
    volume_col="volume",
    window=100,
    wavelet="db4",
    level=4,
)
```

### 通过特征加载器

```python
from src.features.loader import StrategyFeatureLoader

loader = StrategyFeatureLoader()
df = loader.load_features_from_requested(
    df,
    requested_features=["volume_profile_volatility_features"],
    fit=True,
)
```

### 在波动率模型中自动使用

波动率模型训练时，如果配置了 `volume_profile_volatility` 组，会自动计算这些特征。

## 性能说明

- **计算成本**: 单次 WPT Volume Profile 约 2~5ms（4H 数据）
- **窗口大小**: 默认 100 根 K 线（约 16.7 天，4H 数据）
- **适用场景**: 中低频策略（日线/4H），不适用于分钟级高频策略

## 特征组合建议

这些特征与以下特征结合使用，可显著提升对"波动率拐点"的识别能力：

- `vol_zscore`: 波动率 Z-score（regime indicator）
- `vol_slope_10`: 波动率趋势斜率
- `wpt_price_energy_high_ratio`: 高频能量占比（噪声强度）
- `vp_entropy`: 信息熵（不确定性）

## 注意事项

1. **相位滞后**: WPT 特征具有约 `window//2` 的相位滞后，适用于中低频策略
2. **数据要求**: 需要 `close` 和 `volume` 列
3. **NaN 处理**: 使用前向填充，剩余 NaN 填充为 0.0
4. **数值稳定性**: 所有计算都包含异常值处理和边界检查

## 参考

- 原始实现参考: `src/features/time_series/utils_volume_profile.py` 中的 `compute_wpt_volume_profile()`
- 波动率模型配置: `config/volatility_model.yaml`
- 特征依赖配置: `config/feature_dependencies.yaml`

