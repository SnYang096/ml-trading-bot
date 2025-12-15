# POC/HAL 独立特征重构说明

## 重构目标

将 `poc`、`hal_high`、`hal_low`、`hal_mid` 等作为独立的基础特征，避免多个特征重复计算这些列。

## 重构内容

### 1. 创建独立的基础特征 `poc_hal_features`

**配置文件**：`config/feature_dependencies.yaml`

```yaml
poc_hal_features:
  module: baseline
  compute_func: compute_poc_hal_features
  dependencies: ["wpt_price_reconstructed"]
  required_columns: ["high", "low", "close", "volume", "wpt_price_reconstructed"]
  output_columns: ["poc", "hal_high", "hal_low", "hal_mid"]
  category: sr_structure
  description: "POC (Point of Control) 和 HAL (Value Area) 基础特征（基于 WPT 重构价格）"
  compute_params:
    poc_window: 160
    price_col: "wpt_price_reconstructed"
  pass_full_df: true
  notes: "独立的基础特征，可被其他特征复用，避免重复计算"
```

**实现函数**：`src/features/loader/feature_wrappers.py`

```python
def compute_poc_hal_features(
    df: pd.DataFrame,
    poc_window: int = 160,
    price_col: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """计算 POC 和 HAL 特征"""
    # ... 实现代码
```

### 2. 更新 `sqs_hal_high` 和 `sqs_hal_low` 的依赖关系

**之前**：
```yaml
sqs_hal_high:
  dependencies: ["atr", "wpt_price_reconstructed"]
  output_columns: ["sqs_hal_high", "hal_high", "poc"]  # poc 和 hal_high 作为副产品

sqs_hal_low:
  dependencies: ["atr", "wpt_price_reconstructed"]
  output_columns: ["sqs_hal_low", "hal_low", "poc"]  # poc 和 hal_low 作为副产品
```

**现在**：
```yaml
sqs_hal_high:
  dependencies: ["atr", "wpt_price_reconstructed", "poc_hal_features"]  # 依赖 poc_hal_features
  output_columns: ["sqs_hal_high"]  # 只输出自己的特征

sqs_hal_low:
  dependencies: ["atr", "wpt_price_reconstructed", "poc_hal_features"]  # 依赖 poc_hal_features
  output_columns: ["sqs_hal_low"]  # 只输出自己的特征
```

### 3. 更新 `sr_strength_max` 的依赖关系

**之前**：
```yaml
sr_strength_max:
  dependencies: ["atr", "sqs_hal_high", "sqs_hal_low", "wpt_price_reconstructed"]
  # 依赖 sqs_hal_high 和 sqs_hal_low 来获取 poc, hal_high, hal_low
```

**现在**：
```yaml
sr_strength_max:
  dependencies: ["atr", "poc_hal_features", "wpt_price_reconstructed"]
  # 直接依赖 poc_hal_features 来获取 poc, hal_high, hal_low
```

### 4. 更新计算函数

**`compute_sqs_hal_high` 和 `compute_sqs_hal_low`**：
- 移除创建 `poc` 和 `hal_high`/`hal_low` 的逻辑
- 检查这些列是否已存在（应该来自 `poc_hal_features` 依赖）
- 如果不存在，打印警告并尝试计算（向后兼容）

**`compute_sr_strength_max`**：
- 保持自动修复机制（防御性编程）
- 但现在主要依赖 `poc_hal_features` 提供的列

## 优势

1. **避免重复计算**：
   - `poc`、`hal_high`、`hal_low` 只计算一次
   - 其他特征通过依赖关系复用

2. **清晰的依赖关系**：
   - 通过配置文件明确声明依赖
   - 依赖解析器自动处理计算顺序

3. **更好的可维护性**：
   - 职责分离：`poc_hal_features` 负责基础特征，其他特征负责衍生特征
   - 代码更清晰，更容易理解和维护

4. **解决重复列名问题**：
   - 从根本上解决多个特征输出相同列名的问题
   - 不再需要复杂的合并逻辑来处理重复列

## 向后兼容

- 计算函数中保留了自动修复机制（如果列不存在，尝试计算）
- 如果依赖关系配置错误，函数会打印警告并尝试计算
- 不会破坏现有的功能

## 迁移指南

如果其他特征也需要使用 `poc` 或 `hal` 列：

1. 在 `dependencies` 中添加 `poc_hal_features`
2. 在 `required_columns` 中添加需要的列（如 `poc`、`hal_high`、`hal_low`）
3. 从 `output_columns` 中移除这些列（如果之前有）
4. 在计算函数中，检查这些列是否已存在，如果不存在则打印警告

## 相关文件

- `config/feature_dependencies.yaml`: 特征配置
- `src/features/loader/feature_wrappers.py`: 特征计算函数
- `src/features/loader/feature_function_mapping.py`: 函数映射表
- `src/features/time_series/baseline_features.py`: 底层实现

