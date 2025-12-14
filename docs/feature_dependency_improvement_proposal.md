# 特征依赖关系改进方案

## 问题分析

当前实现中，`sqs_hal_high` 和 `sqs_hal_low` 在计算时会创建 `hal_high`、`hal_low`、`poc` 列作为**副产品**（side effects），但这些副产品没有在配置文件中明确声明。这导致：

1. **依赖关系不清晰**：`sr_strength_max` 需要 `hal_high`、`hal_low`、`poc`，但配置中只声明了依赖 `sqs_hal_high` 和 `sqs_hal_low`
2. **自动修复机制被过度依赖**：代码中的自动修复机制应该只是防御性措施，而不是主要机制
3. **测试覆盖了错误的场景**：测试验证了"缺少依赖时的自动修复"，但理想情况下，依赖应该通过配置文件正确解析

## 当前状态

### 配置文件声明
```yaml
sqs_hal_high:
  output_columns: ["sqs_hal_high"]  # 只声明了主输出，没有声明副产品
  
sqs_hal_low:
  output_columns: ["sqs_hal_low"]  # 只声明了主输出，没有声明副产品

sr_strength_max:
  dependencies: ["atr", "sqs_hal_high", "sqs_hal_low", "wpt_price_reconstructed"]
  # 理论上依赖 sqs_hal_high/sqs_hal_low 应该能获得 hal_high/hal_low/poc，但没有明确声明
```

### 实际代码行为
```python
# compute_sqs_hal_high 内部会创建 hal_high, poc 列（作为副产品）
result = BaselineFeatureEngineer.add_poc_hal_dimensionless_features(
    result, required_features={"hal_high"}, ...
)

# compute_sr_strength_max 如果发现这些列不存在，会自动计算（自动修复）
if "hal_high" not in result.columns:
    # 自动修复...
```

## 改进方案

### 方案1：在配置中明确声明副产品（推荐）

在配置文件中明确声明这些特征产生的所有输出列：

```yaml
sqs_hal_high:
  output_columns: ["sqs_hal_high", "hal_high", "poc"]  # 明确声明副产品
  
sqs_hal_low:
  output_columns: ["sqs_hal_low", "hal_low", "poc"]  # 明确声明副产品

sr_strength_max:
  dependencies: ["atr", "sqs_hal_high", "sqs_hal_low", "wpt_price_reconstructed"]
  # 依赖解析器会确保 hal_high, hal_low, poc 在 sr_strength_max 之前计算
```

**优点**：
- 依赖关系清晰
- 依赖解析器会自动处理顺序
- 自动修复机制只需作为防御性措施

### 方案2：创建独立的 POC/HAL 特征

创建一个独立的特征来产生这些列：

```yaml
poc_hal_features:
  module: baseline
  compute_func: BaselineFeatureEngineer.add_poc_hal_dimensionless_features
  dependencies: ["atr", "wpt_price_reconstructed"]
  output_columns: ["hal_high", "hal_low", "poc"]
  
sqs_hal_high:
  dependencies: ["poc_hal_features"]  # 依赖独立特征
  output_columns: ["sqs_hal_high"]
  
sqs_hal_low:
  dependencies: ["poc_hal_features"]  # 依赖独立特征
  output_columns: ["sqs_hal_low"]

sr_strength_max:
  dependencies: ["atr", "poc_hal_features"]  # 明确依赖
  output_columns: ["sr_strength_max"]
```

**优点**：
- 职责分离，更清晰
- 其他特征也可以直接依赖 `poc_hal_features`

**缺点**：
- 需要重构代码，将 `add_poc_hal_dimensionless_features` 包装成特征函数

## 推荐实施

**采用方案1**，因为：
1. 改动最小
2. 能够立即解决依赖关系不清晰的问题
3. 自动修复机制可以保留作为防御性措施，但不应是主要机制
4. 测试应该重点验证依赖解析是否正确，而不是自动修复

## 测试策略调整

测试应该重点验证：
1. ✅ **依赖解析顺序正确**：`hal_high`、`hal_low`、`poc` 在 `sr_strength_max` 之前计算
2. ✅ **配置驱动的计算流程**：通过配置文件正确解析依赖并计算
3. ⚠️ **自动修复作为安全网**：只在异常情况下验证（例如直接调用函数而未经过依赖解析）

而不是：
- ❌ 测试"缺少依赖时的自动修复"作为主要场景

