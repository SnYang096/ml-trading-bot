# `required` 字段说明

## 问题

用户问：既然配置在 YAML 中，就表示想要使用这些特征，为什么还要设置 `required: false`？

## 答案

`required` 字段的作用是**控制警告级别**，而不是控制"是否使用"。

### 实际行为

无论 `required` 是 `true` 还是 `false`，代码都会：

1. **尝试使用这些特征**：如果特征存在，就会被选中并用于训练
2. **尝试计算缺失的特征**：如果特征不存在，会尝试通过 `feature_loader` 计算
3. **区别在于警告**：
   - `required: true` → 如果特征缺失，会打印 **警告信息**
   - `required: false` → 如果特征缺失，**不打印警告**（静默忽略）

### 代码逻辑

```python
# src/time_series_model/pipeline/training/volatility_model_config.py

for group in feature_groups:
    feature_name = group.get("feature_name")
    required = bool(group.get("required", False))  # 读取 required 配置
    columns = group.get("columns", [])
    
    # 1. 检查缺失的特征
    missing_cols = [col for col in columns if col not in X_processed.columns]
    
    # 2. 如果缺失且有 feature_name，尝试计算
    if missing_cols and feature_name:
        _compute_feature(feature_name)  # 尝试计算
        missing_cols = [col for col in columns if col not in X_processed.columns]
    
    # 3. 如果仍然缺失且 required=true，打印警告
    if missing_cols and required:
        print(f"   ⚠️ Required feature group '{group.get('name')}' missing columns: {missing_cols}")
    
    # 4. 无论 required 是什么，都会使用存在的特征
    existing_cols = [col for col in columns if col in X_processed.columns]
    if existing_cols:
        selected_columns.extend(existing_cols)  # 使用存在的特征
```

### 关键点

1. **特征使用不受 `required` 影响**：只要特征存在，就会被使用
2. **`required` 只影响警告**：`true` 会打印警告，`false` 不打印
3. **特征计算不受 `required` 影响**：无论 `required` 是什么，都会尝试计算缺失的特征

## 建议

### 对于核心特征（如 GARCH、扩展波动率）

```yaml
- name: garch
  required: true  # 如果缺失，要警告
```

**原因**：这些是核心特征，如果缺失可能影响模型性能，应该知道。

### 对于可选特征（如 Volume Profile、WPT）

```yaml
- name: volume_profile_volatility
  required: true  # 虽然可选，但如果配置了就想用，所以设为 true
```

**原因**：
- 既然配置在 YAML 中，就表示想要使用
- 如果计算失败或缺失，应该知道（打印警告）
- 这样可以帮助调试特征计算问题

### 对于实验性特征

```yaml
- name: experimental_feature
  required: false  # 实验性特征，缺失也不影响
```

**原因**：这些特征还在测试中，缺失不影响核心功能。

## 总结

- **`required: true`** = "我想要这些特征，如果缺失要告诉我"
- **`required: false`** = "这些特征可有可无，缺失也不用告诉我"

**对于 `volume_profile_volatility`**：
- 既然配置在 YAML 中，就表示想要使用
- 应该设置为 `required: true`，这样如果计算失败，会收到警告
- 有助于及时发现特征计算问题

## 当前配置

```yaml
- name: volume_profile_volatility
  feature_name: volume_profile_volatility_features
  required: true  # ✅ 已改为 true
  columns:
    - vp_width_ratio
    - vp_poc_deviation
    - vp_skewness
    - vp_entropy
    - vp_lv_ratio
    - vp_hv_ratio
```

这样配置后：
- 如果特征计算成功 → 正常使用
- 如果特征计算失败 → 打印警告，帮助调试
- 如果特征缺失 → 打印警告，提醒检查

