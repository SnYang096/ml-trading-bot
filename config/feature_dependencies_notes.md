# 特征依赖配置说明

## WPT 噪音过滤

### 重要说明

某些特征（特别是 POC 和 HAL 相关特征）在计算前需要先使用 **WPT（小波包变换）过滤高频噪音**，以提高 SR（支撑/阻力）识别的准确性。

### 需要 WPT 预处理的特征

1. **POC (Point of Control) 特征**
   - `poc_value`: POC 价格值
   - `price_to_poc_pct`: 价格到 POC 的百分比距离
   - 依赖: `wpt_price_fluctuation`

2. **HAL (High Activity Level) 特征**
   - `hal_high_value`: HAL 高点值
   - `hal_low_value`: HAL 低点值
   - `price_to_hal_high_pct`: 价格到 HAL 高点的百分比距离
   - `price_to_hal_low_pct`: 价格到 HAL 低点的百分比距离
   - 依赖: `wpt_price_fluctuation`

3. **SQS (Structure Quality Score) 特征**
   - `sqs_hal_high`: HAL 高点的结构质量评分
   - `sqs_hal_low`: HAL 低点的结构质量评分
   - 依赖: `atr`, `wpt_price_fluctuation`

### 为什么需要 WPT？

1. **高频噪音干扰**: 原始价格数据包含大量高频噪音，直接计算 POC/HAL 会导致 SR 识别不准确
2. **提高稳定性**: WPT 可以分离趋势和波动，使用去趋势后的信号计算 SR 更稳定
3. **减少假信号**: 过滤噪音后，SR 测试和反应更可靠

### 依赖关系

```
wpt_price_fluctuation (WPT 过滤)
    ↓
poc_hal_features (POC/HAL 特征)
    ↓
sqs_hal_high / sqs_hal_low (SQS 评分)
    ↓
sr_strength_max (SR 强度)
```

### 配置示例

在 `feature_dependencies.yaml` 中，相关特征已标注依赖关系：

```yaml
wpt_price_fluctuation:
  dependencies: []
  notes: "WPT 过滤高频噪音，提高 POC/HAL 等 SR 特征的识别准确性"

sqs_hal_high:
  dependencies: ["atr", "wpt_price_fluctuation"]
  notes: "POC/HAL 特征计算前应使用 WPT 过滤高频噪音，提高 SR 识别准确性"
```

### 使用建议

1. **确保 WPT 先计算**: 在计算 POC/HAL 相关特征前，确保 `wpt_price_fluctuation` 已计算
2. **检查依赖顺序**: 特征加载器会自动处理依赖关系，但需要确保配置正确
3. **验证结果**: 使用 WPT 过滤后的特征应该更稳定，减少假信号

