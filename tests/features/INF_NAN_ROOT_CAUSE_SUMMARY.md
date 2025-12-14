# Inf/NaN 根因分析总结

## 测试结果总结

### 1. `sr_strength_max` - ✅ 已修复
- **测试结果**: PASSED (0 inf, 0 NaN)
- **状态**: 修复有效，没有 inf 值
- **修复措施**:
  - 添加了 `EPS` 到 `price_trend` 计算的分母
  - 添加了 `EPS` 到 `vol_ratio` 计算的分母
  - 添加了 `replace([np.inf, -np.inf], np.nan)` 到 `boundary_strengths`

### 2. RSI - ⚠️ 需要澄清
- **测试结果**: 发现 14 个 "inf" 值
- **实际原因**: 这些实际上是 **NaN**，不是 inf
  - `talib.RSI` 在数据不足时返回 NaN（前 14 个值）
  - `~np.isfinite()` 会同时匹配 NaN 和 inf
  - 之前的调试输出误将 NaN 当作 inf
- **修复状态**: 
  - ✅ 已添加 `replace([np.inf, -np.inf], np.nan)` 
  - ✅ 已添加输入验证
  - ⚠️ 需要区分 NaN（正常，数据不足）和 inf（异常）

### 3. Hurst 特征 - 🔍 正在调查
- **测试结果**: 发现 52 个 inf 值
- **位置**: 前 52 个数据点（数据不足导致）
- **需要进一步调查**: 
  - 检查 `compute_hurst_dfa` 在数据不足时的行为
  - 检查是否有除零操作
  - 检查 `eps` 参数是否正确使用

### 4. Trade Clustering 测试集 - ✅ 已修复
- **测试结果**: PASSED
- **发现**: 测试集 Trade Clustering 特征**有值**，不是全 NaN
  - 191 个测试样本中，所有基础特征都有值
  - 只有 1 个样本的 `trade_cluster_imbalance_ratio` 是 NaN（正常）
- **结论**: 之前的 "全 NaN" 可能是：
  - 缓存问题
  - 时间对齐问题（已修复）
  - 或者是在训练流程中的其他问题

## 关键发现

### 1. `~np.isfinite()` 的混淆
- `~np.isfinite()` 会同时匹配 **NaN** 和 **inf**
- 需要分别使用 `np.isinf()` 和 `np.isnan()` 来区分
- 已修复 `_debug_inf` 函数，现在会分别显示 inf 和 NaN 的数量

### 2. RSI 的 "inf" 实际上是 NaN
- `talib.RSI` 在数据不足时返回 NaN（正常行为）
- 前 14 个值（period=14）都是 NaN，这是预期的
- 需要区分：
  - **NaN**（正常，数据不足）：应该保留，不删除
  - **inf**（异常，计算错误）：应该替换为 NaN 或删除

### 3. Hurst 特征有真正的 inf 值
- 52 个 inf 值，需要进一步调查
- 可能原因：
  - `compute_hurst_dfa` 中的除零操作
  - `log(0)` 或 `log(负数)` 操作
  - 数据不足时的边界情况处理不当

## 修复建议

### 1. 改进调试输出
- ✅ 已修复：`_debug_inf` 现在会分别显示 inf 和 NaN
- ✅ 已修复：`drop_inf_rows` 现在只删除 inf，保留 NaN

### 2. RSI 处理
- ✅ 已修复：添加了输入验证和 `replace([np.inf, -np.inf], np.nan)`
- ⚠️ 建议：在训练流程中，前 `period` 个 NaN 是正常的，不应该删除

### 3. Hurst 特征
- ⚠️ 需要进一步调查 52 个 inf 值的来源
- 建议：检查 `compute_hurst_dfa` 的边界情况处理

### 4. Trade Clustering
- ✅ 已修复：时间格式问题
- ✅ 已修复：索引检查
- ✅ 测试集特征有值（不是全 NaN）

## 下一步行动

1. **运行完整的 Hurst 测试**，找出 52 个 inf 值的具体原因
2. **验证训练流程**，确认修复后的效果
3. **区分 NaN 和 inf**，在训练流程中正确处理

