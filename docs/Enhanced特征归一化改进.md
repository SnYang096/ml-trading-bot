# Enhanced 特征归一化改进

## 📅 改进日期
2024-11-15

## 🎯 改进目标
为 Enhanced 特征工程实现**分组归一化策略**，针对不同特征类型使用最适合的归一化方法，提升多资产训练效果。

---

## 🔍 问题分析

### 改进前的问题

**统一归一化策略**：
- 所有 Enhanced 特征使用同一个 `StandardScaler`
- 没有考虑不同特征类型的特性
- 对于有异常值的特征（如订单流、频谱特征）可能不够鲁棒

### 特征类型分析

Enhanced 特征包含多种类型：

1. **Hurst 特征** (`*_hurst`)
   - 范围：通常在 [0, 1] 之间
   - 特性：已归一化，但可能有 NaN

2. **WPT 能量/比率特征** (`wpt_*_energy`, `wpt_*_ratio`, `wpt_*_entropy`)
   - 范围：通常在 [0, 1] 之间
   - 特性：能量和比率已归一化

3. **WPT 统计特征** (`wpt_*_mean`, `wpt_*_std`)
   - 范围：无界
   - 特性：需要标准化

4. **频谱特征** (`spectral_*`, `fft_*`, `psd_*`, `frequency_*`)
   - 范围：无界，可能有异常值
   - 特性：对异常值敏感

5. **Hilbert 特征** (`hilbert_*`, `phase_*`)
   - 范围：相位在 [-π, π]，频率无界
   - 特性：需要标准化

6. **订单流特征** (`cvd_*`, `ofi_*`, `order_flow_*`, `taker_buy_*`)
   - 范围：无界，可能有极端值
   - 特性：对异常值敏感

---

## ✅ 改进方案

### 分组归一化策略

参考 `TalibFeatureEngineer` 的实现，为不同特征类型选择最适合的归一化方法：

| 特征组 | 归一化方法 | 原因 |
|--------|-----------|------|
| **Hurst 特征** | `MinMaxScaler` | 已在 [0,1] 范围内，保持范围 |
| **WPT 能量/比率** | `MinMaxScaler` | 能量/比率在 [0,1]，保持范围 |
| **WPT 统计** | `StandardScaler` | mean/std 无界，需要标准化 |
| **频谱特征** | `RobustScaler` | 可能有异常值，需要鲁棒处理 |
| **Hilbert 特征** | `StandardScaler` | 相位/频率需要标准化 |
| **订单流特征** | `RobustScaler` | 可能有极端值，需要鲁棒处理 |
| **其他特征** | `StandardScaler` | 默认策略 |

---

## 🔧 实现细节

### 1. 特征分组逻辑

```python
# 根据特征名称模式自动分组
hurst_features = [col for col in feature_columns if "hurst" in col.lower()]
wpt_energy_features = [col for col in feature_columns 
                       if "wpt" in col.lower() and 
                       ("energy" in col.lower() or "ratio" in col.lower())]
spectral_features = [col for col in feature_columns 
                     if any(x in col.lower() for x in ["spectral", "fft", "psd"])]
order_flow_features = [col for col in feature_columns 
                       if any(x in col.lower() for x in ["cvd", "ofi", "order_flow"])]
```

### 2. 分组归一化

```python
feature_groups = {
    "hurst": (hurst_features, MinMaxScaler()),
    "wpt_energy": (wpt_energy_features, MinMaxScaler()),
    "wpt_stat": (wpt_stat_features, StandardScaler()),
    "spectral": (spectral_features, RobustScaler()),
    "hilbert": (hilbert_features, StandardScaler()),
    "order_flow": (order_flow_features, RobustScaler()),
    "other": (other_features, StandardScaler()),
}

# 对每个特征组分别归一化
for group_name, (group_features, scaler_class) in feature_groups.items():
    scaler = scaler_class()
    X_scaled = scaler.fit_transform(X)
    self.group_scalers[f"{timeframe}_{group_name}"] = scaler
```

### 3. 特殊处理

```python
# Hurst 特征：确保在 [0, 1] 范围内
if group_name == "hurst":
    X = np.clip(X, 0.0, 1.0)
```

### 4. 向后兼容

```python
# 保持统一的 scaler（用于旧代码）
if fit:
    unified_scaler = self.scaler_class()
    unified_scaler.fit(all_features.values)
    self.scalers[timeframe] = unified_scaler
```

---

## 📊 优势对比

### 改进前（统一 StandardScaler）

| 问题 | 影响 |
|------|------|
| 异常值敏感 | 订单流/频谱特征的异常值会影响所有特征 |
| 范围不匹配 | Hurst/WPT 能量已在 [0,1]，再次标准化可能损失信息 |
| 多资产训练 | 不同资产的特征分布差异可能被放大 |

### 改进后（分组归一化）

| 优势 | 效果 |
|------|------|
| **鲁棒性** | 订单流/频谱特征使用 RobustScaler，对异常值不敏感 |
| **精确性** | Hurst/WPT 能量使用 MinMaxScaler，保持 [0,1] 范围 |
| **多资产兼容** | 不同特征类型使用最适合的方法，提升跨资产泛化能力 |

---

## 🔄 兼容性

### 保存/加载 Scaler

```python
# 保存时包含分组 scaler
scaler_data = {
    "scalers": self.scalers,           # 统一 scaler（向后兼容）
    "group_scalers": self.group_scalers,  # 分组 scaler（新功能）
    "scaler_type": self.scaler_type,
}

# 加载时兼容旧格式
if isinstance(scaler_data, dict) and "scalers" in scaler_data:
    self.scalers = scaler_data.get("scalers", {})
    self.group_scalers = scaler_data.get("group_scalers", {})
else:
    # 旧格式：直接是 scalers 字典
    self.scalers = scaler_data
    self.group_scalers = {}
```

---

## 📈 预期效果

### 多资产训练

1. **更好的跨资产泛化**
   - 不同资产的特征在相同尺度
   - 模型可以学习通用的市场模式

2. **更稳定的训练**
   - 异常值不会影响所有特征
   - 特征重要性更准确

3. **更好的特征表达**
   - Hurst/WPT 能量保持在 [0,1] 范围
   - 订单流/频谱特征对异常值鲁棒

---

## 🧪 验证方法

### 检查特征分布

```python
# 检查不同资产的特征分布是否相似
for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    symbol_X = X_df[train_df['symbol'] == symbol]
    
    # Hurst 特征应该在 [0, 1]
    hurst_cols = [c for c in symbol_X.columns if 'hurst' in c.lower()]
    if hurst_cols:
        print(f"{symbol} Hurst range: [{symbol_X[hurst_cols].min().min():.3f}, {symbol_X[hurst_cols].max().max():.3f}]")
    
    # 订单流特征应该没有极端值
    of_cols = [c for c in symbol_X.columns if any(x in c.lower() for x in ['cvd', 'ofi'])]
    if of_cols:
        print(f"{symbol} Order flow std: {symbol_X[of_cols].std().mean():.3f}")
```

---

## 📝 总结

### ✅ 改进内容

1. **分组归一化**：不同特征类型使用最适合的归一化方法
2. **鲁棒性提升**：订单流/频谱特征使用 RobustScaler
3. **精确性提升**：Hurst/WPT 能量使用 MinMaxScaler 保持范围
4. **向后兼容**：保持统一的 scaler 用于旧代码

### 🎯 适用场景

- ✅ 多资产训练（主要目标）
- ✅ 单资产训练（也有提升）
- ✅ 包含异常值的数据
- ✅ 需要精确特征范围的应用

### 🔗 相关文档

- [特征归一化检查报告](./特征归一化检查报告.md)
- [LightGBM 多资产归一化说明](./LightGBM多资产归一化说明.md)

