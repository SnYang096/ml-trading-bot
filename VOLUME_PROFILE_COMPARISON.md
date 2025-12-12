# Volume Profile 特征对比分析

## 📊 一、当前状态

### ✅ **两个都有，但都使用了 WPT 降噪**

1. **传统的 POC/HAL**（`baseline_features.py`）
2. **WPT 降噪的 VPVR**（`utils_liquidity_features.py`）

**关键发现**：两者都使用**同一个 WPT 降噪函数** `compute_wpt_volume_profile()`，避免重复计算。

---

## 🔍 二、详细对比

### 1. **POC/HAL（传统 Volume Profile）**

**实现位置**：`src/features/time_series/baseline_features.py`

**核心函数**：
- `compute_poc()` - 计算 POC 和 HAL
- `add_poc_hal_dimensionless_features()` - 添加无量纲特征

**WPT 降噪**：
- ✅ **已使用**：通过 `compute_wpt_volume_profile()` 进行降噪
- 可选使用 `wpt_price_reconstructed` 作为价格输入

**特征列表**：
```python
# POC 相关
- poc: POC 价格（最高成交量对应的价格）
- poc_volume_ratio: POC 位置的成交量占比
- price_to_poc_pct: 当前价格到 POC 的相对距离
- poc_position_ratio: POC 在价格区间中的位置（0-1）

# HAL 相关（Value Area 70%）
- hal_high: HAL 高点（Value Area 上界）
- hal_low: HAL 低点（Value Area 下界）
- hal_mid: HAL 中点
- price_to_hal_high_pct: 当前价格到 HAL 高点的相对距离
- price_to_hal_low_pct: 当前价格到 HAL 低点的相对距离
- price_to_hal_mid_pct: 当前价格到 HAL 中点的相对距离
- hal_bandwidth_pct: HAL 带宽（相对）
```

**关注点**：
- **POC**：最高成交量价格（Point of Control）
- **HAL**：70% 成交量区间（Value Area）

---

### 2. **WPT 降噪的 VPVR（Volume Profile Visible Range）**

**实现位置**：`src/features/time_series/utils_liquidity_features.py`

**核心函数**：
- `build_wpt_denoised_vpvr()` - 构建 WPT 降噪的 VPVR
- `compute_wpt_vpvr()` - 包装函数

**WPT 降噪**：
- ✅ **已使用**：通过 `compute_wpt_volume_profile()` 进行降噪
- 使用典型价格 `(H+L+C)/3` 作为输入

**特征列表**：
```python
# VPVR 相关
- vpvr_pvp: Point of Control（最高成交量价格，类似 POC）
- vpvr_hvn_count: High Volume Node 数量（高成交量节点）
- vpvr_lvn_count: Low Volume Node 数量（低成交量节点）
- vpvr_lvn_distance: 当前价格到最近 LVN 的距离
- vpvr_volume_density: 当前价格的成交量密度
- vpvr_price_in_lvn: 当前价格是否在 LVN 中（1.0/0.0）
```

**关注点**：
- **PVP**：最高成交量价格（Point of Control，类似 POC）
- **HVN**：高成交量节点（High Volume Node）
- **LVN**：低成交量节点（Low Volume Node）

---

## 🔄 三、相同点

### ✅ 1. **都使用 WPT 降噪**

两者都使用**同一个函数** `compute_wpt_volume_profile()` 进行降噪：

```python
# 共享函数
from src.features.time_series.utils_volume_profile import compute_wpt_volume_profile

# POC/HAL 使用
vp_result = compute_wpt_volume_profile(
    price_window=price_window,
    volume_window=volume_window,
    bins=bins,
)

# VPVR 使用
vp_result = compute_wpt_volume_profile(
    price_window=price_window,
    volume_window=volume_window,
    bins=bins,
    wavelet=wavelet,
    level=level,
    drop_high_freq=drop_high_freq,
)
```

### ✅ 2. **都计算 POC/PVP**

- **POC/HAL**：计算 `poc`（Point of Control）
- **VPVR**：计算 `vpvr_pvp`（Point of Control，命名不同但含义相同）

### ✅ 3. **都基于 Volume Profile 直方图**

两者都基于成交量分布直方图进行分析。

---

## 🔀 四、不同点

| 维度 | POC/HAL | VPVR |
|------|---------|------|
| **关注点** | POC（最高成交量价格）+ HAL（70% 成交量区间） | PVP（最高成交量价格）+ HVN/LVN（高/低成交量节点） |
| **特征类型** | 价格位置、相对距离、区间带宽 | 节点数量、节点距离、成交量密度 |
| **HAL/Value Area** | ✅ 有（70% 成交量区间） | ❌ 无 |
| **HVN/LVN** | ❌ 无 | ✅ 有（高/低成交量节点） |
| **价格输入** | 可选 `wpt_price_reconstructed` 或 `(high+low)/2` | 典型价格 `(H+L+C)/3` |
| **窗口大小** | 默认 160 | 默认 100 |
| **特征数量** | 约 10 个 | 约 6 个 |

---

## 📝 五、代码位置

### POC/HAL
- **实现**：`src/features/time_series/baseline_features.py`
  - `compute_poc()` (第 239 行)
  - `add_poc_hal_dimensionless_features()` (第 2305 行)
- **共享函数**：`src/features/time_series/utils_volume_profile.py::compute_wpt_volume_profile()`
- **配置**：`config/feature_dependencies.yaml` 中的 `poc_hal_features`

### VPVR
- **实现**：`src/features/time_series/utils_liquidity_features.py`
  - `build_wpt_denoised_vpvr()` (第 30 行)
  - `compute_wpt_vpvr()` (包装函数)
- **共享函数**：`src/features/time_series/utils_volume_profile.py::compute_wpt_volume_profile()`
- **配置**：`config/feature_dependencies.yaml` 中的 `wpt_vpvr`

---

## ⚠️ 六、当前使用状态

### 配置文件检查

**`config/strategies/sr_reversal_long/features.yaml`**：
```yaml
# VPVR 特征（空间域：流动性聚集区）— 临时移除以排查 inf
# - wpt_vpvr  # ← 被注释掉了
```

**结论**：
- ✅ **POC/HAL**：正在使用
- ❌ **VPVR**：当前被注释掉（临时移除以排查 inf 问题）

---

## 🎯 七、建议

### 1. **功能互补性**

两者**功能互补**，建议都保留：

- **POC/HAL**：
  - 适合识别**价值区间**（70% 成交量区间）
  - 适合判断价格是否偏离价值中枢
  - 适合识别支撑/阻力区间

- **VPVR**：
  - 适合识别**流动性节点**（HVN/LVN）
  - 适合判断价格是否在低流动性区域
  - 适合识别流动性真空区

### 2. **使用建议**

- **如果只需要 POC**：使用 POC/HAL（更成熟，特征更丰富）
- **如果需要 HVN/LVN**：使用 VPVR（专门针对节点识别）
- **如果需要两者**：可以同时使用（功能互补）

### 3. **性能考虑**

两者都使用 `compute_wpt_volume_profile()`，如果同时使用：
- ✅ **不会重复计算** WPT 降噪（共享函数）
- ⚠️ **会重复计算** Volume Profile 直方图（但窗口大小不同：160 vs 100）

---

## 📊 八、总结

| 问题 | 答案 |
|------|------|
| **两个都有吗？** | ✅ 是的，两个都有 |
| **都使用 WPT 降噪吗？** | ✅ 是的，都使用（共享函数） |
| **功能重复吗？** | ❌ 不重复，功能互补 |
| **当前都在使用吗？** | ❌ VPVR 被注释掉了（临时移除） |
| **建议保留哪个？** | ✅ 建议都保留（功能互补） |

**最终建议**：
1. ✅ **保留 POC/HAL**：用于价值区间分析
2. ✅ **恢复 VPVR**：用于流动性节点分析（解决 inf 问题后）
3. ✅ **两者可以同时使用**：功能互补，不冲突

