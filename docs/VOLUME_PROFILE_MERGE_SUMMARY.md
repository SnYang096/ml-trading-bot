# Volume Profile 特征合并总结

## ✅ 合并完成

已将 **POC/HAL** 和 **VPVR** 两个 Volume Profile 实现合并为统一的实现。

---

## 📁 新文件

### `src/features/time_series/utils_volume_profile_unified.py`

**核心函数**：
1. `compute_unified_volume_profile_features()` - 计算基础特征
2. `compute_unified_volume_profile_derived_features()` - 计算衍生特征

**输出特征**（统一使用 `vp_` 前缀）：

#### POC/HAL 特征
- `vp_poc`: Point of Control（最高成交量价格）
- `vp_poc_volume_ratio`: POC 位置的成交量占比
- `vp_hal_high`: HAL 高点（Value Area 上界）
- `vp_hal_low`: HAL 低点（Value Area 下界）
- `vp_hal_mid`: HAL 中点

#### HVN/LVN 特征
- `vp_hvn_count`: High Volume Node 数量
- `vp_lvn_count`: Low Volume Node 数量
- `vp_lvn_distance`: 当前价格到最近 LVN 的距离（归一化）
- `vp_volume_density`: 当前价格的成交量密度
- `vp_price_in_lvn`: 当前价格是否在 LVN 中（1.0/0.0）

#### 衍生特征
- `vp_price_to_poc_pct`: 当前价格到 POC 的相对距离
- `vp_poc_position_ratio`: POC 在价格区间中的位置（0-1）
- `vp_price_to_hal_high_pct`: 当前价格到 HAL 高点的相对距离
- `vp_price_to_hal_low_pct`: 当前价格到 HAL 低点的相对距离
- `vp_price_to_hal_mid_pct`: 当前价格到 HAL 中点的相对距离
- `vp_hal_bandwidth_pct`: HAL 带宽（相对）

---

## 🔄 包装函数更新

### `src/features/loader/feature_wrappers.py`

**新增函数**：
- `compute_unified_volume_profile()` - 统一的 Volume Profile 计算函数

**更新函数**：
- `compute_wpt_vpvr()` - 现在使用统一实现（向后兼容）

---

## 📊 优势

### 1. **避免重复计算**
- ✅ 一次计算同时输出 POC/HAL 和 HVN/LVN 特征
- ✅ 共享 `compute_wpt_volume_profile()` 函数，避免重复 WPT 降噪

### 2. **统一特征命名**
- ✅ 所有特征使用 `vp_` 前缀，便于识别和管理
- ✅ 清晰的命名规范（`vp_poc`, `vp_hal_*`, `vp_hvn_*`, `vp_lvn_*`）

### 3. **向后兼容**
- ✅ `compute_wpt_vpvr()` 仍然可用，自动使用统一实现
- ✅ 旧的配置可以继续使用

### 4. **灵活配置**
- ✅ 支持多种价格输入（原始价格、典型价格、WPT 重构价格）
- ✅ 可配置窗口大小、bins 数量、Value Area 比例等

---

## 🔧 使用方法

### 方法 1：使用统一函数（推荐）

```python
from src.features.loader.feature_wrappers import compute_unified_volume_profile

df = compute_unified_volume_profile(
    df,
    window=160,
    bins="auto",
    value_area_ratio=0.7,
    use_wpt_price=True,  # 如果存在 wpt_price_reconstructed，使用它
)
```

### 方法 2：使用向后兼容函数

```python
from src.features.loader.feature_wrappers import compute_wpt_vpvr

df = compute_wpt_vpvr(
    df,
    vpvr_window=100,  # 现在作为 window 参数
    use_typical_price=True,  # VPVR 使用典型价格
)
```

---

## 📝 配置更新建议

### 更新 `config/feature_dependencies.yaml`

```yaml
unified_volume_profile:
  compute_func: compute_unified_volume_profile
  dependencies: []
  output_columns:
    # POC/HAL
    - vp_poc
    - vp_poc_volume_ratio
    - vp_hal_high
    - vp_hal_low
    - vp_hal_mid
    # HVN/LVN
    - vp_hvn_count
    - vp_lvn_count
    - vp_lvn_distance
    - vp_volume_density
    - vp_price_in_lvn
    # 衍生特征
    - vp_price_to_poc_pct
    - vp_poc_position_ratio
    - vp_price_to_hal_high_pct
    - vp_price_to_hal_low_pct
    - vp_price_to_hal_mid_pct
    - vp_hal_bandwidth_pct
  description: "统一的 Volume Profile 特征（合并 POC/HAL 和 VPVR）"
  default_params:
    window: 160
    bins: "auto"
    value_area_ratio: 0.7
    use_wpt_price: true
```

### 更新 `config/strategies/*/features.yaml`

```yaml
features:
  - unified_volume_profile  # 替换原来的 poc_hal 和 wpt_vpvr
```

---

## ⚠️ 迁移注意事项

### 1. **特征名称变更**

| 旧名称 | 新名称 |
|--------|--------|
| `poc` | `vp_poc` |
| `poc_volume_ratio` | `vp_poc_volume_ratio` |
| `hal_high` | `vp_hal_high` |
| `hal_low` | `vp_hal_low` |
| `hal_mid` | `vp_hal_mid` |
| `vpvr_pvp` | `vp_poc`（与 POC 相同） |
| `vpvr_hvn_count` | `vp_hvn_count` |
| `vpvr_lvn_count` | `vp_lvn_count` |
| `vpvr_lvn_distance` | `vp_lvn_distance` |
| `vpvr_volume_density` | `vp_volume_density` |
| `vpvr_price_in_lvn` | `vp_price_in_lvn` |

### 2. **向后兼容**

- ✅ `compute_wpt_vpvr()` 仍然可用
- ✅ 旧的配置可以继续使用（但会输出新的特征名称）

### 3. **性能提升**

- ✅ 一次计算输出所有特征，避免重复计算
- ✅ 共享 WPT 降噪计算，减少计算开销

---

## 🎯 下一步

1. ✅ **测试统一实现**：运行测试确保功能正常
2. ⏳ **更新配置文件**：更新 `feature_dependencies.yaml` 和策略配置
3. ⏳ **更新特征选择**：在模型训练中使用新的特征名称
4. ⏳ **文档更新**：更新相关文档说明新的特征命名

---

## 📊 总结

✅ **合并完成**：POC/HAL 和 VPVR 已合并为统一实现
✅ **向后兼容**：旧的函数仍然可用
✅ **性能优化**：避免重复计算，提高效率
✅ **统一命名**：所有特征使用 `vp_` 前缀

**建议**：使用 `compute_unified_volume_profile()` 作为新的标准实现。

