# invert_features vs requested_features：关键区别

## 🎯 核心问题

**为什么 `invert_features` 来自 Pool B，但模型测试时又没有使用？**

**答案**：`invert_features` 和 `requested_features` **不是一样的**，它们属于**两个不同的命名空间**！

---

## 一、两种命名空间的区别

### `requested_features`：特征计算节点（Feature Compute Functions）

```yaml
requested_features:
  - macd_f          # 特征节点（计算函数）
  - rsi_f           # 特征节点（计算函数）
  - volume_ratio_f  # 特征节点（计算函数）
```

**作用**：告诉系统"要计算哪些特征"

**命名空间**：特征节点名（通常以 `_f` 结尾）

---

### `invert_features`：输出列名（Output Column Names）

```yaml
invert_features:
  - macd_signal              # 输出列名
  - trade_cluster_net_runs   # 输出列名
  - dtw_bull_flag_dist_w20   # 输出列名
```

**作用**：告诉系统"对哪些列乘以 -1"

**命名空间**：输出列名（不是特征节点名）

---

## 二、具体例子：`macd_f` vs `macd_signal`

### 特征节点 `macd_f` 的定义

```yaml
# config/feature_dependencies.yaml
macd_f:
  output_columns:
    - macd              # 输出列 1
    - macd_signal       # 输出列 2 ← 这个列可以被反向
    - macd_histogram    # 输出列 3
```

### 关系图

```
requested_features: [macd_f]
    ↓
计算特征节点 macd_f
    ↓
生成 3 个输出列：
  - macd
  - macd_signal        ← 这个列可以被反向
  - macd_histogram
    ↓
invert_features: [macd_signal]
    ↓
训练时：macd_signal 列会被乘以 -1
```

---

## 三、你的 compression_breakout 情况

### 实际映射关系

根据 `feature_dependencies.yaml`：

| requested_features (节点) | 输出列 | invert_features 中的列 | 是否被使用 |
|---------------------------|--------|----------------------|----------|
| `compression_duration_f` | `compression_duration` | - | ✅ 被使用 |
| `atr_f` | `atr` | - | ✅ 被使用 |
| `volume_ratio_f` | `volume_ratio` | - | ✅ 被使用 |
| `liquidity_void_f` | `liquidity_void_*` (6列) | - | ✅ 被使用 |
| `trend_r2_20_f` | `trend_r2_20` | - | ✅ 被使用 |
| - | - | `macd_signal` | ❌ **需要 `macd_f` 节点** |
| - | - | `dtw_bull_flag_dist_w20` | ❌ **需要 `dtw_features_f` 节点** |
| - | - | `trade_cluster_net_runs` | ❌ **需要 `order_flow_all_features_f` 节点** |
| - | - | `wpt_price_energy_high_ratio` | ❌ **需要 `wpt_volatility_features_f` 节点** |

### 问题根源

**`invert_features` 中的列对应的特征节点都没有被选中！**

```
Pool B 的 invert_features:
  - dtw_bull_flag_dist_w20  → 需要 dtw_features_f 节点 ❌
  - macd_signal              → 需要 macd_f 节点 ❌
  - trade_cluster_net_runs   → 需要 order_flow_all_features_f 节点 ❌
  - wpt_price_energy_high_ratio → 需要 wpt_volatility_features_f 节点 ❌

但 feature-group-search 只选中了:
  - compression_duration_f ✅
  - atr_f ✅
  - volume_ratio_f ✅
  - liquidity_void_f ✅
  - trend_r2_20_f ✅

→ 所以这些 invert_features 对应的列根本不会被计算出来！
```

---

## 四、为什么写回时保留了这些 `invert_features`？

根据代码逻辑（`feature_group_search.py:116-118`）：

```python
# Therefore we must NOT prune invert_features by requested_features (different namespaces).
# It's safe to keep extra entries: the trainer only applies inversion to columns that
# are actually present in the selected feature columns.
```

**设计原因**：
1. **命名空间不同**：`requested_features` 是节点名，`invert_features` 是列名
2. **不能简单裁剪**：不能用节点名去裁剪列名（会误删）
3. **训练器会忽略**：训练器只会对**实际存在的列**应用反向，不存在的列会被忽略

**所以**：
- ✅ 保留这些 `invert_features` 是**安全的**（不会影响训练）
- ✅ 如果未来把这些特征节点加到 `requested_features`，反向会自动生效
- ⚠️ 但这些列目前**没有被使用**（因为对应的特征节点没有被选中）

---

## 五、总结

### 关键理解

1. **`requested_features` ≠ `invert_features`**
   - `requested_features` = 特征节点（计算什么）
   - `invert_features` = 输出列名（对哪些列取反）

2. **命名空间不同**
   - 不能用 `requested_features` 去裁剪 `invert_features`
   - 它们是两个独立的配置项

3. **反向只在列存在时生效**
   - 如果列不存在（特征节点没被选中），反向会被忽略
   - 不会报错，只是不生效

### 你的情况

- ✅ **反向处理逻辑是正确的**
- ✅ **这些 `invert_features` 没有被使用**（因为对应的特征节点没有被选中）
- ✅ **不需要重新跑**（即使重新跑，结果也一样）
- ✅ **可以清理 `invert_features`**（只保留实际使用的列），或者保留现状（无害）

---

**最后更新**: 2026-01-28
