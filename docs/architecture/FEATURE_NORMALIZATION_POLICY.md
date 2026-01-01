# 特征归一化策略

> **核心原则**：所有特征在计算时就返回归一化值，不保留原始值，使用因果性归一化方法避免未来数据泄露。

---

## 🎯 设计目标

1. **跨资产可比**：BTC 和 ETH 的特征值在同一尺度
2. **NN 友好**：所有特征在 [-3, 3] 或 [0, 1] 范围内
3. **无未来泄露**：只使用当前及历史数据
4. **统一管理**：树模型和 NN 使用相同特征集

---

## ⚠️ 归一化方法对比

### ❌ Preprocessor 归一化（不推荐用于训练）

```python
# 问题：fit 整个训练集会泄露未来数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ❌ 用到了训练集后面的数据
```

**问题**：
- `fit()` 阶段会计算整个训练集的 mean/std
- 第一个样本的 z-score 计算时，用到了最后一个样本的信息
- 回测结果会过于乐观

**例外**：
- 实盘时可以用训练集 fit 的 scaler
- 但需要定期重新 fit（drift 问题）

### ✅ 特征计算时归一化（推荐）

```python
# 因果性归一化：只用过去数据
def compute_atr_normalized(close, high, low, window=14):
    atr = compute_atr(high, low, close, window)
    # 方法 1: 除以 close（跨资产可比）
    return atr / close
    
    # 方法 2: rolling percentile（有界 [0, 1]）
    return atr.rolling(288).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8))
    
    # 方法 3: rolling z-score（适合高斯分布）
    return (atr - atr.rolling(100).mean()) / (atr.rolling(100).std() + 1e-8)
```

---

## 📋 归一化方法选择

| 特征类型 | 推荐方法 | 输出范围 | 示例 |
|----------|----------|----------|------|
| **波动率类** | `/ close` | ~[0, 0.1] | `atr_norm = atr / close` |
| **价格类** | `(price - ref) / atr` | ~[-3, 3] | `poc_norm = (poc - close) / atr` |
| **距离类** | `/ atr` 或 `/ close` | ~[-3, 3] | `dist_norm = dist_to_sr / atr` |
| **成交量类** | `/ rolling_mean` | ~[0, 3] | `vol_norm = volume / volume.rolling(20).mean()` |
| **指标类** | 已归一化 | [0, 100] | `rsi` (直接使用) |
| **DTW 距离** | `exp(-dist/scale)` | (0, 1] | `dtw_score = exp(-dtw_dist / 0.5)` |
| **SR 结构** | `/ close` | ~[0.9, 1.1] | `poc_ratio = poc / close` |
| **比率类** | 已归一化 | [0, 1] | `wick_ratio` (直接使用) |

---

## 🔧 需要修改的特征函数

### 1. ATR 类特征

```python
# 当前（未归一化）
def compute_atr(...) -> pd.Series:
    return atr  # ❌ 原始值

# 修改后
def compute_atr(...) -> pd.Series:
    atr_norm = atr / close  # ✅ 归一化
    return atr_norm.rename("atr")  # 返回归一化值，但保持原名
```

### 2. SR 结构特征

```python
# 当前（未归一化）
def compute_poc_hal_features(...):
    return pd.DataFrame({
        "poc": poc,           # ❌ 原始价格
        "hal_high": hal_high, # ❌ 原始价格
        "hal_low": hal_low,   # ❌ 原始价格
    })

# 修改后
def compute_poc_hal_features(...):
    atr = compute_atr(...)  # 或从参数传入
    return pd.DataFrame({
        "poc": (poc - close) / (atr + 1e-8),      # ✅ 归一化偏离
        "hal_high": (hal_high - close) / (atr + 1e-8),
        "hal_low": (hal_low - close) / (atr + 1e-8),
    })
```

### 3. DTW 特征

```python
# 当前（未归一化）
def extract_dtw_features(...):
    result[f"dtw_{template}_dist"] = distance  # ❌ 原始距离

# 修改后
def extract_dtw_features(..., dist_scale=0.5):
    # 距离转分数：距离越小，分数越高
    score = np.exp(-distance / dist_scale)  # ✅ 归一化到 (0, 1]
    result[f"dtw_{template}"] = score  # 改名去掉 _dist
```

### 4. MACD/BBands 等

```python
# 当前（未归一化）
def compute_macd(...):
    return pd.DataFrame({
        "macd": macd,          # ❌ 原始值
        "macd_signal": signal, # ❌ 原始值
        "macd_histogram": hist # ❌ 原始值
    })

# 修改后
def compute_macd(..., atr: pd.Series):
    return pd.DataFrame({
        "macd": macd / (atr + 1e-8),           # ✅ 归一化
        "macd_signal": signal / (atr + 1e-8), # ✅ 归一化
        "macd_histogram": hist / (atr + 1e-8) # ✅ 归一化
    })
```

---

## 📊 当前状态

| 指标 | 数值 |
|------|------|
| 已归一化特征列 | 278 个 (25.2%) → Phase 1 完成后更多 |
| 未归一化特征列 | 823 个 (74.8%) → 逐步减少 |

### 未归一化特征分类

| 类别 | 数量 | 优先级 | 状态 |
|------|------|--------|------|
| DTW | 257 | P1 | Phase 2 |
| Other (ATR, MACD 等) | 419 | P1 | ✅ Phase 1 完成 |
| Volume | 82 | P2 | Phase 3 |
| Price (WPT 重构等) | 37 | P2 | Phase 3 |
| SR Structure | 15 | P2 | ✅ Phase 1 完成 |
| CVD | 13 | P3 | 已归一化 |

---

## 🚀 实施计划

### ✅ Phase 1: 核心特征归一化（已完成 2026-01-01）

1. [x] `atr_f` → 输出 `atr / close` (~[0.001, 0.1])
2. [x] `macd_f` → 输出 `macd / atr` (~[-3, 3])
3. [x] `bb_width_f` → 输出 `bb_width_normalized`, `bb_position` (移除原始价格)
4. [x] `poc_hal_features_*` → 输出 `(level - close) / atr` (~[-3, 3])
5. [x] `sr_strength_max_*` → `dist_to_nearest_sr` 归一化为 ATR 倍数

### Phase 2: DTW 特征语义化（中优先）

1. [ ] `dtw_features_*` → 输出 `exp(-dist/scale)` 转为分数
2. [ ] 考虑直接用 `dtw_scene_semantic_scores_f`（已语义化）

### Phase 3: 其他特征（低优先）

1. [ ] Volume 类特征
2. [ ] WPT 重构价格类特征

---

## ⚠️ 注意事项

### 1. 保持列名不变

```python
# 虽然值变了，但列名保持不变，避免破坏下游依赖
"atr" → 仍然叫 "atr"，但值是 atr/close
```

### 2. 向后兼容问题

修改后需要：
- 删除旧的 FeatureStore 缓存
- 重新运行 feature-group-search
- 更新测试中的断言范围

### 3. 验证归一化效果

```python
# 归一化后的特征应该满足：
assert feature.abs().quantile(0.99) < 5, "99% 分位数应该 < 5"
assert feature.std() > 0.01, "标准差应该 > 0.01（避免常量）"
```

---

## 🔗 相关文档

- `docs/strategies/BEST_FEATURE_COLUMNS_BY_STRATEGY.md` - 各策略最佳特征列
- `config/feature_dependencies.yaml` - 特征定义

---

## 📝 变更记录

### 2026-01-01

**Phase 1 完成**：
- ✅ `compute_atr_from_series` → 返回 `atr / close`
- ✅ `compute_macd_from_series` → 新增，返回 `macd / ATR`
- ✅ `compute_bb_width_features_from_series` → 只返回 `bb_width_normalized`, `bb_position`
- ✅ `compute_poc_hal_features_from_series` → 返回 `(level - close) / ATR`
- ✅ `compute_sr_strength_max_from_series` → `dist_to_nearest_sr` 归一化为 ATR 倍数

**Phase 2 完成**：
- ✅ `compute_sma_position_from_series` → 新增，返回 `(close - sma_200) / close`
  - 正值=多头趋势，负值=空头趋势
  - 范围 [-1, 1]，可跨资产比较
- ✅ `compute_volume_ratio_from_series` → 新增，返回 `volume / rolling_mean_volume`
  - 1.0=正常，>1.0=放量，<1.0=缩量
  - 范围 [0, 10]，可跨资产比较
- ✅ `extract_dtw_features` → DTW 距离转换为相似度分数
  - 使用 `exp(-dist/scale)` 转换
  - 范围 [0, 1]，1.0=完美匹配，0.0=不匹配

**测试**：
- ✅ `tests/features/test_phase1_normalization.py` - 8 个测试全部通过
- ✅ `tests/features/test_sr_structure_features.py` - 7 个测试全部通过
- ✅ `tests/features/test_baseline_remaining_narrow.py` - 2 个测试全部通过

**配置更新**：
- ✅ `config/feature_dependencies.yaml` 中的相关特征定义已更新
- ✅ 新增 `sma_200_position_f` 和 `volume_ratio_f` 特征节点

---

*更新时间: 2026-01-01*
