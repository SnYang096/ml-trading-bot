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

## ✅ 主流归一化方法清单（与你给的表对齐）& repo contract 映射

我们在 repo 里不做“训练前 fit 的 scaler”（避免泄露），而是做 **rolling/causal 版本**。

| 你给的方法 | 典型公式 | 我们的 contract method 命名 | 适用（本 repo） |
|---|---|---|---|
| **Z-Score** | \( (x-\mu)/\sigma \) | `zscore_rolling` | 连续型、近似稳定分布：`roc_5`、`volume_anomaly`、（部分）`vpin_*_zscore_*` |
| **Min-Max** | \( (x-x_{min})/(x_{max}-x_{min}) \) | `bounded_0_1` / `bounded_-1_1` / `bounded_0_100` | 有明确物理边界：RSI/随机指标；或我们显式构造的 0..1 语义分数（scene scores） |
| **Robust Scaling** | \( (x-\mathrm{median})/\mathrm{IQR} \) | `robust_rolling` | 金融长尾/有 outlier 的连续值（rolling median/IQR） |
| **Log + Robust/Z** | `log1p(x)` → robust/z | `log1p_robust_rolling` / `log_robust_rolling` | 成交量/订单流/比值类长尾：如 `fp_delta_poc`、`fp_max_imbalance_ratio`、Hilbert 比值类 |
| **Rank Transform** | percentile rank → [0,1] | `rank_rolling` | 跨资产最稳：Hilbert 包络（`replace_env_with_qnorm`），或任何 “只关心相对强弱” 的信号 |
| **Unit Vector (L2)** | \( x/\|x\|_2 \) | `l2_norm` | embedding（你提到的 `dl_sequence_features_f`，我们也支持输出 tanh；L2 可作为备选） |

### 额外：量化常用但不在 scaler 表里的“波动率尺度归一化”

| 方法 | 典型公式 | contract method | 适用场景 |
|---|---|---|---|
| **ATR 尺度化** | \( (level-close)/ATR \) 或 \( x/ATR \) | `atr_distance`（level 类）/ `atr`（强度类） | 价格位置信号最稳的跨资产做法：POC/HAL、Footprint 的 `fp_poc/fp_vah/...` |

---

## 🤖 `dl_sequence_features_f`（Mamba/Transformer 序列特征）是否需要归一化？

### 结论

- **输入序列**：`dl_sequence_features_f` 内部已经做了**严格因果的 EMA z-score 归一化**（只使用过去数据），用于防止信息泄露并提升跨时间稳定性。
- **输出 embedding（例如 64 维）**：
  - **树模型（GBDT）**：通常不需要额外归一化（树对尺度不敏感），建议当作 **Pool B 候选**，不要作为必选特征。
  - **NN（包括 MLP 多头）**：如果把 embedding 当作普通特征输入，**建议**做一个明确的“输出尺度约束/归一化”，否则不同 seed / 不同市场阶段的 embedding 尺度漂移会放大训练不稳定性。

### 推荐做法（输出 embedding 的“可选归一化”）

当 `dl_sequence_features_f` 的输出需要喂给 NN 时，推荐二选一（都必须是因果的）：

1. **逐维 rolling/EMA z-score（推荐）**
   - 对每个 embedding 维度，按时间做 rolling 或 EMA 均值/方差，并输出 z-score
   - 约束到常见范围（例如 99% 分位落在 [-5, 5]）
2. **逐行 L2 normalize（可选）**
   - 对每个时间点的 embedding 向量做 \(x / (\|x\|_2 + \epsilon)\)
   - 好处：天然有界，跨 symbol 更稳定
   - 注意：会改变向量幅度信息（仅保留方向）

### Feature Contract 要求

如果你把 `dl_sequence_features_f` 纳入某个任务/策略的输入：
- 必须在 Feature Contract 中明确它是 **optional block**（高成本、易 drift）
- 必须明确 **missingness policy**（无 GPU / 无依赖 / 计算超时的回退行为）
- 如果做了输出归一化，必须把“输出归一化版本”的列命名策略写清（避免训练/推理不一致）

---

## 🔧 需要修改的特征函数

### 1. ATR 类特征

```python
# 重要结论（已落地到代码）：`atr` 这列必须是 **价格单位 ATR**
# 因为它会被用于：
# - 路径原语 labels：mfe_atr = (price_diff) / atr(t)
# - SR/结构特征的“反归一化”：level_raw = level_norm * atr + close
#
# 如果你需要跨资产可比的波动率，请使用单独列：
# - atr_ratio = atr / close（无量纲）
# - natr_14（Normalized ATR）
# - atr_percentile（波动率分位数/状态）
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

1. [x] `atr_ratio` / `natr_14` / `atr_percentile` → 作为“跨资产可比”的 ATR 类指标（无量纲）
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
# `atr` 仍然叫 "atr"，且语义固定为“价格单位 ATR”（用于尺度/反归一化/labels）
# 若需要 `atr/close`，请使用 `atr_ratio`（或 `natr_14`）
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

### 4. 验证“归一化后仍然有用”的方法（建议顺序）

归一化本质是 **尺度对齐/数值稳定**，不自动保证收益提升。要验证“信息没有被洗没”，建议做两层验证：

**A. 信息/分布层（最快，无需训练）**
- **非退化**：不能变成常量/几乎常量（std、unique count、缺失率）。
- **饱和度**（对 `bounded_*` / `tanh`）：统计落在边界附近的比例（例如 \(|x|\ge 0.99\)、或接近 0/1 的占比），过高说明 clip/scale 可能太强。
- **跨资产可比性**：同一列在多资产上的分布尺度比（std ratio / IQR ratio）应在合理范围（你已有 multi-asset normalization tests 的思路）。

**B. 预测有效性层（需要训练/回测，但可以很轻量）**
- **IC/IR / rank IC**：用 `make ts-feature-eval` / `make ts-factor-eval` 在固定 horizon 下评估（建议同时做 scene slice：near_sr / compression_high / trend_high）。
- **Ablation（最可信）**：同一策略、同一窗口、同一 seed，比较“加入某一组归一化特征前后”的 Sharpe/收益/回撤（可以复用 `make ts-strategy-feature-compare`）。
- **解释稳定性**：对赢家模型导出 gain importance / SHAP，检查归一化后的列是否仍进入 top-k，且跨 seed/时间窗稳定。

**最省算力的投入产出顺序**
1. 先做 A（分布/饱和/非退化）。
2. 再做 IC/IR（比训练便宜很多）。
3. 最后用 multi-seed ablation 定型。

---

## 🔗 相关文档

- `docs/architecture/树模型策略report/BEST_FEATURE_COLUMNS_BY_STRATEGY.md` - 各策略最佳特征列
- `config/feature_dependencies.yaml` - 特征定义
- `docs/architecture/树模型策略report/ATR_SEMANTICS_AND_NORMALIZATION.md` - ATR 的语义统一（为什么 `atr` 必须是价格单位）

---

## 📝 变更记录

### 2026-01-01

**Phase 1 完成**：
- ✅ `compute_atr_ratio_from_series` / `natr_14_f` / `compute_atr_percentile_from_series` → 提供无量纲 ATR 类指标
- ✅ `compute_macd_from_series` → 新增，返回 `macd / ATR`
- ✅ `compute_bb_width_features_from_series` → 只返回 `bb_width_normalized`, `bb_position`
- ✅ `compute_poc_hal_features_from_series` → 返回 `(level - close) / ATR`
- ✅ `compute_sr_strength_max_from_series` → `dist_to_nearest_sr` 归一化为 ATR 倍数

### 2026-01-03

**ATR 语义修正（避免下游计算错误）**：
- ✅ `atr_f` 的 `atr` 统一为 **价格单位 ATR**（用于 labels 与 SR 反归一化）
- ✅ 归一化/跨资产可比的 ATR 形态转由 `atr_ratio` / `natr_14` / `atr_percentile` 承担

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
