# WPT 特征实现分析

## 当前状态

### 1. `utils_wpt_features.py`（独立工具模块）
**位置**：`src/features/time_series/utils_wpt_features.py`

**功能**：
- `wpt_decompose()`: 基础 WPT 分解函数
- `extract_wpt_features()`: 从 DataFrame 提取 WPT 特征
- `wpt_reconstruct_subband()`: 重构指定子带

**特点**：
- ✅ 独立函数，可以被按需调用
- ✅ 对整个序列做 WPT（一次性处理）
- ✅ 输出特征：`wpt_price_trend`, `wpt_price_fluctuation`, `wpt_price_reconstructed`, `wpt_price_energy_*_ratio` 等

**使用场景**：
- 被策略特征文件直接调用（sr_reversal, sr_breakout, compression_breakout, trend_following）
- 被 `feature_function_mapping.py` 使用（用于特征加载器）
- 按需加载场景

---

### 2. `enhanced_features.py` 中的 WPT（集成在类中）
**位置**：`src/features/time_series/enhanced_features.py`

**功能**：
- `calculate_wavelet_packet_features()`: 计算单个窗口的 WPT 特征
- `add_wavelet_packet_features()`: 对多个信号源（close, open, volume, cvd, taker_buy_ratio）做滚动窗口 WPT

**特点**：
- ✅ 滚动窗口方式（每次只对窗口内的数据做 WPT）
- ✅ 包含归一化处理（滚动窗口 z-score + IQR）
- ✅ 输出特征：`{source}_wpt_*_energy`, `{source}_wpt_*_mean`, `{source}_wpt_*_std` 等

**使用场景**：
- 被 `ComprehensiveFeatureEngineer` 使用
- 批量特征工程场景

---

## 问题分析

### 重复实现
1. **基础 WPT 分解逻辑重复**：
   - `utils_wpt_features.py` 中的 `wpt_decompose()` 和 `enhanced_features.py` 中的 `calculate_wavelet_packet_features()` 都实现了 WPT 分解
   - 但实现方式略有不同（一个是对整个序列，一个是滚动窗口）

2. **特征输出不同**：
   - `utils_wpt_features.py` 输出：趋势、波动、能量比等
   - `enhanced_features.py` 输出：每个子带的能量、均值、标准差等

### 是否可以合并？

**建议：部分合并，保留两者**

**理由**：
1. **使用场景不同**：
   - `utils_wpt_features.py` 适合按需加载、独立使用
   - `enhanced_features.py` 适合批量特征工程、滚动窗口处理

2. **输出特征不同**：
   - `utils_wpt_features.py` 关注趋势/波动分离和能量比
   - `enhanced_features.py` 关注每个子带的详细统计信息

3. **处理方式不同**：
   - `utils_wpt_features.py` 对整个序列一次性处理
   - `enhanced_features.py` 滚动窗口处理（更适合在线/增量场景）

---

## 优化方案

### 方案 1：让 `enhanced_features.py` 复用 `utils_wpt_features.py`（推荐）

**优点**：
- 减少重复代码
- 保持向后兼容
- 统一 WPT 分解逻辑

**实现**：
```python
# enhanced_features.py
from src.features.time_series.utils_wpt_features import wpt_decompose

def calculate_wavelet_packet_features(self, data: np.ndarray) -> Dict[str, float]:
    """使用 utils_wpt_features 的基础函数"""
    wpt_result = wpt_decompose(data, wavelet=self.wavelet, level=self.wpt_level)
    # 然后从 wpt_result 中提取需要的特征
    ...
```

### 方案 2：完全合并（不推荐）

**缺点**：
- 需要重构大量代码
- 可能破坏现有功能
- 滚动窗口和全序列处理方式难以统一

---

## 最终建议

**保留两者，但让 `enhanced_features.py` 复用 `utils_wpt_features.py` 的基础函数**

1. **保留 `utils_wpt_features.py`**：
   - 作为独立的工具模块
   - 被策略特征文件使用
   - 被特征加载器使用

2. **优化 `enhanced_features.py`**：
   - 让 `calculate_wavelet_packet_features()` 调用 `utils_wpt_features.py` 的 `wpt_decompose()`
   - 减少重复代码
   - 保持滚动窗口处理方式（这是它的特色）

3. **文档说明**：
   - 明确两者的使用场景
   - 说明何时使用哪个

---

## 总结

- **`utils_wpt_features.py`**：独立工具，适合按需加载
- **`enhanced_features.py` 中的 WPT**：集成在特征工程流程中，适合批量处理
- **建议**：保留两者，但让 `enhanced_features.py` 复用 `utils_wpt_features.py` 的基础函数

