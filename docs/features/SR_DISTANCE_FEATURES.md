# SR 距离和附近判断特征说明

## 1. dist_to_nearest_sr（到最近SR的距离）

### 计算逻辑

**位置**: `src/features/time_series/baseline_features.py` 的 `_add_price_action_features` 函数

**计算步骤**:

1. **收集所有SR边界**:
   - 从 `boundaries` 配置中获取所有SR边界列（如 `poc_hal_high`, `poc_hal_low`, `vwap` 等）
   - 每个边界都有类型：`support`（支撑）、`resistance`（阻力）、`mid`（中间，如VWAP）

2. **找到最近的SR边界**:
   ```python
   for i in range(len(data)):
       # 获取当前时刻所有非NaN的边界价格
       valid_boundaries = boundary_df.iloc[i].dropna()
       
       current_price = data["close"].iloc[i]
       # 找到最近的边界
       distances = abs(valid_boundaries - current_price)
       nearest_idx = distances.idxmin()
       nearest_price = valid_boundaries[nearest_idx]
   ```

3. **计算相对距离**:
   ```python
   dist_to_sr.iloc[i] = (
       (current_price - nearest_price) / current_price
       if current_price > 0
       else 0.0
   )
   ```
   - **单位**: 相对百分比（不是绝对价格差）
   - **正负含义**:
     - **正数**: 当前价格 > SR价格（价格在SR上方）
     - **负数**: 当前价格 < SR价格（价格在SR下方）
   - **绝对值**: 距离的远近

4. **避免未来信息泄露**:
   ```python
   data["dist_to_nearest_sr"] = dist_to_sr.shift(1).fillna(0.0)
   ```
   - 使用 `shift(1)` 确保在时刻 `t` 使用的是 `t-1` 的距离，避免未来信息泄露

### 示例

假设：
- 当前价格: 100
- 最近SR价格: 95（支撑位）

计算：
- `dist_to_nearest_sr = (100 - 95) / 100 = 0.05` (5%)
- 含义：价格在支撑位上方 5%

如果当前价格是 90：
- `dist_to_nearest_sr = (90 - 95) / 90 = -0.056` (-5.6%)
- 含义：价格在支撑位下方 5.6%

---

## 2. 其他SR相关特征

### 2.1 direction_to_nearest_sr（到最近SR的方向）

**计算逻辑**:
```python
direction_to_sr.iloc[i] = 1.0 if current_price < nearest_price else -1.0
```

**含义**:
- `1.0`: 价格在SR下方（向上接近SR）
- `-1.0`: 价格在SR上方（向下接近SR）

**用途**: 判断价格相对于SR的位置方向

---

### 2.2 sr_distance_normalized（归一化SR距离）

**计算逻辑**:
```python
sr_distance_normalized = dist_to_nearest_sr / atr
```

**位置**: `src/features/time_series/utils_interaction_features.py` 的 `compute_sr_distance_normalized` 函数

**含义**:
- 将相对距离转换为 ATR 倍数
- 更直观：1.5 ATR 表示距离为 1.5 倍的平均真实波幅

**用途**: 
- 判断是否在SR附近（如 `sr_distance_normalized <= 1.5` 表示在1.5倍ATR内）
- 用于特征工程和模型训练

---

### 2.3 is_near_sr（是否在SR附近）

**计算逻辑**:
```python
dist_normalized = abs(dist_to_nearest_sr) / atr
is_near = dist_normalized <= dist_atr_mult  # 默认 dist_atr_mult = 1.5
```

**位置**: `src/features/time_series/utils_interaction_features.py` 的 `compute_is_near_sr` 函数（新添加）

**含义**:
- **布尔值**: `True` 表示在SR附近（距离 <= 1.5 ATR），`False` 表示不在附近
- **阈值**: 默认 `dist_atr_mult = 1.5`，可配置

**用途**:
- 标签过滤：只在SR附近生成标签
- 特征工程：作为二值特征输入模型

---

### 2.4 sr_density（SR密度）

**计算逻辑**:
```python
tolerance = data["atr"].iloc[i] * tolerance_window  # tolerance_window = 0.5
# 计算在当前价格 ± tolerance 范围内的边界数量
count = 0
for boundary in boundaries:
    sr_price = data[boundary["column"]].iloc[i]
    if abs(sr_price - current_price) <= tolerance:
        count += 1
sr_density.iloc[i] = count
```

**含义**:
- 在价格附近 ±0.5 ATR 范围内的SR边界数量
- **高密度**: 多个SR重合，可能形成强支撑/阻力
- **低密度**: 单一SR，可能较弱

**用途**: 判断SR强度，多个SR重合时反转更可靠

---

### 2.5 breakout_status（突破状态）

**计算逻辑**:
```python
if sr_type == "resistance":
    # 刚上破阻力
    if current_high > nearest_sr_price and prev_close <= nearest_sr_price:
        breakout_status.iloc[i] = 1
elif sr_type == "support":
    # 刚下破支撑
    if current_low < nearest_sr_price and prev_close >= nearest_sr_price:
        breakout_status.iloc[i] = -1
```

**含义**:
- `1`: 向上突破阻力位
- `-1`: 向下突破支撑位
- `0`: 无突破

**用途**: 检测突破事件，用于突破策略

---

### 2.6 price_reversed_before_sr（未到SR就回头）

**计算逻辑**:
```python
# 应上涨但回落（距离SR为正，方向为正，但价格下跌）
if dist_to_sr.iloc[i] > 0 and direction_to_sr.iloc[i] == 1:
    if data["close"].iloc[i] < data["close"].iloc[i - 1]:
        # 检查成交量是否放大
        if data["volume"].iloc[i] / avg_vol > volume_spike_threshold:
            price_reversed_before_sr.iloc[i] = True
```

**含义**:
- **布尔值**: 价格在接近SR时提前反转
- **条件**: 价格应该继续向SR移动，但反而回头，且成交量放大

**用途**: 反转策略的信号，可能表示SR有效

---

### 2.7 fake_breakout（假突破迹象）

**计算逻辑**:
```python
# 检查突破后3根K线是否收回
if breakout_status.iloc[check_idx] == 1:  # 向上突破
    # 如果后续收盘价回到阻力位下方，可能是假突破
    if (data["close"].iloc[check_idx + 1 : i + 1] < nearest_sr_price).any():
        fake_breakout.iloc[i] = True
```

**含义**:
- **布尔值**: 突破后价格又回到SR另一侧
- **时间窗口**: 突破后3根K线内

**用途**: 识别假突破，避免在假突破时入场

---

## 3. 特征对比总结

| 特征名 | 类型 | 单位 | 含义 | 用途 |
|--------|------|------|------|------|
| `dist_to_nearest_sr` | 连续值 | 相对百分比 | 到最近SR的相对距离（正=上方，负=下方） | 基础距离特征 |
| `direction_to_nearest_sr` | 离散值 | ±1 | 方向（1=下方，-1=上方） | 方向判断 |
| `sr_distance_normalized` | 连续值 | ATR倍数 | 归一化距离（绝对值） | 距离判断 |
| `is_near_sr` | 布尔值 | True/False | 是否在SR附近（≤1.5 ATR） | 标签过滤、特征 |
| `sr_density` | 整数 | 数量 | 附近SR数量（±0.5 ATR内） | SR强度判断 |
| `breakout_status` | 离散值 | 0/±1 | 突破状态 | 突破策略 |
| `price_reversed_before_sr` | 布尔值 | True/False | 未到SR就回头 | 反转信号 |
| `fake_breakout` | 布尔值 | True/False | 假突破迹象 | 假突破识别 |

---

## 4. 使用建议

### 4.1 标签过滤

**方案1**: 使用 `is_near_sr`（需要先计算该特征）
```yaml
label_generator:
  params:
    sr_mask_col: is_near_sr
    dist_atr_mult: 1.5
```

**方案2**: 使用 `dist_to_nearest_sr`（直接可用）
```yaml
label_generator:
  params:
    dist_to_sr_col: dist_to_nearest_sr
    dist_atr_mult: 1.5
```

### 4.2 特征工程

- **距离特征**: `dist_to_nearest_sr`, `sr_distance_normalized`
- **方向特征**: `direction_to_nearest_sr`
- **强度特征**: `sr_density`
- **行为特征**: `breakout_status`, `price_reversed_before_sr`, `fake_breakout`

### 4.3 模型训练

- 可以将这些特征作为输入特征
- `is_near_sr` 可以作为二值特征
- `sr_density` 可以表示SR强度（多个SR重合时更强）

---

## 5. 注意事项

1. **未来信息泄露**: 所有特征都使用了 `shift(1)` 避免未来信息泄露
2. **边界类型**: SR边界有不同类型（support/resistance/mid），计算时会考虑
3. **缺失值处理**: 如果没有SR边界，距离会设为 0.0
4. **归一化**: 使用 ATR 归一化可以适应不同波动率环境

