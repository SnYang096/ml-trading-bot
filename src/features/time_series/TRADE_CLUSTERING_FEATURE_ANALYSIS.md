# Trade Clustering 特征分析与优化建议

## 一、当前实现分析

### ✅ 已实现的基础特征

1. **`trade_cluster_max_buy_run`**: 窗口内最大连续买单数
2. **`trade_cluster_max_sell_run`**: 窗口内最大连续卖单数
3. **`trade_cluster_avg_buy_run`**: 窗口内平均连续买单数
4. **`trade_cluster_avg_sell_run`**: 窗口内平均连续卖单数
5. **`trade_cluster_buy_run_count`**: 窗口内买单 run 的数量
6. **`trade_cluster_sell_run_count`**: 窗口内卖单 run 的数量
7. **`trade_cluster_imbalance_ratio`**: 不平衡比率 `(buy_count - sell_count) / (buy_count + sell_count)`
8. **`trade_cluster_directional_entropy`**: 方向熵（衡量买卖 runs 的随机性）

### ✅ 已实现的派生特征

1. **`trade_cluster_max_run_ratio`**: 最大连续长度比率
2. **`trade_cluster_avg_run_ratio`**: 平均连续长度比率
3. **`trade_cluster_directional_entropy_ma{w}`**: 方向熵的移动平均（w=5,10,20）
4. **`trade_cluster_directional_entropy_change`**: 方向熵的变化率
5. **`trade_cluster_directional_entropy_zscore_{w}`**: 方向熵的 Z-score（w=20,50）
6. **`trade_cluster_max_buy_run_ma{w}`**: 最大买单 run 的移动平均（w=5,10,20）
7. **`trade_cluster_imbalance_ratio_ma{w}`**: 不平衡比率的移动平均（w=5,10,20）

## 二、计算逻辑验证

### ✅ Run 定义正确

- 连续相同 `side` 的交易被识别为一个 run
- `side = 1` 为买方主动，`side = -1` 为卖方主动
- 方向改变时结束当前 run，开始新 run

### ✅ 滑动窗口实现

- 使用固定 tick 数（`window_size`）作为滑动窗口
- 使用 `deque` 维护窗口内的 runs，高效实现 FIFO
- 支持跨月状态传递，确保连续性

### ⚠️ 潜在问题

1. **窗口大小按 tick 数而非时间**：
   - 当前：固定 tick 数（如 100 个 tick）
   - 建议：考虑使用固定时间窗口（如 1 小时），因为 tick 频率变化大
   - 影响：在高频和低频时段，窗口的实际时间跨度不同

2. **当前 run 的部分窗口处理**：
   - 代码正确处理了当前 run 可能部分在窗口内的情况
   - 使用 `min(current_run_length, remaining_window)` 计算窗口内的部分

## 三、缺失的特征（建议补充）

根据学术/业界实践，以下特征对量化交易很有价值：

### 1. 净 Runs 特征

```python
# 净 runs（buy - sell）
trade_cluster_net_runs = buy_run_count - sell_run_count

# 总 runs 数（衡量活跃度）
trade_cluster_total_runs = buy_run_count + sell_run_count

# 净 runs 比率（标准化）
trade_cluster_net_runs_ratio = net_runs / (total_runs + TOL)
```

### 2. 长度比率特征

```python
# 买方 vs 卖方平均连续长度比
trade_cluster_buy_sell_avg_ratio = avg_buy_run / (avg_sell_run + TOL)

# 最大连续长度比
trade_cluster_buy_sell_max_ratio = max_buy_run / (max_sell_run + TOL)
```

### 3. 强度特征

```python
# 总连续长度（buy + sell）
trade_cluster_total_run_length = avg_buy_run * buy_run_count + avg_sell_run * sell_run_count

# 平均连续长度（所有 runs）
trade_cluster_avg_run_length = total_run_length / (total_runs + TOL)
```

### 4. 极端行为特征

```python
# 最大连续长度（buy 或 sell 中的较大值）
trade_cluster_max_run = max(max_buy_run, max_sell_run)

# 最大连续长度比率（标准化）
trade_cluster_max_run_normalized = max_run / (window_size + TOL)
```

### 5. 反转信号特征

```python
# Runs 的 Z-score（识别超买/超卖）
for w in [20, 50]:
    rolling_mean = imbalance_ratio.rolling(window=w).mean()
    rolling_std = imbalance_ratio.rolling(window=w).std()
    trade_cluster_imbalance_zscore_{w} = (imbalance_ratio - rolling_mean) / (rolling_std + TOL)
```

### 6. 多尺度特征（如果支持多时间窗口）

```python
# 不同时间窗口的聚合（需要修改计算逻辑支持多窗口）
# 例如：5分钟、1小时、1天的聚类比率
trade_cluster_ratio_5min
trade_cluster_ratio_1h
trade_cluster_ratio_1d
```

## 四、建议的优化

### 1. 补充缺失特征

在 `extract_trade_clustering_features` 函数的派生特征部分添加上述特征。

### 2. 考虑时间窗口

如果可能，支持固定时间窗口（如 1 小时）而非固定 tick 数，使特征更稳定。

### 3. 添加验证测试

创建测试用例验证：
- Run 检测的正确性
- 窗口滑动的正确性
- 跨月连续性的正确性
- 特征值的合理性（如 entropy 应在 [0, 1] 范围内）

## 五、特征重要性评估

根据量化交易实践，以下特征通常最有价值：

1. **`trade_cluster_imbalance_ratio`**: ⭐⭐⭐⭐⭐ 最重要的方向性特征
2. **`trade_cluster_max_buy_run` / `trade_cluster_max_sell_run`**: ⭐⭐⭐⭐ 捕捉极端行为
3. **`trade_cluster_directional_entropy`**: ⭐⭐⭐⭐ 衡量市场混乱度
4. **`trade_cluster_net_runs`**: ⭐⭐⭐⭐ 净方向强度（建议补充）
5. **`trade_cluster_buy_sell_avg_ratio`**: ⭐⭐⭐ 买卖力量对比（建议补充）
6. **`trade_cluster_imbalance_zscore`**: ⭐⭐⭐ 超买/超卖信号（建议补充）

## 六、总结

| 项目 | 状态 | 说明 |
|------|------|------|
| Run 定义 | ✅ 正确 | 连续相同 side 的交易被正确识别 |
| 滑动窗口 | ✅ 实现 | 使用 deque 高效实现，但建议考虑时间窗口 |
| 跨月连续性 | ✅ 支持 | 通过 initial_state 实现 |
| 基础特征 | ✅ 完整 | 8 个基础特征都已实现 |
| 派生特征 | ⚠️ 部分 | 已实现部分，但缺少净 runs、长度比等 |
| 多尺度特征 | ❌ 缺失 | 需要修改计算逻辑支持多时间窗口 |

**建议优先级**：
1. **高优先级**：补充净 runs、长度比、Z-score 特征
2. **中优先级**：考虑支持固定时间窗口
3. **低优先级**：多尺度特征（需要较大改动）

