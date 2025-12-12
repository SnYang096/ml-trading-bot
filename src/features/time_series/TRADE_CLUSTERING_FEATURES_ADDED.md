# Trade Clustering 新增特征总结

## 一、新增特征列表

### 1. 净 Runs 特征（3个）

- **`trade_cluster_net_runs`**: 净 runs（buy_run_count - sell_run_count）
  - 用途：衡量买卖方向的净强度
  - 范围：无界（可为正负）
  
- **`trade_cluster_total_runs`**: 总 runs 数（buy_run_count + sell_run_count）
  - 用途：衡量市场活跃度
  - 范围：[0, +∞)
  
- **`trade_cluster_net_runs_ratio`**: 净 runs 比率（标准化）
  - 用途：标准化的净方向强度
  - 范围：[-1, 1]

### 2. 长度比率特征（4个）

- **`trade_cluster_max_run`**: 最大连续长度（buy 或 sell 中的较大值）
  - 用途：捕捉极端行为
  - 范围：[0, +∞)
  
- **`trade_cluster_buy_sell_max_ratio`**: 买方 vs 卖方最大连续长度比
  - 用途：买卖极端力量对比
  - 范围：[0, +∞)
  
- **`trade_cluster_buy_sell_avg_ratio`**: 买方 vs 卖方平均连续长度比
  - 用途：买卖平均力量对比
  - 范围：[0, +∞)
  
- **`trade_cluster_avg_run_length`**: 平均连续长度（所有 runs）
  - 用途：整体连续性的平均强度
  - 范围：[0, +∞)

### 3. 总长度特征（1个）

- **`trade_cluster_total_run_length`**: 总连续长度（buy + sell）
  - 用途：衡量所有连续交易的总强度
  - 范围：[0, +∞)

### 4. Z-score 特征（8个）

- **`trade_cluster_imbalance_zscore_{w}`**: 不平衡比率的 Z-score（w=20, 50）
  - 用途：识别超买/超卖状态
  - 范围：无界（通常 [-3, 3]）
  
- **`trade_cluster_net_runs_zscore_{w}`**: 净 runs 的 Z-score（w=20, 50）
  - 用途：识别净方向的异常
  - 范围：无界（通常 [-3, 3]）
  
- **`trade_cluster_max_buy_run_zscore_{w}`**: 最大买单 run 的 Z-score（w=20, 50）
  - 用途：识别极端买单行为
  - 范围：无界（通常 [-3, 3]）
  
- **`trade_cluster_max_sell_run_zscore_{w}`**: 最大卖单 run 的 Z-score（w=20, 50）
  - 用途：识别极端卖单行为
  - 范围：无界（通常 [-3, 3]）

### 5. 移动平均特征（4个）

- **`trade_cluster_net_runs_ma{w}`**: 净 runs 的移动平均（w=5, 10, 20）
- **`trade_cluster_total_runs_ma{w}`**: 总 runs 的移动平均（w=5, 10, 20）

## 二、特征统计

### 总计新增特征数

- **基础特征**：8个（已存在）
- **原有派生特征**：7个（已存在）
- **新增派生特征**：20个
- **总计特征数**：35个

### 特征分类

| 类别 | 数量 | 特征名称 |
|------|------|----------|
| 基础特征 | 8 | max_buy_run, max_sell_run, avg_buy_run, avg_sell_run, buy_run_count, sell_run_count, imbalance_ratio, directional_entropy |
| 比率特征 | 5 | max_run_ratio, avg_run_ratio, max_run, buy_sell_max_ratio, buy_sell_avg_ratio |
| 净/总特征 | 3 | net_runs, total_runs, net_runs_ratio |
| 长度特征 | 2 | total_run_length, avg_run_length |
| 移动平均 | 11 | directional_entropy_ma, max_buy_run_ma, imbalance_ratio_ma, net_runs_ma, total_runs_ma |
| Z-score | 8 | imbalance_zscore, net_runs_zscore, max_buy_run_zscore, max_sell_run_zscore |
| 变化率 | 1 | directional_entropy_change |

## 三、特征用途说明

### 短期预测（<1小时）

- **`trade_cluster_imbalance_ratio`**: 最重要的方向性特征
- **`trade_cluster_net_runs`**: 净方向强度
- **`trade_cluster_max_buy_run` / `trade_cluster_max_sell_run`**: 极端行为
- **`trade_cluster_imbalance_zscore_20`**: 超买/超卖信号

### 趋势识别

- **`trade_cluster_buy_sell_avg_ratio`**: 买卖力量对比
- **`trade_cluster_directional_entropy`**: 市场混乱度
- **`trade_cluster_net_runs_ma20`**: 净方向的趋势

### 反转信号

- **`trade_cluster_imbalance_zscore_50`**: 长期超买/超卖
- **`trade_cluster_max_buy_run_zscore_50`**: 极端买单后的反转
- **`trade_cluster_max_sell_run_zscore_50`**: 极端卖单后的反转

### 流动性监测

- **`trade_cluster_total_runs`**: 市场活跃度
- **`trade_cluster_avg_run_length`**: 平均连续性强度
- **`trade_cluster_total_run_length`**: 总连续性强度

## 四、使用建议

### 1. 特征选择

根据策略类型选择特征：
- **趋势跟踪**：使用 `imbalance_ratio`, `net_runs`, `buy_sell_avg_ratio`
- **反转策略**：使用 `imbalance_zscore`, `max_buy_run_zscore`, `max_sell_run_zscore`
- **波动率预测**：使用 `directional_entropy`, `total_runs`, `avg_run_length`

### 2. 特征工程

可以进一步组合特征：
- **强度指标**：`net_runs * avg_run_length`
- **极端指标**：`max_run * imbalance_ratio`
- **稳定性指标**：`directional_entropy / (total_runs + 1)`

### 3. 特征重要性

根据经验，以下特征通常最有价值：
1. ⭐⭐⭐⭐⭐ `trade_cluster_imbalance_ratio`
2. ⭐⭐⭐⭐ `trade_cluster_net_runs`
3. ⭐⭐⭐⭐ `trade_cluster_max_buy_run` / `trade_cluster_max_sell_run`
4. ⭐⭐⭐⭐ `trade_cluster_directional_entropy`
5. ⭐⭐⭐ `trade_cluster_imbalance_zscore_20`
6. ⭐⭐⭐ `trade_cluster_buy_sell_avg_ratio`

## 五、验证建议

1. **特征值范围检查**：
   - `imbalance_ratio` 应在 [-1, 1]
   - `directional_entropy` 应在 [0, 1]
   - `zscore` 通常在 [-3, 3]

2. **相关性检查**：
   - `imbalance_ratio` 与 `net_runs_ratio` 应该高度相关
   - `max_buy_run` 与 `max_sell_run` 应该独立

3. **时间序列检查**：
   - 特征应该与价格变动有一定相关性
   - 特征应该具有一定的预测能力

## 六、总结

✅ **已完成**：
- 补充了 20 个新的派生特征
- 涵盖了净 runs、长度比、Z-score 等关键特征
- 保持了与现有代码的兼容性

⚠️ **待优化**：
- 考虑支持固定时间窗口（而非固定 tick 数）
- 多尺度特征需要修改计算逻辑（较大改动）

📊 **特征总数**：35个（8基础 + 27派生）

