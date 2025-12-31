# Factor-Eval 输出特征数量分析

## 筛选标准

`factor-eval` 使用以下标准筛选 qualified factors：

### Positive Alpha Candidates（正向因子）

- **IC Mean > 0**：平均预测能力为正
- **IC t-stat > 1.96**：统计显著（p-value < 0.05，95% 置信度）
- **(IC IR > 0 or Sharpe > 0)**：信息比率或夏普比率为正（宽松约束，保留更多候选）
- **ic_p_value < 1.0**：排除数据不一致的特征

### Strong Negative Factors（强负向因子）

- **IC Mean < 0**：平均预测能力为负
- **IC t-stat < -1.96**：统计显著（p-value < 0.05，95% 置信度）
- **(IC IR < 0 or Sharpe < 0)**：信息比率或夏普比率为负
- **ic_p_value < 1.0**：排除数据不一致的特征

### 额外过滤（可选）

- **`--remove-correlated`**：移除高度相关的特征（默认阈值：0.9）
- **`--filter-by-best-lag`**：只保留最佳 lag 与目标 lag 匹配的特征（默认容差：±5 bars）

## 经验输出数量

### 典型情况

基于实际运行经验：

- **输入特征数**：~200-650 个（从 `features_all.yaml`）
- **经过 IC/IR 筛选后**：**50-150 个** qualified factors
  - Positive factors: 30-80 个
  - Negative factors: 20-70 个
- **经过相关性过滤后**：**30-100 个**（如果使用 `--remove-correlated`）
- **经过 lag 过滤后**：**20-80 个**（如果使用 `--filter-by-best-lag`）

### 影响因素

1. **数据质量**：数据质量越好，qualified factors 越多
2. **策略类型**：不同策略的 qualified factors 数量差异很大
3. **时间窗口**：更长的历史数据通常产生更多 qualified factors
4. **筛选严格程度**：
   - 严格（t-stat > 2.58，p < 0.01）：更少但更可靠
   - 宽松（t-stat > 1.96，p < 0.05）：更多但可能包含噪声

### 实际案例

**SR Reversal (sr_reversal_rr_reg_long)**：
- 输入：~650 个特征
- 输出：~100-150 个 qualified factors

**SR Breakout (sr_breakout)**：
- 输入：~200 个特征
- 输出：~50-80 个 qualified factors

**Compression Breakout (compression_breakout)**：
- 输入：~200 个特征
- 输出：~40-70 个 qualified factors

## 没有硬性数量限制

**重要**：`factor-eval` **不限制输出数量**，所有满足条件的特征都会被输出。这意味着：

- 如果很多特征都满足条件，输出会很多（例如 200+ 个）
- 如果很少特征满足条件，输出会很少（例如 10-20 个）
- 这是**数据驱动**的结果，不是人为设定的阈值

## 优化建议

如果输出特征过多（> 200 个），可以考虑：

1. **提高筛选标准**：
   - 提高 t-stat 阈值（例如 > 2.58，p < 0.01）
   - 要求 IC IR > 0.1（而不是 > 0）
   - 要求 Sharpe > 0.5（而不是 > 0）

2. **使用相关性过滤**：
   ```bash
   --remove-correlated --correlation-threshold 0.85  # 更严格
   ```

3. **使用 lag 过滤**：
   ```bash
   --filter-by-best-lag --target-lag 10 --lag-tolerance 3  # 更严格
   ```

4. **Top-K 选择**（需要手动后处理）：
   - 按 IC IR 或 Sharpe 排序
   - 只保留 top 50-100 个

## 相关文档

- `docs/strategies/RECOMMENDED_FEATURE_WORKFLOW.md`：推荐的特征工作流
- `docs/strategies/SEMANTIC_GROUPS_SINGLETON_EXPANSION.md`：Semantic groups 单例展开

