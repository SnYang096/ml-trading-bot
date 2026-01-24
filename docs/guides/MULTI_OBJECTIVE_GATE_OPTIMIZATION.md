# Gate多目标优化策略指南

## 概述

Gate规则优化需要在多个目标之间进行权衡：
- **Robustness Score（稳健性分数）**：衡量规则在不同市场条件下的稳定性
- **Trade Rate（交易率）**：允许通过gate的交易比例

这两个目标通常是冲突的：更严格的规则（更高的robustness）会降低交易率，而更宽松的规则（更高的交易率）可能降低稳健性。

## 多目标优化方法

### 1. Pareto前沿（Pareto Frontier）

Pareto前沿是一组"不被其他解支配"的最优解集合。一个解被支配意味着存在另一个解在所有目标上都更好或相等，且至少在一个目标上更优。

**定义**：
- 解A支配解B，当且仅当：
  - A的robustness ≥ B的robustness 且 A的trade_rate ≥ B的trade_rate
  - 且至少有一个严格大于

**优点**：
- 提供多个最优解选择
- 不依赖主观权重
- 展示目标之间的权衡关系

### 2. 选择策略

#### 2.1 max_robustness（最大稳健性）⭐ **推荐**

**策略**：在满足最低trade_rate要求的前提下，选择robustness_score最大的阈值。

**适用场景**：
- 优先考虑规则的稳健性和可靠性
- 可以接受较低的交易率
- 适合生产环境，需要稳定的性能

**实现**：
```python
valid = results[
    (results["robustness_score"] >= min_robustness) &
    (results["trade_rate"] >= min_trade_rate)
]
best_idx = valid["robustness_score"].idxmax()
```

**推荐使用**：✅ **生产环境首选**

#### 2.2 max_trade_rate（最大交易率）

**策略**：在满足最低robustness要求的前提下，选择trade_rate最大的阈值。

**适用场景**：
- 需要最大化交易机会
- 可以接受较低的稳健性
- 适合探索性研究或数据充足的情况

**实现**：
```python
valid = results[
    (results["robustness_score"] >= min_robustness) &
    (results["trade_rate"] >= min_trade_rate)
]
best_idx = valid["trade_rate"].idxmax()
```

#### 2.3 balanced（平衡策略）

**策略**：使用加权平均选择平衡点，同时考虑robustness和trade_rate。

**公式**：
```
combined_score = w_r * (robustness / max_robustness) + w_t * (trade_rate / max_trade_rate)
```

其中：
- `w_r`：robustness权重（默认0.5）
- `w_t`：trade_rate权重（默认0.5）

**适用场景**：
- 需要在两个目标之间平衡
- 可以根据业务需求调整权重

#### 2.4 pareto_midpoint（Pareto前沿中点）

**策略**：选择Pareto前沿中robustness和trade_rate都接近中值的点。

**适用场景**：
- 希望选择Pareto前沿上的解
- 避免极端选择（最高robustness或最高trade_rate）

## 推荐策略

### 生产环境：max_robustness ⭐

**理由**：
1. **稳定性优先**：生产环境需要稳定的规则，robustness_score衡量规则在不同市场条件下的表现
2. **风险控制**：更高的robustness意味着规则更可靠，减少意外损失
3. **可预测性**：稳健的规则在不同市场条件下表现一致

**使用示例**：
```bash
python scripts/optimize_gate_plateau_progressive.py \
    --multi-objective \
    --multi-objective-strategy max_robustness \
    --min-trade-rate 0.001 \
    ...
```

### 研究环境：balanced

**理由**：
1. **探索性**：可以同时考虑两个目标
2. **灵活性**：可以通过调整权重适应不同需求

**使用示例**：
```bash
python scripts/optimize_gate_plateau_progressive.py \
    --multi-objective \
    --multi-objective-strategy balanced \
    --multi-objective-weights 0.6 0.4 \
    ...
```

## 集成到渐进式优化

在渐进式优化的第二步（平坦高原优化）中，如果启用多目标优化：

1. 扫描阈值，计算每个阈值的(robustness_score, trade_rate)对
2. 计算Pareto前沿
3. 根据选择策略从Pareto前沿或所有结果中选择最优阈值
4. 如果选择的是max_robustness，优先考虑稳健性

## 架构关联

Gate优化是交易系统架构中的关键组件：

```
FeatureStore → Gate Rules → Execution → Returns
                ↑
        多目标优化在这里
```

相关文档：
- [Gate优化架构](../architecture/FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md)
- [Hard-Gate System指南](./HARD_GATE_SYSTEM.md)
- [渐进式优化工作流](./PLATEAU_OPTIMIZATION_WORKFLOW.md)

## 实现细节

### Pareto前沿计算

```python
def compute_pareto_frontier(results: pd.DataFrame) -> pd.DataFrame:
    """
    计算Pareto前沿
    
    算法：
    1. 归一化robustness_score和trade_rate
    2. 对每个解，检查是否被其他解支配
    3. 返回不被支配的解（Pareto最优解）
    """
    # 归一化
    max_robustness = results["robustness_score"].max()
    max_trade_rate = results["trade_rate"].max()
    
    results_norm = results.copy()
    results_norm["robustness_norm"] = results_norm["robustness_score"] / max_robustness
    results_norm["trade_rate_norm"] = results_norm["trade_rate"] / max_trade_rate
    
    # 找到Pareto最优解
    pareto_indices = []
    for idx, row in results_norm.iterrows():
        is_pareto = True
        for other_idx, other_row in results_norm.iterrows():
            if idx == other_idx:
                continue
            # 检查是否被支配
            if (other_row["robustness_norm"] >= row["robustness_norm"] and
                other_row["trade_rate_norm"] >= row["trade_rate_norm"] and
                (other_row["robustness_norm"] > row["robustness_norm"] or
                 other_row["trade_rate_norm"] > row["trade_rate_norm"])):
                is_pareto = False
                break
        if is_pareto:
            pareto_indices.append(idx)
    
    return results.loc[pareto_indices]
```

### 选择策略实现

见 `scripts/optimize_gate_plateau.py` 中的 `select_multi_objective_threshold` 函数。

## 使用建议

1. **首次优化**：使用 `max_robustness` 策略，确保规则稳健
2. **调整交易率**：如果交易率太低，可以降低 `min_robustness` 要求或使用 `balanced` 策略
3. **探索Pareto前沿**：使用 `pareto_midpoint` 查看权衡关系
4. **生产部署**：始终使用 `max_robustness` 策略

## 注意事项

1. **最低要求**：无论使用哪种策略，都必须满足最低robustness和trade_rate要求
2. **数据质量**：多目标优化的效果依赖于数据的质量和代表性
3. **过拟合风险**：避免过度优化，保持规则的简洁性
4. **定期重优化**：市场条件变化时，需要重新优化规则
