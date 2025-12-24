# 样本权重对比测试说明

## 策略对比

### 1. sr_reversal_rr_reg_long（原版本）
- **位置**：`config/strategies/sr_reversal_rr_reg_long/`
- **权重策略**：无（uniform，所有样本权重 = 1.0）
- **标签函数**：`compute_sr_reversal_rr_continuous_label`
- **特点**：基线版本，不使用样本权重，只做多（long_only）

### 2. sr_reversal_rr_reg_long_weighted（带权重版本）
- **位置**：`config/strategies/sr_reversal_rr_reg_long_weighted/`
- **权重策略**：`result_based_rr`（基于未来 RR 的结果驱动加权）
- **标签函数**：`compute_sr_reversal_rr_continuous_label_with_weights`
- **特点**：
  - 权重公式：`Logic_Score * log(1 + RR)`
  - 三重共振样本：逻辑分加成 1.5 倍
  - 亏损样本（RR < 1）：权重降至 0.05

## 对比测试方法

### 方法 1：分别训练两个版本

```bash
# 训练原版本
python scripts/train_strategy_pipeline.py \
  --config config/strategies/sr_reversal_rr_reg_long \
  --output_dir results/sr_reversal_rr_reg_long

# 训练带权重版本
python scripts/train_strategy_pipeline.py \
  --config config/strategies/sr_reversal_rr_reg_long_weighted \
  --output_dir results/sr_reversal_rr_reg_long_weighted
```

### 方法 2：使用对比脚本（如果有）

```bash
# 如果系统支持批量对比
python scripts/compare_strategies.py \
  --configs \
    config/strategies/sr_reversal_rr_reg_long \
    config/strategies/sr_reversal_rr_reg_long_weighted
```

## 评估指标对比

训练完成后，对比以下指标：

### 1. 模型性能指标
- **Precision（精准率）**：预期带权重版本更高
- **Recall（召回率）**：可能略有下降
- **F1-Score**：综合评估
- **RMSE**（回归任务）：预期带权重版本更低

### 2. 回测指标
- **Sharpe Ratio**：风险调整后收益
- **Max Drawdown**：最大回撤
- **Win Rate**：胜率
- **Average R/R**：平均风险回报比

### 3. 权重分布分析
检查权重分布：
- 高权重样本占比
- 权重范围
- 权重与标签的相关性

## 预期结果

### 带权重版本预期优势
1. **更高的 Precision**：模型更关注高质量信号
2. **更好的信号质量**：三重共振样本权重更高
3. **更少的噪音干扰**：亏损样本权重降低

### 可能的风险
1. **样本数量减少**：高权重样本可能较少
2. **过拟合风险**：过度关注少数高质量样本
3. **泛化能力**：需要验证在测试集上的表现

## 分析建议

### 1. 权重分布分析
```python
# 检查权重分布
import pandas as pd
import numpy as np

# 加载训练数据
df = pd.read_parquet("data/train.parquet")
weights = df["sample_weight"]

print(f"权重统计:")
print(f"  均值: {weights.mean():.2f}")
print(f"  中位数: {weights.median():.2f}")
print(f"  最大值: {weights.max():.2f}")
print(f"  最小值: {weights.min():.2f}")
print(f"  高权重样本 (>3.0): {(weights > 3.0).sum()} ({(weights > 3.0).mean()*100:.1f}%)")
```

### 2. 特征重要性对比
对比两个版本的特征重要性，看权重是否改变了模型关注的特征。

### 3. 样本分布对比
对比两个版本训练时使用的样本分布，看权重是否有效筛选了高质量样本。

## 文件位置

- **原版本配置**：`config/strategies/sr_reversal_rr_reg_long/labels.yaml`
- **带权重版本配置**：`config/strategies/sr_reversal_rr_reg_long_weighted/labels.yaml`
- **权重配置详情**：`config/strategies/sr_reversal_rr_reg_long_weighted/README.md`

## 注意事项

1. **确保特征完整**：带权重版本需要 VPIN、CVD 斜率、SR 强度等特征
2. **数据一致性**：两个版本使用相同的数据集
3. **随机种子**：建议设置相同的随机种子以确保可复现性
4. **训练时间**：带权重版本可能训练时间稍长（需要计算权重）

