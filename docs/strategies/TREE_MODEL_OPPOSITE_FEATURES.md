# 树模型对相反特征的处理能力

## 问题

当 semantic groups 不展开时，同一个 feature node 可能包含多个语义（如 `trade_cluster_scene_semantic_scores_f` 包含 compression/ignition/absorption/exhaustion），这些语义可能对策略有相反的作用。树模型能否自动处理这些相反的特征？

## 树模型的能力

### 理论上可以处理

**树模型（LightGBM/XGBoost）理论上可以处理相反特征**：

1. **特征重要性学习**：
   - 树模型可以通过特征重要性自动学习哪些特征有用
   - 如果某个特征（或特征组合）对目标有害，模型会降低其重要性或忽略它

2. **条件分割**：
   - 树模型可以通过条件分割（if-else）学习特征之间的交互
   - 例如：`if ignition_score > 0.7: use_ignition; else: use_exhaustion`

3. **正则化**：
   - L1/L2 正则化可以自动将无用特征的权重降为 0
   - `min_gain_to_split` 等参数可以防止过拟合

### 实际限制

**但实际中可能不够好**：

1. **样本不足**：
   - 如果相反特征的数量 > 样本数，模型可能无法充分学习
   - 例如：100 个相反特征，但只有 1000 个样本

2. **噪声干扰**：
   - 如果相反特征引入噪声，模型可能学习到错误的模式
   - 例如：`ignition` 和 `exhaustion` 同时存在，模型可能混淆

3. **特征重要性不稳定**：
   - 在样本不足或噪声较大时，特征重要性可能不稳定
   - 不同 seed 可能得到不同的特征重要性排序

4. **过拟合风险**：
   - 如果相反特征过多，模型可能过拟合到训练数据
   - 在测试集上表现可能很差

## 实验证据

### 不展开 semantic groups 的效果

**从实际运行结果看**：

- **SR Reversal**：`vpin_scene`（包含 4 个语义）单独使用，Sharpe 达到 1.52
- **SR Breakout**：`wpt_scene`（包含 4 个语义）单独使用，Sharpe 达到 1.11

**结论**：
- 树模型**可以**处理不展开的 semantic groups
- 但可能**不是最优**的，因为模型需要学习如何组合相反的语义

### 展开 semantic groups 的潜在优势

**理论优势**：

1. **更精细的选择**：
   - 可以选择只使用 `ignition` 而不使用 `exhaustion`
   - 避免模型学习错误的组合

2. **更稳定的特征重要性**：
   - 每个语义单独评估，特征重要性更稳定
   - 不同 seed 的结果更一致

3. **更好的可解释性**：
   - 可以明确知道哪些语义对策略有效
   - 例如：只选择 `exhaustion` 和 `compression`，不选择 `ignition`

## 推荐策略

### 方案 1：先不展开，看效果（推荐）

1. **先运行不展开的版本**：
   ```bash
   mlbot diagnose feature-group-search \
     --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
     --pool-b-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
     ...
   ```

2. **如果效果不好**（例如 Sharpe < 1.0 或过拟合），再尝试展开：
   ```bash
   mlbot diagnose feature-group-search \
     --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
     --pool-b-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
     --expand-semantic-singletons \
     ...
   ```

### 方案 2：直接展开（如果担心相反特征）

如果担心相反特征会影响模型学习，可以直接展开：

```bash
mlbot diagnose feature-group-search \
  --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
  --pool-b-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  --expand-semantic-singletons \
  ...
```

**优点**：
- 更精细的特征选择
- 避免相反语义冲突
- 更好的可解释性

**缺点**：
- 评估时间增加约 27%
- 候选组数量增加

## 性能对比

### 不展开（当前默认）

- **候选组数**：Pool B (100) + Semantic groups (10) = 110
- **评估时间**：~110 小时（假设每次 2 分钟）
- **优点**：快速，简单
- **缺点**：可能包含相反语义

### 展开后

- **候选组数**：Pool B (100) + Semantic singletons (40) = 140
- **评估时间**：~140 小时（增加 27%）
- **优点**：更精细，避免冲突
- **缺点**：更慢

## 结论

1. **树模型理论上可以处理相反特征**，但实际效果可能不够好
2. **不展开 semantic groups 可能有效**（从实际结果看），但可能不是最优
3. **展开 semantic groups 更安全**，可以获得更精细的特征选择
4. **推荐**：先尝试不展开，如果效果不好再展开

## 相关文档

- `docs/strategies/SEMANTIC_GROUPS_SINGLETON_EXPANSION.md`：Semantic groups 单例展开详细说明
- `docs/strategies/RECOMMENDED_FEATURE_WORKFLOW.md`：推荐的特征工作流

