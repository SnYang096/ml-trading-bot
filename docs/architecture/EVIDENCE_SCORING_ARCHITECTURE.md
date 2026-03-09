# Evidence Scoring 架构规范

## 1. 模块职责分离

| 模块         | 方法                        | 输出           | 职责                        |
| ------------ | --------------------------- | -------------- | --------------------------- |
| Gate         | 硬规则 + time_session       | allow/deny     | **做不做** — 尾部风险 veto  |
| Evidence     | Spearman → quantile mapping | 0~1 连续分     | **仓位缩放** — 环境质量评估 |
| Entry Filter | Spearman → bins + OR        | pass/fail      | **入场过滤** — 二值入场条件 |
| Execution    | execution.yaml              | SL/TP/trailing | **怎么做** — 执行参数       |

### 1.1 Gate vs Evidence

- **Gate** 回答 P(return < -0.8R)，关注极端尾部风险
- **Evidence** 回答 E[return | environment]，关注期望收益
- 两者正交：Gate 可以放行一笔尾部安全但期望不高的交易（Evidence 给低分缩小仓位）
- `time_session` 等结构性条件留在 Gate（硬 veto），不走 Entry Filter

### 1.2 Evidence vs Entry Filter

- **Evidence** = 连续仓位缩放（每个特征 5-bin quantile mapping → suppress/downweight/neutral/favor/amplify）
- **Entry Filter** = 二值入场过滤（每个特征独立分 bins，OR 组合 → pass/fail）
- `min_score` 作为极端兜底（evidence score < threshold 直接拒绝），主入场决策走 Entry Filter
- Entry Filter 的 bins+OR 比 Evidence 的 min_score 更好：每个特征独立验证，更鲁棒，且只要任一 OR 条件满足就放行

### 1.3 不用模型 → interaction alpha 损失

150 样本下 2D interaction 不可靠，系统已有规范禁止 2D interaction surface。单特征 Spearman + 单调性验证在小样本下更稳定。等数据量 >1000 再考虑模型。

## 2. Evidence 候选发现方法

### 2.1 全量扫描（当前实现）

`discover_evidence_candidates.py` 扫描 `logs_gated.parquet` 全部数值列：

1. **自动排除**：gate/prefilter/direction 已用特征（防 double counting）+ 元数据 + 前瞻标签
2. **可选配置**：`evidence_candidates.yaml` 提供额外排除 + 分类标签提示（不限制候选池）
3. **Spearman + Quintile 分析**：对每个特征计算 spearman_r、p_value、5-bin 单调性
4. **筛选**：p < 0.01, WR/tail/expectancy 任一单调性 ≥ 0.6, 方向一致
5. **相关性去重**：|corr| > 0.7 的特征只保留 |sp_r| 最大的（替代手动冗余组）
6. **输出**：top 5 → `archetypes/evidence.yaml`

### 2.2 为什么全量扫描而非人工候选池

旧设计在 `evidence_candidates.yaml` 中手动列出 ~40 个候选特征。问题：

- **遗漏风险**：人工难以枚举全部有效特征（如 `vpin_volatility_20` 在全量扫描中 sp_r=-0.514 但从未被列入候选）
- **维护负担**：每次新增特征都需手动更新候选池
- **冗余组过时**：手动维护的冗余组可能不反映真实相关性

全量扫描 + 统计去重 = 零人工干预 + 自动发现。`evidence_candidates.yaml` 退化为仅提供：
- `exclude`：额外排除（如已用于 position sizing 的 vol_percentile_approx）
- `category_hints`：分类标签（仅影响报告可读性）

## 3. Evidence Curve 与 Score Calibration

### 3.1 理论框架

Evidence Curve 描述 **环境特征值 → alpha 强度** 的映射关系：

```
Evidence Curve: f(feature_value) → alpha_multiplier ∈ [0, 1]
```

- 横轴：环境特征值（如 VPIN、波动率）
- 纵轴：alpha 强度（用 Quintile WR/avgR 代理）
- 形状通常是单调的（高 VPIN → 低 alpha，或反之）

Score Calibration 将多特征 Evidence Score 校准为概率尺度：

```
Calibration: g(evidence_score) → P(trade quality > threshold)
```

方法选项：
- **Isotonic Regression**：非参数校准，需 ≥1000 样本
- **Platt Scaling**：逻辑回归校准，需 ≥500 样本
- **Quantile Mapping**：离散版校准，5-bin 分位映射，≥50 样本

### 3.2 当前实现（离散版）

当前系统采用离散 Evidence Curve（5-bin quantile mapping）：

1. 对每个 evidence 特征，按 quintile 分 5 组
2. 每组标记语义标签：suppress / downweight / neutral / favor / amplify
3. 多特征 score 加权求和 → evidence_score ∈ [0, 1]
4. min_score 做极端兜底

这本质是 **离散版 Isotonic Regression**：用分位数代替连续拟合，避免小样本过拟合。

### 3.3 升级路径

| 条件           | 方法                    | 说明                  |
| -------------- | ----------------------- | --------------------- |
| n < 200        | 5-bin quantile mapping  | 当前实现，离散版      |
| 200 ≤ n < 1000 | 10-bin quantile mapping | 更细粒度              |
| n ≥ 1000       | Isotonic Regression     | 连续版 Evidence Curve |

跨策略可比性（Score Calibration）在当前阶段不需要：150 样本做 Isotonic Regression 会过拟合。等单策略数据量 >1000 再统一校准。

## 4. 数据隔离

### 4.1 当前状态

- IS 数据 → 模型训练 → `predictions.parquet` 只输出 holdout 预测
- OOS 数据 = `logs_gated.parquet` 中 holdout 部分
- Gate Optimize / Evidence Discovery / Entry Filter 均在 OOS 上拟合阈值
- **问题**：拟合和评估是同一段 OOS 数据（验证集 = 测试集）

### 4.2 推荐方案：OOS 再划分

将 OOS 按时间 70/30 划分为 optimize 区 + eval 区：

- **optimize 区 (70%)**：Gate 阈值优化、Evidence 发现、Entry Filter 拟合
- **eval 区 (30%)**：最终回测评估，不参与任何拟合

为什么不用 IS 数据拟合阈值：IS 上的模型预测是过拟合的，gate-passed 分布不真实，在 IS 上学到的阈值无法泛化到 OOS。
