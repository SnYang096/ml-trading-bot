# 数据切分与 Look-Ahead 风险分析

> 日期: 2026-02-22
> 结论: 全量 holdout 找阈值 + 定期重训 = 最优自动化方案

## 1. Look-Ahead 风险评估

### 模型训练 ✅ 无泄漏
```
Train: 2023-01 ~ 2024-05   →  模型训练 (内部 CV)
Holdout: 2024-05 ~ 2026-01 →  predictions.parquet (模型未见过)
```

### 后处理阈值 ⚠️ 在 holdout 上优化

| 阶段 | 参数数量 | 风险 | 缓解 |
|------|---------|------|------|
| Gate | 1-2 个阈值 | 低 | plateau 稳定区间 |
| Evidence | 2-3 个权重 | 低 | plateau + z-test |
| Entry Filter | 1-2 个阈值 | 低 | plateau + 去冗余 + z-test |
| Execution | 3 个参数 | 中 | grid search + plateau CV |

**结论**:
- 报告 Sharpe 高估 20-30% (vs 纯 OOS)
- Alpha 来源真实 (模型在纯 OOS 上预测)
- 阈值总共 <10 个 + plateau 方法 → 过拟合风险低

## 2. 为什么不需要额外的 OOS 切分

### 模型 vs 阈值的本质区别

| | 模型 | 阈值 |
|---|---|---|
| 目标 | 学通用模式 | 找当前市场的合理切点 |
| 参数量 | 上千 | <10 |
| 过拟合风险 | 高 → 需要严格 train/holdout | 低 → plateau 已足够 |
| 数据需求 | 全量历史 | 越多越稳定 |

### 关键洞察

**阈值稳定性 > Sharpe 精度**

- 划出 OOS 段 → Sharpe 更准，但阈值用更少数据 → 实盘反而更不稳定
- 全量 holdout 找阈值 → Sharpe 偏高 20-30%，但阈值最稳定 → 实盘更 robust
- Sharpe 高估的问题用**实盘真实数据**验证，不需要牺牲阈值质量

## 3. 重训时间窗口规则

### 自动化规则 (硬编码禁止)

```
end-date         = 数据集最新可用日期 (禁止固定)
start-date       = 2023-01-01 (或最早可用数据)
holdout-start    = end-date - 14 个月
Train            = start-date ~ holdout-start
Holdout          = holdout-start ~ end-date
```

### 当前数据集 (end-date = 2026-01-01)

```
Train:   2023-01-01 ~ 2024-11-01  (22 个月)
Holdout: 2024-11-01 ~ 2026-01-01  (14 个月)
```

之前用 `--holdout-start-date 2024-05-01` (20 个月 holdout) 偏保守，
模型少学了 6 个月数据。现已统一调整为 `2024-11-01`。

### 14 个月的依据

- FER 最稀疏: prefilter 后 ~167 条/月 → 14 月 ≈ 2,300 条，plateau 够用
- 覆盖至少 1 个完整市场周期 (牛→震荡→回调)
- 模型训练 22 个月 > 之前 16 个月，学习更充分

### 重训示例

```bash
# 2026 Q2 重训 (数据到 2026-04-01)
--start-date 2023-01-01 --end-date 2026-04-01 \
--holdout-start-date 2025-02-01 --holdout-end-date 2026-04-01

# 2026 Q3 重训 (数据到 2026-07-01)
--start-date 2023-01-01 --end-date 2026-07-01 \
--holdout-start-date 2025-05-01 --holdout-end-date 2026-07-01
```

## 4. 全自动化方案

**一个方案，不需要切换**：

```
┌──────────────────────────────────────────────────────┐
│                 定期重训 (每季度)                       │
│                                                      │
│  1. 模型训练: 全量历史 → predictions.parquet           │
│  2. 阈值优化: 全量 holdout (plateau 方法)              │
│  3. 回测评估: 全量 holdout                             │
│  4. 部署上线                                          │
│  5. 监控: 实盘 vs 回测 Sharpe 衰减                     │
│                                                      │
│  每季度重训时 holdout 自然前移 → 阈值自动基于最新 regime  │
└──────────────────────────────────────────────────────┘
```

### 重训时间线示例

```
2026 Q1:  Train 2023-01~2025-02  |  Holdout 2025-02~2026-04  ← 阈值基于最近14个月
2026 Q2:  Train 2023-01~2025-05  |  Holdout 2025-05~2026-07  ← 自然前移
2026 Q3:  Train 2023-01~2025-08  |  Holdout 2025-08~2026-10  ← 永远贴近当前市场
```

**为什么不需要"近6月"切分**: holdout 本身就是近期数据。定期重训后 holdout 自动前移，不需要手动选时间窗口。

### 自动化命令 (每次重训复制即用)

```bash
# 日期参数: 只需要调整 end-date 和 holdout-start-date
END_DATE="2026-04-01"           # 当前日期
HOLDOUT_START="2025-02-01"      # end-date 往前推 14 个月

# Step 1-8: 完全不变，用上面的日期参数
# Step 9: 全量回测 (不切分)
python scripts/backtest_execution_layer.py \
  --logs ${DIR}/predictions.parquet --strategy fer
```

## 4. 监控与应急

| 实盘 Sharpe 衰减 | 判断 | 操作 |
|---|---|---|
| < 30% | 正常，回测偏乐观是预期的 | 继续运行 |
| 30-50% | regime 可能变化 | 提前重训 (不等季度) |
| > 50% | 可能有结构性问题 | 暂停交易 + 诊断 |

### 诊断工具 (仅在衰减严重时使用)

如果需要精确测量"阈值过拟合程度"，可临时用 60/40 切分:

```bash
# 阈值优化: 只用前 60%
python scripts/optimize_gate_unified.py --strategy fer \
  --logs ${DIR}/logs_gated.parquet --end-date 2025-05-01

# 纯 OOS 评估: 后 40%
python scripts/backtest_execution_layer.py \
  --logs ${DIR}/predictions.parquet --strategy fer --start-date 2025-05-01
```

这需要给优化脚本加 `--start-date`/`--end-date` 支持 (当前未实现，改动很小)。

## 5. 需要的代码改动

### 当前阶段: 无需改动 ✅

当前流程（全量 holdout 找阈值）已是最优方案，直接上实盘。

### 未来 (届时实现):

```yaml
# 诊断用途，给优化脚本加时间过滤:
--start-date: 数据起始日期
--end-date: 数据截止日期

# 影响脚本:
# - optimize_gate_unified.py
# - optimize_evidence_plateau.py
# - optimize_entry_filter_plateau.py
# - optimize_execution_grid.py
# - backtest_execution_layer.py
```
