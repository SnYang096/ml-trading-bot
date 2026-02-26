# Walk-Forward Validation 设计与原理

## 一、解决什么问题

`auto_research_pipeline` 产出的 Sharpe（如 ME=0.59）可能包含过拟合成分：
- 模型预测本身是 OOS（模型只在 holdout 前的数据上训练）
- 但 **gate/evidence/entry_filter/execution 的优化是在 holdout 数据上做的**
- 最终回测也在 **同一份 holdout** 数据上
- 因此报告的 Sharpe 对优化层来说是 **样本内（In-Sample）**

**核心问题**: 优化层选出的参数在未来新数据上还能保持多少 Sharpe？

## 二、术语

| 术语 | 含义 |
|------|------|
| **IS (In-Sample)** | 用于优化参数的数据上评估的 Sharpe（乐观偏差） |
| **OOS (Out-of-Sample)** | 在参数优化时完全未见过的数据上评估的 Sharpe（真实估计） |
| **Fold** | 一个时间窗口，包含训练期 + holdout 期 |
| **Decay Ratio** | OOS Sharpe / IS Sharpe，衡量参数稳健性 |
| **Frozen Config** | 冻结的配置文件（gate.yaml, entry_filters.yaml, execution.yaml） |

## 三、Anchored Walk-Forward 原理

### 3.1 数据切分

将历史数据按时间分成 N 个不重叠的 fold，每个 fold 包含 6 个月的 holdout：

```
数据: 2023-01 ──────────────────────────────────────── 2026-02

Fold 1: train [2023-01 → 2024-08]  holdout [2024-08 → 2025-02]
Fold 2: train [2023-01 → 2025-02]  holdout [2025-02 → 2025-08]
Fold 3: train [2023-01 → 2025-08]  holdout [2025-08 → 2026-02]
         ↑ 训练窗口逐步扩大（anchored）   ↑ holdout 不重叠
```

**为什么叫 "Anchored"**: 训练窗口的起点固定（anchored at 2023-01），终点逐步后移。
每个 fold 的 holdout 期完全不重叠。

### 3.2 两阶段验证

**Phase 1: 运行各 fold（耗时）**

对每个 fold，调用完整的 `auto_research_pipeline`：
- 训练模型（LightGBM）
- 优化 prefilter / gate / evidence / entry_filter / execution
- 在 holdout 上回测 → 得到 **IS Sharpe**（因为优化和回测用的同一份 holdout）

**Phase 2: OOS 回测（轻量）**

对 Fold N（N ≥ 2）：
1. 取 Fold N 的**模型预测**（predictions.parquet，holdout 期）
2. 应用 Fold N-1 的**冻结配置**（gate.yaml / entry_filters.yaml / execution.yaml）
3. 在 Fold N 的 holdout 上回测 → 得到 **OOS Sharpe**

```
Fold 1 的配置 ──冻结──→ 应用到 Fold 2 的预测数据 → OOS Sharpe
Fold 2 的配置 ──冻结──→ 应用到 Fold 3 的预测数据 → OOS Sharpe
```

**为什么 Fold 1 是"校准"**: Fold 1 没有前一个 fold 的冻结配置可用，
只能产出 IS 结果，作为后续 fold 的配置供应者。所以它只有 IS、没有 OOS。

### 3.3 Decay Ratio 判读

```
Decay Ratio = OOS Sharpe / IS Sharpe

≥ 0.7  → ✅ 参数稳健，优化层没有引入严重过拟合
0.5~0.7 → ⚠️  中等衰减，可接受但需持续观察
0.3~0.5 → 🟡 较大衰减，优化层可能过拟合（减少优化参数数量）
< 0.3  → 🔴 严重过拟合，优化层在拟合噪声（需简化管线）
```

## 四、与 auto_research_pipeline 的关系

```
auto_research_pipeline    →  生产工具，产出可部署的策略配置
walk_forward_validation   →  验证工具，回答"配置能否在未来复现"
                              （内部多次调用 auto_research_pipeline）
```

WF 不替代 auto_research_pipeline，而是对其输出做独立验证。

## 五、使用方法

```bash
# 1. 预览 fold 配置（不执行）
python scripts/walk_forward_validation.py --strategy me --folds 3 --dry-run

# 2. 运行 WF（支持断点续跑）
python scripts/walk_forward_validation.py --strategy me --folds 3 --resume

# 3. 已有 fold 结果时，只做 OOS 对比
python scripts/walk_forward_validation.py --strategy me --folds 3 --oos-only
```

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--strategy` | 必填 | 策略名 (bpc/fer/me) |
| `--folds` | 3 | Fold 数量（至少 2） |
| `--seed` | 42 | 训练 seed（单 seed 加速） |
| `--resume` | false | 跳过已完成的 fold |
| `--oos-only` | false | 只做 Phase 2 |
| `--end-date` | 自动 | 最新数据日期 |

### 耗时估算

- 单 fold ≈ 1.5h（单策略, 单 seed）
- 3 folds ≈ 4.5h
- 支持 `--resume`，中断后可断点续跑

### 输出

```
results/walk_forward/{strategy}/
  fold_1/          → Fold 1 完整管线输出
  fold_2/          → Fold 2 完整管线输出
  fold_3/          → Fold 3 完整管线输出
  oos_fold_2/      → Fold 1 冻结配置 × Fold 2 预测 → OOS 回测
  oos_fold_3/      → Fold 2 冻结配置 × Fold 3 预测 → OOS 回测
  wf_summary.json  → 汇总结果
```

### 输出示例

```
══════════════════════════════════════════════════════════════════════════════
📊 Walk-Forward Validation Summary: ME
──────────────────────────────────────────────────────────────────────────────
 Fold          Holdout  Sharpe_pt   Trades  WinRate    MeanR
    1  2024-08→2025-02     0.6200      45    52.0%    1.350
    2  2025-02→2025-08     0.5500      38    50.0%    1.200
    3  2025-08→2026-02     0.5947     121    52.9%    1.402
──────────────────────────────────────────────────────────────────────────────
 Fold  Frozen  IS Sharpe  OOS Sharpe   Decay  OOS Trades
    2      F1     0.5500      0.4800     87%          38
    3      F2     0.5947      0.5100     86%         121
══════════════════════════════════════════════════════════════════════════════
  📋 判定: ✅ 参数稳健 (OOS/IS ≥ 70%)
     Decay Ratio: 87%
```

## 六、局限性

1. **计算成本高**: 每个 fold 需完整跑一遍管线
2. **交易数少**: 单 fold 可能只有 30-50 笔交易，统计置信度有限
3. **不测试模型衰减**: WF 测的是优化层的过拟合，不直接测试 LightGBM 模型本身的衰减
4. **非严格时序隔离**: 同一个 fold 内，优化和回测用同一份 holdout（IS），
   跨 fold 的冻结配置测试才是 OOS
