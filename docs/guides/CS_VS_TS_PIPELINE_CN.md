# CS vs TS：两条 pipeline 的差异、因子集合定义与评估指标取舍

本文对比：
- **TS（Time-Series）**：单资产或“多资产拼接但仍是时间序列监督学习”的研究/训练/回测流程
- **CS（Cross-Sectional）**：同一时刻跨资产的截面排序/回归/组合（long-short）流程

---

## 1) 因子/特征集合分别在哪里定义？

### TS（时间序列）
TS 的“特征集合”通常来自策略配置与 Pool‑B/feature groups：
- 策略 features：`config/strategies/*/features.yaml`（以及 `features_suggested_*.yaml` 等）
- feature groups：`config/feature_groups.yaml`（语义组/分类）
- Pool‑B：由 `mlbot ... factor-eval` 等导出 `features_pool_b*.yaml`，再用于 search/train

TS 的特征加载以 `StrategyFeatureLoader` 为中心，面向 **单个 symbol 的时间序列**，特征列会被 wrapper/contract/缓存体系约束。

### CS（截面）
CS 的“因子集合”是独立维护的 `factor_sets`（面向跨资产可比性）：
- 推荐集合：`config/cross_sectional/cs_factor_sets_crypto.yaml`
- CS 的 panel 还可通过 `mlbot cross-section catalog` 自动按列名/启发式分组生成 factor sets

**CS rank**（`mlbot cross-section rank`）本质上只需要一个“可从 FeatureStore 加载到的列名”，因此：
- 你可以直接 `--factor <col>`（FeatureStore 列名）
- 也可以用 `--factor-set-yaml/--factor-set` 校验这个列名属于某个 CS factor set（防止拼错/用错列）

---

## 2) 为什么 CS factor_eval 指标比 TS “多”？

`src/cross_sectional/scripts/factor_eval.py` 输出：
- **rank‑IC**（Spearman）：衡量“截面排序相关性”
- **quantile long/short spread**：把因子变成组合（Top vs Bottom）直接看收益差
- **turnover/fee**：截面组合每期会换仓，必须看换手与成本敏感性
- **Sharpe（gross/net）**：把收益序列汇总成可比较的风险调整指标

这些在 CS 里是“自然需要的”，原因很简单：
- CS 的最终用法通常就是 **排序→建仓→换仓**，所以“组合表现 + 成本敏感性”是一级公民。

---

## 3) TS 也需要这些指标吗？

分场景：
- **TS 单币种策略（多数现有 TS 策略）**：
  - 你的执行不是“按截面排序建组合”，而是对单币种做信号/概率→下单
  - 因此 **CS 的 long/short spread + turnover** 不一定是核心指标
  - TS 更常见的指标是：分类/回归质量（AUC、calibration）、策略回测指标（PnL/DD/胜率/交易次数）等

- **TS 多币种统一模型，但执行是“每天/每根 bar 选 Top‑K”**：
  - 这时执行本质变成“截面选股”，你就需要 CS 这一套：rank‑IC、Top‑K/L‑S、turnover、fee、Sharpe
  - 建议复用 CS 的 backtest/eval，而不是把它塞回 TS 的 factor-eval 里（保持职责清晰）

结论：**不是 TS “应该补齐 CS 指标”，而是当 TS 的执行方式变成截面组合时，应直接用 CS 的评估模块**。


