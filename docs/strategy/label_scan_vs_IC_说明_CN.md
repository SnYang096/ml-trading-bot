# Label Scan 与 IC 的区别（TPC R&D 用）

> 配套文档：[`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md)（Step 0 流程）、[`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md)（架构层）、[`时序过滤与横截面排序_为何先TPC再Rank_CN.md`](时序过滤与横截面排序_为何先TPC再Rank_CN.md)（先过滤再 rank、小资金组合、调仓频率与对冲 §8–§10）。
>
> IC baseline 数值见：`config/monitoring/factor_ic_baseline_tpc_20260526.json`、
> `results/factor_health/ic_baseline_20260526/report.md`（本地，results 不入 git）。

---

## 0. 术语：什么是 1pp（1 个百分点）

在本文和 `quick_layer_scan` 报告里，**pp = percentage point（百分点）**，表示**成功率本身的差值**，不是相对变化率。

| 说法 | 含义 | 例子 |
|---|---|---|
| **+1.01pp** | 成功率从 A% 升到 B%，**绝对差** = B − A = 1.01 | 56.48% → 57.49%，差 **+1.01pp** |
| **+1%（相对）** | 在旧成功率上乘 1.01 | 56.48% × 1.01 ≈ 57.04%（**不是**我们用的口径） |

**为什么要单独说 pp？**

- 胜率在 50% 附近时，+1pp 就是「每 100 笔多 1 笔好单」；
- 写成「+1% 相对提升」会夸大或缩小（56%→57% 相对只涨约 1.8%，但绝对是 +1pp）。

`quick_layer_scan` 的 **Δpp vs base** 列 = `succ_in − base_success`，单位都是**百分点**。

---

## 1. 数据从哪来

每一根 2h K 线（在 `features_labeled.parquet` 里）大致有：

```text
每一根 bar
    │
    ├─ 特征 x（如 ema_1200_position、vol_persistence、tpc_pullback_depth）
    │
    └─ 若 TPC 形态在该 bar 可评估 → 有 label
           success_no_rr_extreme  → 这单算不算「好」（0/1）
           forward_rr             → 这单实际赚了多少 R（连续，可正可负）
```

- **Label scan** 主要用 `success_no_rr_extreme`。
- **IC** 主要用 `forward_rr`（与 ML4T 里 factor → forward return 一致）。

样本多半是 **archetype 能触发的 bar**（稀疏），不是全市场每一根 K 线。

---

## 2. Label Scan（`quick_layer_scan.py`）

### 在问什么

在满足某条件时，**「好单」比例** 比不满足时高多少？（或比全体 base 高多少？）

### 例子（F' 条件，20260526）

在 `tpc_semantic_chop ≤ 0.4` 的 bar 上：

| 条件 | n | success 率 | Δpp vs base |
|---|---:|---:|---:|
| 全体（base） | 23194 | 56.48% | — |
| F'：`\|ema\|>0.10` 且 `\|slope\|>0.002` | 6937 | 57.49% | **+1.01pp** |
| 仅 H：`\|ema\|>0.10` | 8556 | 56.53% | +0.06pp |

**+1.01pp** = 57.49% − 56.48% = 多 **1.01 个百分点** 的好单占比；|z|≈2 表示这个差在统计上不太像纯噪声。

### 特点

| 维度 | 说明 |
|---|---|
| 目标 | 离散好坏（0/1） |
| 贴合 | yaml 规则：满足 → allow / deny |
| 样本 | 与 prefilter/gate 同一类「能走到决策链」的 bar |
| 局限 | **不直接回答**「多赚多少 R」「会不会挡掉少数大 R 单」 |

### 典型用法（TPC Step 0）

```bash
python scripts/quick_layer_scan.py condition-set \
  --features-parquet results/<train_final>/tpc/features_labeled.parquet \
  --label success_no_rr_extreme \
  --filter "tpc_semantic_chop<=0.4" \
  --condition "F': abs(ema_1200_position)>0.10 AND abs(ema_1200_slope_10)>0.002" \
  --out results/tpc/quick_scan/regime_candidates_<日期>.md
```

**判读**：Δpp ≥ +0.5pp 且 |z|>2 → 值得拉 `event_backtest`；|z|<2 → 当噪声，不必烧回测。

---

## 3. IC / ICIR（Information Coefficient）

### 在问什么

特征数值与 **未来收益 `forward_rr`** 之间，有没有**稳定的秩相关**（Spearman rank IC）？

- **IC** = 一段时期内，特征排名与收益排名的相关程度（可正可负）。
- **ICIR** = 各月 IC 的 `mean(IC) / std(IC)` → 信号是否**月月同向**（不是单月偶然）。

### 例子（`ema_1200_position` vs `forward_rr`）

| 时段 | rank IC | 粗读 |
|---|---:|---|
| 2024 bull | **+0.027*** | 位置越高，平均 R 略高 |
| 2025–2026 recent | **-0.076*** | 位置越高，平均 R 反而更低 |

→ **同一特征，两段市场相关符号相反**（符号翻转）。这是「单一 regime 配置无法两段都最优」的连续维度证据。

### 特点

| 维度 | 说明 |
|---|---|
| 目标 | 连续 R |
| 贴合 | 因子模型（按分数排序、做多高分做空低分） |
| 对规则系统 | **间接**：IC 好 ≠ gate 一定该开；IC 差也可能挡掉大 R 单 |
| 强项 | 跨时段漂移、符号翻转、W 级健康监控 |

### 典型用法

- 离线 baseline：`scripts/_factor_ic_baseline_oneshot.py` 或读 `config/monitoring/factor_ic_baseline_tpc_20260526.json`
- 全量重算（慢、易踩 DAG）：`make ts-factor-eval` / `mlbot analyze factor-eval`（需完整 feature pipeline）
- 计划：并入 `quick_layer_scan.py --mode ic-decay`（读已有 parquet，与 label scan 同一入口）

---

## 4. 对照总表

| | Label scan | IC / ICIR |
|---|---|---|
| **目标变量** | success（好/坏，0/1） | forward_rr（赚多少 R） |
| **核心输出** | Δpp、\|z\| | rank_IC、ICIR、IC 正号月占比 |
| **最适合** | 设计 gate / regime 的 allow·deny | 特征是否还有预测力、是否漂移 |
| **TPC Step** | R&D Step 0（改 yaml 前） | W 级监控 + 解释 backtest 与 label 分歧 |
| **时间成本** | 约 1–2 分钟（parquet） | 同 parquet 约 5 秒；全 pipeline 可数十分钟 |

---

## 5. 为什么 TPC 两个都要看？

TPC 是 **规则链**，不是单一打分模型：

```text
regime → prefilter → direction → gate → entry → 成交 → R
```

- **Label scan**：这条规则在「能评估的 bar」里，筛得对不对（**胜率**）。
- **IC**：这个特征与「最终 R」在整体上怎么相关（**收益**，含被后面层挡掉的 bar）。

两者会**分歧**——本次实验已验证：

| 话题 | Label scan 说什么 | IC 说什么 | Backtest 印证 |
|---|---|---|---|
| 关 vol gate（B） | bear 子样本上略复杂 | recent：vp/vla **+IC** → 高 vol 对应更高 R | B recent +60R |
| H（bull 开 vol） | ema_bull 上保留 vp 略提 success | 同上；H 是为 **DD** 不是为加 R | H 2024 DD -7.6% vs B -13.5% |
| F'（加 slope） | F' **+1.01pp** succ | recent：ema **负 IC** | F' 2024 bull 已弱于 H（totR +13 vs +16，DD -11% vs -7.6%） |
| `tpc_cvd_absorption` entry | 未单独强调 | IC ≈ 0，全段不显著 | E 换 entry 方向合理 |

**固定原则**（见方法论 Step 0）：

1. Label scan 筛假设（快、贴规则）
2. 有信号 → `event_backtest` 双段（真 R）
3. 上线后 → IC baseline + `regime_watchdog`（漂移、符号翻转）

**禁止**：只看 label 就 promote；只看 IC 就改 gate（IC 不区分「被挡掉的大 R」）。

---

## 6. 实验变体 F' 在做什么（等 backtest 时参考）

| 层 | H（已 commit 到 config/live） | F'（实验，`config_experiments/Fp_ema_plus_slope_strategies/`） |
|---|---|---|
| **regime** | `\|ema_1200_position\| ≥ 0.10` | 同上 **且** `\|ema_1200_slope_10\| ≥ 0.002` |
| **gate** | vol 仅在 `ema_1200_position > 0.10` 时 deny；chop 照旧 | 与 H 相同 |
| prefilter / direction / entry | 不变 | 不变 |

**直觉**：H =「价格离慢均线够远」；F' = 再加「慢均线本身在斜着走」，过滤横盘假趋势。

**双段 event_backtest 结果（已完成，gate=H，仅 regime 不同）**：

| 时段 | 变体 | trades | totR | maxDD |
|---|---|---:|---:|---:|
| 2024 bull | H | 168 | +16.30 | -7.57% |
| 2024 bull | F' | 151 | +13.17 | -11.31% |
| 2025–2026 recent | H | 172 | +47.06 | -7.48% |
| 2025–2026 recent | F' | 102 | +39.14 | -7.58% |

→ **结论：不 promote F'**。Label 上 F' +1.01pp，但 IC 在 recent 段 ema/slope 为负相关，且两段 totR 均弱于 H（recent 少约 8R、笔数少 70）。维持已上线的 **variant H**。

---

## 7. Panel IC 与单币时序 IC 的区别

两种算法都是 `corr_rank(特征, forward_return)`，但 **「在哪些样本上、按什么维度切分」** 不同。

### 7.1 单币时序 IC（本仓库 parquet baseline 用的）

**做法**：把某一标的（或已拼好的多标的行）在 **整段历史上所有 bar** 混在一起，算一次 Spearman IC。

```text
样本：BTC 2024-01 … 2026-04 上每一根可评估 bar（可含多币行，但不做「同一时刻横截面」）
计算：IC = corr_rank( ema_1200_position[], forward_rr[] )   # 一条长向量
```

| 维度 | 说明 |
|---|---|
| 时间结构 | 前后 bar **混在同一池子里**；BTC 2024 牛市与 2025 熊市 bar 一起算 |
| 截面结构 | **不**要求「同一 timestamp 上多币可比」 |
| 回答的问题 | 「在这个策略、这段历史上，特征高是否倾向于 R 高？」（**时序 +  regime 混合**） |
| 本仓库 | `_factor_ic_baseline_oneshot.py` / `factor_ic_baseline_tpc_20260526.json` |
| 注意 | TPC 的 parquet 常是 **多币行拼表**（107k 行），仍是「全样本 corr」，**不是** Panel IC |

**月度 ICIR**：把样本按 **日历月** 分组，每月算一个 IC，再 `mean(IC)/std(IC)` → 看信号是否月月同向（与横截面无关）。

### 7.2 Panel IC（拼 panel、经典因子横截面）

**做法**：先 **stack** 多标的成 panel `(symbol, datetime)`，再在每个 **时间点 t** 上，只用「t 这一刻」所有币的截面算 IC。

```text
时刻 t：
  BTC   f=0.12   r=+0.5
  ETH   f=0.08   r=+0.2
  SOL   f=-0.05  r=-0.3
  …
  → IC_t = corr_rank( f_i at t , r_i at t )     # 只有 6～N 个点，一截面

时间序列：IC_1, IC_2, …, IC_T
汇总：IC_mean = mean(IC_t)，ICIR = mean(IC_t) / std(IC_t)
```

| 维度 | 说明 |
|---|---|
| 时间结构 | **每个 t 一个 IC**；牛市、熊市分月体现在 IC 序列上 |
| 截面结构 | **同一 bar 上多币排序**；因子高 = 相对更强 |
| 回答的问题 | 「今天买因子值高的币、卖低的，有没有相对超额？」（**纯横截面**） |
| 典型用途 | 股票多因子、crypto 多币同周期 rank |
| 与 TPC live | 你们是 **每币独立决策链**，不是「同一 t 选最强 2 个币」→ Panel IC **不是主指标** |

### 7.3 对照示意

```text
单币时序 IC（全样本）:
  t1_BTC  t2_BTC  t3_ETH  t4_SOL  …  →  一个大 corr  →  一个 IC 数字

Panel IC（按时刻）:
  t1: corr(BTC,ETH,SOL,…) → IC_t1
  t2: corr(BTC,ETH,SOL,…) → IC_t2
  …
  → IC 序列 → IC_mean / ICIR
```

### 7.4 何时用哪一种（本仓库建议）

| 场景 | 建议 |
|---|---|
| TPC 规则 R&D、解释 H/B/F' | **单币时序 IC**（或分 `cal_2024_bull` / `cal_recent` 桶，见 baseline json） |
| 多币同 bar 选强弱、因子组合 | **Panel IC**（需按 timestamp groupby 重算，当前脚本未默认实现） |
| 树模型特征入池（Pool B） | 可先单币/分币 IC 筛特征，再 OOS；若做「全市场一个模型」再考虑 Panel |
| W 级监控 | 与 R&D 同口径即可：**分桶时序 IC** + 月度 ICIR，不必强行 Panel |

**不要混读**：parquet 里虽有 BTC+ETH+… 多行，baseline 的「全段 IC = -0.002」是 **混在一起的全样本 corr**，不能当成「横截面因子 IC=0.002」。

---

## 8. 规则时序 vs 树模型时序：IC 有没有用？

### 8.1 经典用法（因子 + 横截面）

同一时刻 t：多标的各有因子值与 forward return → **横截面 IC** → 做多高分、做空低分。收益来自 **相对排序**。

这与 TPC **单币、规则链** 的决策形态不同。

### 8.2 规则时序（当前 TPC live）

| 用法 | 有没有用 | 说明 |
|---|---|---|
| 横截面多币排序 | ❌ 非主路径 | 6 币各跑一条链，不是同一 t 比分数选币 |
| 定 gate/regime 阈值 | ⚠️ 间接 | 阈值用 **label scan + event_backtest**；IC 不直接给 0.10 vs 0.12 |
| 特征是否还有方向性 | ✅ | ema bull **+IC**、recent **-IC** → 解释 H/B、否决 F' |
| W 级健康监控 | ✅ | rolling IC 符号翻转、\|IC\| 塌陷 → 触发 R&D |
| 与 label scan 交叉验证 | ✅ | label +1pp 但 IC 反号 → 勿只看 label promote |

> **规则系统**：IC 管「特征还活不活、方向有没有翻」，不管「今天买哪只币」。

**主 R&D 指标**：label scan → 双段 event_backtest → 上线后 IC baseline + `regime_watchdog`。

### 8.3 树模型时序（LightGBM / `tree_strategies`）

| 阶段 | IC 的作用 |
|---|---|
| 进模型前（特征池） | ✅ 筛对 forward return 有秩相关的列；`ts-factor-eval` / Pool B |
| 模型输出 | ✅ **IC(predict_proba, forward_rr)**、IC@H decay，常比单看 AUC 贴交易 |
| 与 SHAP | SHAP = 模型内贡献；IC = 边际秩相关；doctrine：**SHAP audit，不 auto-promote** |
| 上线后 | ✅ 预测分 IC 按月塌陷 → 树失效往往早于 Sharpe 变差 |

**树时序 ≠ 自动等于横截面**：可训「每币一条序列」，也可训 panel；只有 **按 t 做截面 IC** 才是 Panel IC（§7.2）。

**本仓库现状**：TPC **live 的 `decide()` 不走 `predict_proba`**；树主要在研究管线 / `tree_strategies`。若未来「树只做 entry、regime/gate 仍规则」：规则层 = label + backtest；树层 = IC + OOS pred IC + backtest。

### 8.4 三种范式对照

```text
                    横截面因子模型          规则时序 (TPC)           树模型时序
                    ─────────────          ─────────────           ───────────
决策单位            多币同一时刻排序        单币 bar 上 Y/N 规则      score → 阈值/仓位
IC 主战场            ★★★ Panel IC          ★ 时序 IC（辅助）         ★★ 特征+模型 IC
选特征/阈值          IC + 组合              label scan + backtest    IC 初筛 + SHAP + WF
本仓库当前            非主路径               ★ 当前 TPC 主路径         研究/树策略，未进 TPC live
```

---

## 9. 与 ML4T 的对应关系

| ML4T 概念 | 本仓库实现 |
|---|---|
| Factor evaluation (IC, IC decay) | `factor_ts_eval.py` / 计划 `quick_layer_scan ic-decay` |
| Label / meta-labeling | `success_no_rr_extreme` + label scan |
| Walk-forward / purged CV | `event_backtest` 双段（2024 bull + recent）；完整 CPCV 仍缺 |
| Feature selection via importance | SHAP（slow 管线，**audit only**，不 auto-promote） |

Rule-based archetype（TPC）的 **主 R&D 指标** = label scan + event_backtest R；IC = **辅助 + 运维**（时序 IC + 分桶），不是选 gate 阈值的唯一依据，也**不替代** Panel 因子选股流程。
