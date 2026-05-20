# 00 — 为什么做 Wave 3 + 原始计划

## 症状：BPC 在 2024-04~06 的退化

从 [baseline_vs_early_good_reference.md](baseline_vs_early_good_reference.md) 的月度矩阵（截取关键月）：

| 月份 | 早期好配置 (20260413) | Wave 1+2 当前 | delta R |
|---|---|---|---|
| 2024-04 | 20 笔 / +120.5R | 12 笔 / −35.8R | **−156.3** |
| 2024-05 | 20 笔 / +20.6R  | 10 笔 / −42.7R | **−63.3**  |
| 2024-06 | 12 笔 / −8.9R   | 8 笔 / +11.3R  | +20.2      |
| **trend_favorable 合计** | **52 笔 / +132.2R** | **30 笔 / −67.2R** | **−199.4** |

同时 Wave 1+2 在 2025-11~12 死月有改善（从 0 笔 → 17 笔 / +83.4R），但代价是趋势月大幅退化。

## 两个候选根因方向

**方向 1 — meta-algo 选规则错位**
- Prefilter/Gate/EF 的 label 都是 `is_good = forward_rr >= -0.8`
- 这个 label 学的是 "避开极端负 RR"，不是 "BPC 三段结构是否成立"
- 假说：meta-algo 会选 `ema_1200_position > 0.1 deny` 这类把 BPC 主战场（趋势中 pullback）砍掉的规则
- 证据线索：Wave 2 之前 Gate 确实选过这条规则

**方向 2 — execution / Prefilter 阈值不对**
- 2024-04 mean RR = −3R/笔 → SL 可能过紧
- Prefilter 3 条 locked 规则（pullback_score / breakout / recovery）阈值或在急涨环境下过松
- 样本内过拟合到 2023 数据，2024 新 regime 不适应

Wave 3 押在**方向 1**，设计了 5 步 label 改造方案验证。

## Wave 3 原始 5 步计划

### Step 0 — 建立 baseline
- 用当前 Wave 1+2 代码跑 6 个关键月 fast_month：2024-03/04/05/06 + 2025-11/12
- 产出锚点 timestamps（见 [baseline_bpc_wave2_runs.txt](baseline_bpc_wave2_runs.txt)）
- 建工具 [scripts/compare_monthly_pnl.py](../../scripts/compare_monthly_pnl.py) 做月度对比

### Step 1 — Wave 3-A：Prefilter label 改为 `rr >= 0`
- 假设：Prefilter 用 "成功 = 正 R"，meta-algo 会选出真能带来 alpha 的特征
- 期望：trend_favorable delta ≥ +50R

### Step 2 — Wave 3-B：Gate label 改为 tail-only（bottom 5%）
- 假设：Gate 用 "deny 极端不利场景"，不和 Prefilter 语义重叠
- 期望：trend_favorable delta ≥ +50R

### Step 3 — Wave 3-C：EF label 改为 `rr >= +0.3`（可选）
- 假设：EF 用 "高质量入场 = 显著正 R"
- 期望：补充改善

### Step 4 — 5 策略交叉验证
- 确认 Step 1~3 的改进对 ME/TPC/FBF/SRB 也成立（或至少不退化）

### Step 5 —（长期）Nested CV
- 给 meta-algo 加 purged K-fold 防过拟合

## 成功指标（写在 plan 里的硬规则）

对每一步（Step 1/2/3）用 [compare_monthly_pnl.py](../../scripts/compare_monthly_pnl.py) 产出月度对比，**bucket 化裁决**：

| bucket | 月份 | PASS 条件 |
|---|---|---|
| `trend_favorable` | 2024-04, 2024-05, 2024-06 | new R 不低于 baseline −20R **且** 绝对值 ≥ −100R |
| `death_months` | 2025-11, 2025-12 | 至少 1 月 n_trades > 0 |
| `small_sample` | 2024-01, 2024-03 | 仅诊断，不阻塞 |

任一 bucket FAIL → 本 step 打回，不进入下一步。

## 结果速览（详情见 [01](01_experiments_and_findings.md)）

| Step | 结果 | 说明 |
|---|---|---|
| 0 | ✅ DONE | baseline 锚点建立，工具可用 |
| 1 | ⚠️ NO-OP | label 改动未生效（根因：scoring method 不消费 label） |
| 2 | ❌ FAIL | trend_favorable −96.2R（根因：tail-only 让 Gate 变松） |
| 3 | 取消 | Step 2 已证伪前提 |
| 4 | 取消 | 前置步骤未通过 |
| 5 | 取消 | 见 [02](02_meta_findings_on_meta_algo.md)：K-fold 不适合金融生产 |

**方向 1（label 语义）证伪 → 转向方向 2（execution / Prefilter 阈值）**。Wave 3 正式结束。
