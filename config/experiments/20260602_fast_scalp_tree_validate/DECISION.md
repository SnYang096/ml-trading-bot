# fast_scalp 树模型验证 — 决策（跑完后填写）

| 字段 | 值 |
|------|-----|
| 实验 | `20260602_fast_scalp_tree_validate` |
| 训练流程 | [`TRAINING.md`](TRAINING.md) |
| Promote 门禁 | [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) |

## TL;DR（recent_6m_oos = 唯一干净 OOS 段）

| 方案 | recent_6m_oos | 判定 |
|------|--------------|------|
| **G3 = H=3 signed short-only（现有基线）** | **+12.02%** | ✅ **OOS 赢家，保留** |
| G16 = g5-label + adverse gate | +3.61% | gate 有效但 ranker 无 edge，不及 G3 |
| G7 = 双 head | −3.75% | ❌ 过拟合，证伪 |
| G14 = g5-label 裸 ranker | −12.91% | ❌ 无 edge，证伪 |

**结论：两条新假设（双 head、execution-aligned g5-label ranker）均被 OOS 证伪；现有 H=3 短单基线（G3）反而是最好的，OOS +12.02%。**
唯一值得继续的副产品是 **adverse gate 方法学**（IC-prune 选特征 + 真实 MAE，OOS 把烂 ranker 从 −12.91% 救到 +3.61%）——应在 G3 这类**本身有正 edge 的 ranker** 上叠加复测，而非用在无 edge 的 g5-label 上。

## 轨道 A — 双 head（G7 vs G3）

net return（注入分数后，6 币；G3=H=3 signed short-only，G7=双 head agreement；head train_end=2025-10-01）：

| segment | G3 (H=3 short) | G7 (双 head) | 窗口 | 结论 |
|---------|----|----|------|------|
| bear_2022 | +11.18% (4276) | +12.49% (1250) | 训练内 | G7 ≈ G3，换手 ~1/3 |
| bull_2023_2024 | — | +21.70% (1744) | 训练内 | G7 样本内强 |
| recent_range_to_bear | — | +19.06% (1282) | 训练内 | G7 样本内强 |
| recent_6m_oos | **+12.02%** (2160) | **−3.75%** (607) | **真 OOS** | **G3 完胜 G7** |

**关键发现：双 head 过拟合，且 OOS 劣于 H=3 baseline。**
- G7 样本内三段（bear/bull/range，<2025-10）非常漂亮：+12.49% / +21.70% / +19.06%，换手仅 H=3 short 的 ~1/3。
- **但真 OOS 的 recent_6m_oos（2025-10+）：G7 = −3.75%，而 G3（H=3 signed short-only）= +12.02%。**
- 即：双 head 样本内的高选择性/低换手优势在 OOS 完全瓦解，且**显著劣于现有简单 H=3 短单基线**。
- （G3 bull/range 样本内段未跑完——短单极密集，事件 30min+/段；OOS 决策不依赖它们，已停跑。）

**Promote 双 head？** **否，证伪。** 双 head 在唯一干净 OOS 段为负且远逊于 H=3 short baseline。
**现有 H=3 signed short-only（G3）才是 OOS 赢家（recent_6m +12.02%），应保留为基线。**

## 轨道 B — exec-aligned + gate（G14/G16 vs G5）— **证伪**

pooled-6 execution-aligned `g5-label` (realized `r_long − r_short`，G5 tight SL/TP) entry ranker
事件回测（net return，6 币，τ=q0.30：long −0.213 / short −0.309）：

| segment | G14 (g5-label, 无 gate) | **G16 (g5-label + 新 gate)** | gate 增益 | 窗口 |
|---------|----|-----|------|------|
| bear_2022 | −51.67% (1442) | **+12.64%** (801) | +64pp | gate 训练内 |
| bull_2023_2024 | −61.50% (1897) | **+0.04%** (1218) | +62pp | gate 训练内 |
| recent_range_to_bear | −54.87% (1425) | **−17.66%** (970) | +37pp | gate 训练内 |
| recent_6m_oos | −12.91% (719) | **+3.61%** (475) | **+16pp** | **gate 真 OOS** |

**关键发现（修正了首轮"证伪"的草率结论）：**
- **g5-label ranker 本身无 edge**：四段全负 + vector τ-scan 全段负 Sharpe（无 plateau，最佳 q=0.30 Sharpe −0.67）。
- **但 adverse gate 真正有效**：把四段全负的 ranker 在**真 OOS 的 recent_6m 翻为 +3.61%**，bear +12.64%，bull 打平；
  仅 range 段仍 −17.66%。gate train_end=2025-10-01，recent_6m_oos=2025-10+ 对 gate 与 ranker 均为 OOS。
- bear/bull/range 三段在 gate 训练窗口内（样本内增益不足为据）；唯一干净的 OOS 证据是 recent_6m **+3.61%**。

**Gate selected features（IC-prune + 真实 MAE lift，8 个，非写死）：**
`vol_accel, me_accel_5k, evt_scale, evt_scale_left, me_accel_5k_long, evt_var_99, evt_var_99_left, spectrum_cvd_low_freq_ratio`
（`adverse_avoided=0.145`：被否决交易比放行真实 adverse 率高 14.5pp；见 `track_b/gate/ic_prune_v2/train_summary.json`）

- G5 baseline 快照无注入分数（stale），事件 0 trades，不构成对照（已知问题，待补 H=3 注入）。

**Promote？** **g5-label ranker 不 promote**（无独立 edge）。**gate 方法学值得保留并进一步验证**：
单段 OOS 正收益不足以 promote，需在一个**本身有正 edge 的 ranker**（如 H=3 / 双 head）上叠加该 gate 复测，
确认 gate 在正 ranker 上是净增益而非偶然。

## 过程 bug 与防过拟合

- Bug 清单：[`../20260530_fast_scalp_alts_majors/TREE_BUG_AUDIT.md`](../20260530_fast_scalp_alts_majors/TREE_BUG_AUDIT.md)
- 特征 / 过拟合：[`FEATURES_AND_OVERFITTING.md`](FEATURES_AND_OVERFITTING.md)
