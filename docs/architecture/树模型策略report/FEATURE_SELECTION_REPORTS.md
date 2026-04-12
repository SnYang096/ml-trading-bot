# Feature Selection Reports (Unified Index)

> **目的**：把“最佳特征配置（节点/组级）”与“最佳特征列（列级 importance）”以及 **2026-01-02 rerun** 的最新 greedy 结果放到同一处，避免多份文档各自维护、口径不一致。
>
> **优先级/口径建议（读这份文档时）**：
> 1. **最新 rerun**（`docs/architecture/树模型策略report/feature_group_search_summary_20260102_rerun.md` + `features_suggested_greedy_20260102_rerun.yaml`）
> 2. **列级别建议**（`BEST_FEATURE_COLUMNS_BY_STRATEGY.md`，用于“组内裁剪/排除列/invert 候选”）
> 3. **历史 best_combo 汇总**（`BEST_FEATURE_CONFIGURATIONS.md`，用于回溯历史，不保证可比）

---

## 1) Latest rerun (2026-01-02) — Greedy multi-seed (tree models)

来源：
- 报告：`docs/architecture/树模型策略report/feature_group_search_summary_20260102_rerun.md`
- 写回 YAML：
  - `config/strategies/sr_breakout/features_suggested_greedy_20260102_rerun.yaml`
  - `config/strategies/compression_breakout/features_suggested_greedy_20260102_rerun.yaml`
  - `config/strategies/trend_following/features_suggested_greedy_20260102_rerun.yaml`

### Summary table (rerun)

| 策略 | 最终选择 groups | Sharpe_mean（baseline → final） | 推荐 YAML |
|---|---|---:|---|
| `sr_breakout` | `kline_core` | **-0.8581 → 1.6618** | `config/strategies/sr_breakout/features_suggested_greedy_20260102_rerun.yaml` |
| `compression_breakout` | `market_cap_norm`, `vpin_scene` | **-1.0175 → 2.3153** | `config/strategies/compression_breakout/features_suggested_greedy_20260102_rerun.yaml` |
| `trend_following` | `kline_core`, `trend_core` | **-1.6617 → 0.6501** | `config/strategies/trend_following/features_suggested_greedy_20260102_rerun.yaml` |
| `sr_reversal_rr_reg_long` | *(pending)* | *(pending)* | *(pending: rerun output dir exists)* |

> 注：rerun 的 `requested_features` 是 **feature nodes（`*_f`）**，不是列；一个 node 可能输出多列（见 rerun 报告的 node-vs-column 说明）。

### Overlap: “历史 best_combo（未归一化时期）” vs “归一化后 rerun”

这里按历史 best_combo run 目录对齐（同策略的 `results/feature_group_search/*best_combo*`），并与 rerun 的 writeback YAML 做 **feature node 交集**（相同 `*_f`）。

对齐口径（用于计算交集）：
- 历史 best_combo：
  - `sr_breakout`: `results/feature_group_search/sr_breakout_best_combo_v4/feature_group_search_result.json`
  - `compression_breakout`: `results/feature_group_search/compression_breakout_best_combo_v5/feature_group_search_result.json`
  - `trend_following`: `results/feature_group_search/trend_following_best_combo_v5/feature_group_search_result.json`
- 归一化后 rerun：
  - `config/strategies/*/features_suggested_greedy_20260102_rerun.yaml`

交集结果（相同 features / nodes）：
- `sr_breakout`: **∅**（无交集）
- `compression_breakout`: `atr_f`, `vpin_scene_semantic_scores_f`
- `trend_following`: `atr_f`, `rsi_f`

> 解释：交集为 ∅ 往往意味着两次 run 的 **base feature pool / groups 候选空间**差异很大（例如历史 best_combo 允许/使用了更多语义 block，而 rerun 从最小 base + 少量 groups 出发，greedy 很早停止）。

---

## 2) Column-level recommendations (importance / exclude / invert candidates)

来源：
- `docs/architecture/树模型策略report/BEST_FEATURE_COLUMNS_BY_STRATEGY.md`（2026-01-01）

这份内容适合做：
- **组内裁剪**（未来：`exclude_columns`）
- **invert_candidates**（先做候选，再与最终 requested_features 求交集得到 invert_features）
- **定位“某个 node 输出的哪几列有用/有害”**

### High-signal takeaways (from column report)

- `sr_reversal_rr_reg_long`
  - 强正向列：`sqs_hal_high`（来自 `sqs_hal_high_f`）
  - 强负面列：`trend_r2_50`（建议排除或 invert）
  - DTW 类在当时的 expanded 结果里整体偏负面（建议谨慎）
- `sr_breakout`
  - 强正向列：`turnover_over_mcap`（来自 `market_cap_normalized_orderflow_f`）
  - 语义列：`vpin_ignition_score` 贡献显著（当时口径）
- `trend_following`
  - 语义列：`fp_ignition_score`、`turnover_over_mcap` 在当时口径下提升较大

> **重要提醒**：列级建议来自 **expanded/v4/v5 等历史目录**，与 2026-01-02 rerun **不一定 apples-to-apples**（数据切片、labels、base pool、groups 口径都可能不同）。用法建议：把它当作“组内诊断线索”，不要把数值当作最终结论。

---

## 3) Historical “best configurations” (for reference / archaeology)

来源：
- `docs/architecture/树模型策略report/BEST_FEATURE_CONFIGURATIONS.md`（更新时间显示 2024-12-30）

用途：
- 用于回看历史 best_combo（multisymbol / quick / v4 / v5 等）的“当时最佳组合”
- 不建议直接拿来和最新 rerun 做数值对比（很多 run 配置、数据口径不同）

---

## 4) Recommended maintenance: single source of truth

为了让“明天早上看到稳定闭环”更可控，建议把未来的“对外结论”只沉淀在：
- `docs/architecture/树模型策略report/feature_group_search_summary_YYYYMMDD_*.md`（每次 rerun 产出一份）
- `config/strategies/<strategy>/features_suggested_greedy_YYYYMMDD_*.yaml`（可直接用于训练/回测）

而：
- `BEST_FEATURE_CONFIGURATIONS.md`：标注为历史/不再更新
- `BEST_FEATURE_COLUMNS_BY_STRATEGY.md`：标注为列级诊断/不保证口径一致

