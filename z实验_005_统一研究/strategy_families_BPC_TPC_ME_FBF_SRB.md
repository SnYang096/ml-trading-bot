# 策略家族对照：BPC / TPC / ME / FBF / SRB

本文固定 **产品语义与边界**，避免与 `meta.yaml` 单行描述漂移；实现以各策略目录下 archetype 为准。

## 对照表

| 家族 | 核心触发语义 | 与相邻家族的边界 |
|------|----------------|------------------|
| **BPC** | Donchian / 突破链 + EMA1200 带通下的回踩–延续 | **必须先有「突破语义」**（如 `bpc_breakout_direction`、`bpc_score_breakout`），再谈回踩与延续；见 [config/strategies/bpc/archetypes/direction.yaml](../config/strategies/bpc/archetypes/direction.yaml)。 |
| **TPC** | 趋势已确立时的 **回踩后再延续** | **不要求** Donchian 突破门控；以 EMA1200 等趋势为先验，用 `tpc_soft_phase_f` 等识别回踩/恢复；见 [config/strategies/tpc/meta.yaml](../config/strategies/tpc/meta.yaml)。 |
| **ME** | 动量扩散、流动性加速 | 偏 **动能与波动放大**，不以「关键位假破/真破」为主叙事；见 [config/strategies/me/meta.yaml](../config/strategies/me/meta.yaml)。 |
| **FBF** | 关键位 **失败突破** 后的区间回复 | 与 SRB **事件极性相反**（假破 vs 真破）。慢滚基线见 [FBF_slow_rolling_baseline_20260413.md](FBF_slow_rolling_baseline_20260413.md)；Git tag：`fbf-slow-baseline-20260413-36R-240t`。 |
| **SRB** | 关键位 **成功突破** 后的结构延续（Structural Range Breakout） | 相对 **TPC**：TPC 是「已在趋势里等回调」；SRB 强调 **边界被决定性打破** 这一事件。相对 **BPC**：BPC 是 Donchian+回踩链；SRB 更贴 **SR/结构位 + 确认**（MVP 可用频谱/SR 强度等占位特征，后续可加专用 score）。实现目录：[config/strategies/srb](../config/strategies/srb)。 |

## SRB 命名与管线

- **代号**：SRB（Structural Range Breakout）。
- **慢滚管线**：[config/prod_train_pipeline_2h_slow_srb_only.yaml](../config/prod_train_pipeline_2h_slow_srb_only.yaml)  
- **结果目录**：`results/srb/slow-rolling-sim/`（与 TPC/FBF 并列）。

## 实验：FBF 单笔更「肥」（只抬止盈 R）

- **目的**：在 **不改小止损**（`initial_r` 保持 1.0）前提下提高 `take_profit.target_r`，观察总 R / 胜率 / 回撤相对基线的变化。
- **配置**：独立策略根 `config/strategies/fbf_exp_fatter_tp/`（仅 execution 与基线不同：`target_r: 3.0`，`time_stop_bars: 48`）；流水线 [config/prod_train_pipeline_2h_slow_fbf_only_exp_fatter_tp.yaml](../config/prod_train_pipeline_2h_slow_fbf_only_exp_fatter_tp.yaml)。
- **对比基线**（跑完 stitch 后）：

```bash
# 基线
git show fbf-slow-baseline-20260413-36R-240t:results/fbf/slow-rolling-sim/_rolling_sim/20260413_162634/stitched_summary.json
# 或本地若仍保留：
cat results/fbf/slow-rolling-sim/_rolling_sim/<baseline_run_id>/stitched_summary.json

# 肥 TP 实验
cat results/fbf/slow-rolling-sim-exp-fatter-tp/_rolling_sim/<run_id>/stitched_summary.json
```

```bash
mlbot pipeline run --all --config config/prod_train_pipeline_2h_slow_fbf_only_exp_fatter_tp.yaml --stage rolling_sim
```

## 宪法 / PCM（SRB）

已在 [config/constitution/constitution.yaml](../config/constitution/constitution.yaml) 注册 **SRB**：`enabled_archetypes`、`per_strategy_limits.srb`（不加仓、`max_add_times: 1`）、`intent_selection_policy.archetype_priority` 中插在 **tpc 与 me 之间**（`bpc, tpc, srb, me, fer`）。FBF 仍为研究单策略管线，未写入宪法白名单。

## SRB 冒烟验证

- **配置校验（推荐）**：`mlbot pipeline run --config config/prod_train_pipeline_2h_slow_srb_only.yaml --strategy srb --stage prefilter --dry-run`（exit 0，已验证）。
- **实跑 prefilter（较长）**：同上去掉 `--dry-run`，可加 `--skip-shap`；产物在 `results/srb/slow-rolling-sim/srb/<timestamp>/`。

## MVP 风险（SRB）

首版无专用「成功突破分数」时，信号可能与 **TPC** 部分重叠；后续迭代方向：专用标签/特征或收紧 prefilter 语义。
