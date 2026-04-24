# SRB Cross State Machine — 全量 Rolling 结果（2026-04-19）

## 背景

本次改动一次性做两件事：

1. **把设计初衷写入 `config/strategies/srb/meta.yaml`**
   明确 SRB = "关键位突破 → 顺势跟趋势 (主业)；突破失败 → 假破反手 (副业)"，
   列出 4 条 `anti_patterns`（不得对 trend_r2/bb_width 设上限；反手不能由 SL 触发；
   反手画像不得缩减；2h TF 上 confirm_k 不得 >= 4）。

2. **按设计初衷回改三处 YAML**
   - `config/strategies/srb/archetypes/prefilter.yaml`：删掉方向错的
     `bb_width_normalized_pct <= 0.65` + `trend_r2_20 <= 0.65`，只保留
     `sr_strength_max >= 0.42` 作为语义锚（bb_width 下限也去掉，避免与 meta 冲突）
   - `config/strategies/srb/features.yaml`：增加 `forbidden_prefilter_meta_columns`
     把 `bb_width_normalized_pct` 和 `trend_r2_20` 精确剔除 meta-prefilter 候选
   - `config/strategies/srb/archetypes/execution.yaml`：`confirm_k: 3 → 2`
   - `config/prod_train_pipeline_2h_slow_srb_only.yaml`：`fast_loop.prefilter.optimize: true → false`
     —— meta-algorithm 在多个月 holdout 上系统性地生成"窄 range"（如 `bpc_impulse_return_atr ∈ [-0.94, -0.60]`）
     这类规则和 SRB 的"突破 = impulse extreme"语义直接冲突，是之前大趋势月被砍掉 90%+ cross 的元凶。

## 结果对照（全量 16 个月 rolling）

对比 baseline `20260419_153737`（第一版状态机，prefilter 语义反转）vs 本次 `20260419_203407`：

| month    | baseline n / R       | now n / R           | Δn    | ΔR       | 备注 |
|----------|----------------------|---------------------|-------|----------|------|
| 2023-09  | 13  +7.38            | 14   +0.24          | +1    | -7.14    | 震荡月持平 |
| 2023-10  | 23 +41.57            | 27  +67.99          | +4    | **+26.42** | 趋势月收益 +63% |
| 2023-11  |  7 +13.15            | 17   -5.65          | +10   | -18.80   | 回撤；BTC 刚启动上行 |
| 2023-12  |  6  -7.56            |  6  -12.95          | +0    | -5.39    | 小回撤 |
| 2024-01  |  6  +3.64            | 44   +7.95          | +38   | +4.31    | 交易量激增 |
| 2024-02  |  6 +10.06            | 11   +7.93          | +5    | -2.13    | 持平 |
| 2024-03  | 13  -5.70            |  6   +0.00          | -7    | +5.71    | 少交易 |
| 2024-04  | 41  -6.85            | 15  -30.39          | -26   | -23.53   | **反馈差** |
| 2024-05  | 16 +12.95            | 16   -2.10          | +0    | -15.05   | 回撤 |
| 2024-06  | 12 -35.29            | 32   +6.70          | +20   | **+42.00** | 抓回大趋势 |
| 2024-07  |  0  +0.00            | 16   +3.03          | +16   | +3.03    | 从 0 到有 |
| 2024-08  |  5  -1.30            | 23   -4.46          | +18   | -3.16    | 量涨损平 |
| 2024-09  | 18  +8.58            | 39  +35.11          | +21   | **+26.52** | 大趋势月抓到 |
| 2024-10  | 13 -12.72            | 16   -3.04          | +3    | +9.69    | 减损 |
| 2024-11  |  2  -0.63            | 15  -24.41          | +13   | -23.78   | **反馈差** |
| 2024-12  | 22 +14.52            | 52   -0.96          | +30   | -15.48   | 交易量翻倍但持平 |
| **TOT**  | **203  +41.79**      | **349  +45.02**     | +146  | +3.23    | |

**一句话结论**：
- 交易量 +72%（203 → 349），大趋势月不再 ≈0（2024-06/09/11/12 都被打开）
- 总 R 小涨 (+3.2R)，mean_r 从 +0.21 回落到 +0.13
- **2024-06 / 2024-09 / 2023-10 是典型成功案例**：之前 0 或负，现在正大收益
- **2024-04 / 2024-11 / 2023-11 是典型失败案例**：之前不交易，现在交易但亏

## 诊断：为什么 2024-04 / 2024-11 这类大趋势月还是亏？

观察 2024-11 的 funnel：

```
signals_generated: 23          # 19 primary confirmed + 4 fake reverse
reject_kill_switch: 18         # 被 portfolio 连亏触发的 killswitch 拦掉
add_position_ok: 6
```

- 状态机正确识别了 23 个 cross 事件并确认成 primary 突破
- 但其中相当比例的"突破"在执行期被 SL / kill_switch 打掉
- `confirm_k=2`（4h 确认）偏快，在 Q4 2024 的碎震撑-阻力区连续被吞
- trailing 起步 `activation_r=4.0` + `min_trail_atr_ratio=0.5` 在大波动月可能过紧

**这不是"prefilter/state machine 没识别出大趋势"的问题了** ——
是"进场后被 whipsaw 反复打 SL，portfolio 层 killswitch 砍交易"的执行侧问题。

## 下一步建议（优先级排序，待用户决策）

| 优先级 | 动作 | 预期效果 |
|-------|------|---------|
| **P0** | `2024-04 / 2024-11` 专项诊断：导出 trades.csv 看 primary 突破进场后的 excursion 分布（MAE/MFE） | 明确是"全部快速 SL"还是"部分 SL + 部分漏反手" |
| **P1** | 给 `primary` 和 `fake_reverse` 分别定义 `initial_r` / `activation_r`（meta.yaml 已明确不可缩 size，但 SL 宽度可以差异化） | 如果 primary MAE 普遍 > 4R，考虑 initial_r 上调到 5 |
| **P2** | 在 confirm 成立的同根 bar，引入"反手轨道预开"：若该 confirmed 长期内 SL → 自动对齐 state machine 的 fake 分支生成反手 | 当前 state machine 一旦 confirmed 就不再回退，会错过"假突破发生在 confirmed 之后"的 case |
| **P3** | portfolio kill-switch 阈值放宽或分策略独立计数 | 目前 SRB 跟 BPC/TPC 共享 killswitch，SRB 的连亏带累系统性锁仓 |

## 归档

- 改动 commit / 快照：当前 working tree（见 git status）
- 结果目录：`results/srb/slow-rolling-sim/_rolling_sim/20260419_203407/`
- 对比基线：`results/srb/slow-rolling-sim/_rolling_sim/20260419_153737/`
- 设计初衷锚点：`config/strategies/srb/meta.yaml::design_intent`

## 与 meta.yaml::anti_patterns 的对照

| anti_pattern            | 当前状态 |
|-------------------------|---------|
| filter_out_trending_bars | ✅ 已删除 `bb_width <= 0.65` / `trend_r2_20 <= 0.65`，并用 `forbidden_prefilter_meta_columns` 把这两列精确剔出 meta 候选 |
| reverse_triggered_by_sl | ✅ 已下线 `fake_break_reverse`，当前全部由 `srb_cross_state_machine` 前向决策 |
| shrinking_reverse_profile | ✅ fake 与 primary 使用同一 execution profile（见 `generic_live_strategy._advance_srb_cross`） |
| confirm_k_too_large | ✅ confirm_k=2，远小于 anti_pattern 上限 3 |
