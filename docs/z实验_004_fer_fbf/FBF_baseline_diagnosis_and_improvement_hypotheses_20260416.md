# FBF 基线诊断与改进假设（2026-04-16）

## 结论先行

- 当前 `FBF` 是 **可持续盈利但上限偏低** 的辅助策略，不是坏策略。
- 它的问题不是完全跑不动，而是 **止损占比偏高、收益分布偏二元、做空侧拖后腿**。
- 从现有结果看，`FBF` 仍有改进余地，但更像是**微结构与执行优化**，不太像能被改造成主线暴利策略。

## 本文对应基线

- pipeline: `config/prod_train_pipeline_2h_slow_fbf_only.yaml`
- strategy root: `config/strategies/fbf`
- baseline run: `results/fbf/slow-rolling-sim/_rolling_sim/20260413_162634`
- stitched summary: `results/fbf/slow-rolling-sim/_rolling_sim/20260413_162634/stitched_summary.json`

## 当前基线结果

### stitched 总览

- `stitched_total_r`: `+36.7554`
- `stitched_total_trades`: `240`
- 覆盖月份: `2023-09` 至 `2024-12`，共 `16` 个月
- 单笔平均: `+0.153R`

### 交易分布

- 胜率: `103 / 240 = 42.9%`
- 平均盈利: `+1.792R`
- 平均亏损: `-1.079R`
- 出场原因:
  - `sl`: `137` 笔，占 `57.1%`
  - `tp`: `102` 笔，占 `42.5%`
  - `timeout`: `1` 笔

这说明当前 FBF 基本是一个 **固定 1R 止损 / 2R 止盈** 的二元分布系统:

- 对了就吃 `~2R`
- 错了就吃 `~-1R`
- 因为胜率高于 2:1 RR 的盈亏平衡线（33.3%），所以整体仍然赚钱
- 但单笔 edge 不厚，因此总收益不高

## 关于“止损非常快速”的复核

从 `event_backtest_fbf.json` 的 `bars_held` 统计看，当前结论要更精确一些:

- 全部交易平均持有: `788.3` bars
- 全部交易中位持有: `396` bars
- 止损交易平均持有: `677.6` bars
- 止损交易中位持有: `357` bars
- 真正“很快止损”的 `SL (<100 bars)` 只有 `15 / 137`

因此更准确的说法不是“多数单子止损非常快”，而是:

- **止损比例确实偏高**
- 但多数止损不是秒死，而是反打修复没有走成，持有一段时间后回落到止损
- 用户感受到“容易被洗”，更像是因为 **收益很容易回撤掉**，而不是所有交易都瞬间被打脸

## 当前最明显的问题

### 1. SHORT 侧为净拖累

- LONG: `203` 笔，`+42.27R`，胜率 `44.8%`
- SHORT: `37` 笔，`-5.52R`，胜率 `32.4%`

做空侧胜率已经低于 2:1 RR 的盈亏平衡线，因此是明确负贡献。

这意味着:

- FBF 的现版本更像 **long-biased failed-breakout repair**
- 或者至少说明当前 short 过滤不够强

### 2. 某些 symbol 明显不适配

- `BNBUSDT`: `+27.47R`
- `XRPUSDT`: `+13.80R`
- `BTCUSDT`: `+4.38R`
- `ETHUSDT`: `-10.28R`

说明 FBF 的 edge 并不是均匀分布在所有币上。
当前版本已经像是在吃某些更“箱体化 / 回吐化”品种的结构，而在部分更趋势化标的上持续失效。

### 3. 没有利润保护

当前执行配置:

- `breakeven.enabled: false`
- `trailing.enabled: false`
- `take_profit.target_r: 2.0`

这意味着交易一旦没有走到 TP，就更容易把中间浮盈吐回去。

因此用户观察到的“止损多、赚钱不够稳”是合理的。

### 4. 入场仍然偏右侧

当前正式 `entry_filters.yaml` 只要求:

- `fer_range_pos_20` 靠近区间边缘
- `bars_since_local_high/low` 足够近
- `fer_sr_failed_breakout_score >= 0.38`

但它 **没有显式要求价格必须仍然贴近最近 SR**。

相比之下，备选版本 `entry_filters_ab_B_cvd_confirm.yaml` 多了:

- `dist_to_nearest_sr` 约束
- `fer_ols_pos` 区间位置约束
- `rsi` 与 `cvd_divergence_score_pct` 确认

因此“入场偏右侧、不够贴 SR”的怀疑是有配置依据的。

## funnel 侧观察

16 个月总 funnel 汇总:

- `total_signals_checked`: `31986`
- `reject_prefilter_deny`: `31453`
- `signals_generated`: `307`

对应:

- prefilter 通过率只有 `1.7%`
- 最终 signal 生成率约 `0.96%`

说明当前 FBF 已经是一个 **非常稀疏、强过滤** 的系统。

这带来两个含义:

- 优点: 噪声已经被压得很低，系统不是因为“乱开仓”而坏
- 风险: 再继续无脑收紧，可能会把仅有的 edge 一起砍掉

所以后续实验应优先做:

- 执行保护
- 做空过滤
- 更贴 SR 的 entry 约束

而不是先继续大幅加严 prefilter

## 当前最可信的改进假设

### H1. breakeven 可以改善收益质量

因为当前系统没有任何利润保护，若交易先走到一定浮盈再回撤，仍可能回吐到 `-1R`。
开启 `breakeven` 后，预期:

- 总体 `SL` 占比可能不明显下降
- 但 `avg_loss`、回撤与收益波动有机会改善

### H2. 禁做空或强化 short filter 有望直接提升 stitched

SHORT 当前是负贡献，因此:

- `long_only`
- 或对 short 侧增加更强确认

都属于优先级很高的实验方向。

### H3. 把 entry 拉回 SR 附近，可能改善“右侧追入”

优先候选不是重写整个 archetype，而是先实验:

- `entry_filters_ab_B_cvd_confirm.yaml`
- 或把其中 `dist_to_nearest_sr` / `fer_ols_pos` 中最关键的条件引入正式 entry

### H4. gate draft 可能有增益

当前正式 `archetypes/gate.yaml` 为空，但 `config/strategies/fbf/gate_draft.yaml` 已有 3 条统计 deny 规则。

如果这些规则在 rolling 下稳定，可能能进一步过滤失败样本。

## 本轮实验优先级

建议按下面顺序跑:

1. `breakeven`
2. `long_only` 或强化 short filter
3. promote `gate_draft`
4. 切换到更贴 SR 的 entry 版本
5. 再决定是否做 prefilter 微调

## 一句话结论

**FBF 现在不是“彻底没救”，而是“有正期望，但 edge 偏薄；最值得优化的是执行保护、short 过滤和 SR 邻近入场，而不是指望它长成趋势主策略”。**

## FBF 冻结决议（2026-04-18）

与 `docs/z实验_005_统一研究/FBF_ema1200_slow_pipeline_regime_dualtrack_20260417.md` 中的收口结论对齐：

- **主版本（stitched 验收口径）**：继续以本文基线为准 — run `20260413_162634`（`stitched_total_r` **+36.7554**，**240** 笔；见 `results/fbf/slow-rolling-sim/_rolling_sim/20260413_162634/stitched_summary.json`）。
- **近期结构实验**：`EMA1200` 方向入场（Stage A/B）与 **no-EMA + mild AB-B + gate**（run `20260418_095724`，`stitched_total_r` **+18.8362**，**300** 笔）均未超过上述基线；**不再推进为新的默认配置**。
- **上文「本轮实验优先级」**：在出现新的硬需求或明确假设前 **暂停按该顺序批量开实验**；若仅做定点验证，须以「不慢于 `20260413` stitched」为门槛。

诊断性判断（short 拖累、SR 邻近、执行保护等）仍可作为**未来小步假设**，但不改变当前冻结线。
