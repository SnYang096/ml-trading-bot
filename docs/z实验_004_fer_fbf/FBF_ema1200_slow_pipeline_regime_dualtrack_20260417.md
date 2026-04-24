# FBF EMA1200 慢管线与 Regime 双轨结论（2026-04-17）

## 实验目标

1. 验证在 **慢管线自动阈值** 下，`EMA1200` 方向约束是否带来正向收益。  
2. 验证 FBF 是否在“波动区/中枢区”更有效，并评估 `EMA1200` 在震荡穿越场景下是否失效。

## 实验版本与运行信息

### 阶段 A（EMA1200 + gate，不加 AB-B）

- run_id: `20260417_184110`
- 摘要: `results/fbf/slow-rolling-sim/_rolling_sim/20260417_184110/stitched_summary.json`
- 日志: `results/fbf/slow-rolling-sim/_logs/fbf_stageA_ema_gate_20260417_184109.log`

### 阶段 B（EMA1200 + AB-B 放宽 + gate）

- run_id: `20260417_191527`
- 摘要: `results/fbf/slow-rolling-sim/_rolling_sim/20260417_191527/stitched_summary.json`
- 日志: `results/fbf/slow-rolling-sim/_logs/fbf_stageB_ema_abb_loosen_gate_20260417_191527.log`

### 对照基线（生产版 FBF）

- run_id: `20260413_162634`
- 摘要: `results/fbf/slow-rolling-sim/_rolling_sim/20260413_162634/stitched_summary.json`

## 慢管线 stitched 结果

| 版本 | stitched_total_r | stitched_total_trades | 正月/负月/零月 |
|---|---:|---:|---:|
| baseline_prod | +36.7554 | 240 | 10 / 4 / 1 |
| stageA_ema_gate | +14.8628 | 187 | 8 / 6 / 2 |
| stageB_ema_abb_loosen_gate | +16.9550 | 213 | 8 / 5 / 3 |

## 关键结论（慢管线）

- `EMA1200` 方向约束在慢管线自动阈值下 **没有提供正向增益**。  
- 阶段 A/B 都显著弱于 baseline（总 R 与月度稳定性均下降）。  
- 阶段 B（AB-B 放宽）相对阶段 A 有改善（+2.09R，+26 笔），但仍远低于 baseline。

结论：按本次完整慢管线证据，`EMA1200` 方向约束应回退（去掉）。

## Regime 双口径分析

分析输出目录：`results/fbf/regime-analysis-20260417/`

- `trade_regime_enriched.csv`
- `feature_regime_summary.csv`
- `cross_regime_summary.csv`
- `regime_summary.json`

### 口径 1：特征口径（feature-based）

使用指标（可在交易时间点重建）：

- `trend_r2_20`（趋势强度）
- `atr_pct`（波动分位）
- `bb_width_norm`（波动宽度）
- `ema_pos` 绝对值（作为中枢代理）

说明：

- 由于回测产物中没有每笔 `dist_to_nearest_sr` 的直接留存，这里使用 `|ema_pos|` 作为中枢近似代理。
- 结论用于策略定位，不用于精确阈值发布。

观察：

- 3 个版本的高频桶都集中在 `chop/*` 组合，但收益在不同子桶高度分化。  
- 不是“只要震荡就好”，而是“震荡中的某些结构子区有效、某些子区无效”。  
- `EMA1200` 版本在多个 `ema_center` 子桶表现恶化，提示“中枢附近反复穿越”场景中约束噪声较大。

### 口径 2：EMA1200 穿越口径（cross-count）

定义（120 根窗口）：

- `cross_high_chop`: 穿越次数 >= 18
- `cross_mid`: 穿越次数 8~17
- `cross_low_trend`: 穿越次数 < 8

结果摘要：

- baseline: `cross_low_trend` 223 笔，+35.34R；`cross_mid` 17 笔，+1.41R
- stageA: `cross_low_trend` 181 笔，+18.38R；`cross_mid+high` 合计为负
- stageB: `cross_low_trend` 206 笔，+18.55R；`cross_mid+high` 合计为负

解读：

- FBF 的主要收益来源并非“高穿越震荡区”。  
- `EMA1200` 约束并没有把策略推进到更有利的穿越 regime，反而总体收益下降。  
- 用户提出的怀疑成立：在波动/中枢中，EMA1200 可能被来回穿越，约束价值不稳定。

## 最终决策

1. **去掉 EMA1200 方向约束**（本轮证据不支持保留）。  
2. FBF 建议保留为辅助腿，但按 regime 做条件化控制：  
   - 避开“高穿越”噪声子区；  
   - 保留对结构位与失败突破质量的约束（AB-B 思路可继续，但不要叠过严门槛）。  
3. 若进入下一轮，建议从“无 EMA1200 约束 + 温和 AB-B + 可调 gate”重新做慢管线验证。

## 建议的下一步实施

- 将 `config/strategies/fbf/archetypes/entry_filters.yaml` 回退为无 EMA1200 约束版本。  
- 保留 `entry_filters_stageA_ema_gate.yaml`、`entry_filters_stageB_ema_abb_loosen.yaml` 作为实验档案。  
- 基于本报告结论，单开一轮 “no-ema + regime-aware gate” 慢管线。

## 2026-04-18 复核（no-EMA + mild AB-B + gate）

### 运行信息

- run_id: `20260418_095724`
- pipeline: `config/prod_train_pipeline_2h_slow_fbf_noema_mildabb_gate.yaml`
- 摘要: `results/fbf/slow-rolling-sim-noema-mildabb/_rolling_sim/20260418_095724/stitched_summary.json`
- 对照基线: `results/fbf/slow-rolling-sim/_rolling_sim/20260413_162634/stitched_summary.json`

### stitched 对比

| 版本 | stitched_total_r | stitched_total_trades |
|---|---:|---:|
| baseline_prod_20260413 | +36.7554 | 240 |
| noema_mildabb_gate_20260418 | +18.8362 | 300 |

### 复核结论

1. 本轮 `no-EMA + mild AB-B + gate` 相比生产基线表现明显更弱（总 R 约为基线的 51%）。  
2. 交易笔数增加（300 vs 240），但新增交易没有转化为更高净收益，说明当前“放宽入场 + 增加条件”的组合未形成稳定 edge。  
3. 结合 2026-04-17 的 Stage A/B 与本轮复核，FBF 近期实验结论可收口为：  
   - `EMA1200` 方向约束不应作为核心入场条件；  
   - 但“mild AB-B 放宽版”也未超过现生产基线；  
   - 现阶段应以 `20260413` 基线作为主版本，避免继续扩大搜索空间。

### 建议是否继续实验

- 建议先暂停大规模 FBF 结构实验（新的慢滚全量试错），把资源让给更高潜力策略/模块。  
- 若必须继续，仅建议做 1 轮“低成本、可证伪”的定点验证（例如只测 short-side 限制或 symbol 白名单），且以“不劣于 20260413 stitched”作为硬门槛。  
- 未达到硬门槛前，不再推进 FBF 新版本进入主线。
