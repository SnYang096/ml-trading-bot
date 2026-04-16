# FBF 定位评估（是否保留、是否仅作特定 regime 辅策略）

## 结论

- **FBF 不建议删除**，但也**不应继续作为主线主策略**。
- 更合理的定位是：**保留为辅助策略 / 次级策略**，在组合里提供一条与趋势延续类不同的收益腿。
- `fbf_exp_fatter_tp`（`TP=3R, time_stop=48`）**不如基线 `fbf`**，建议归档到 `config/strategies/bad-candidates/`。

## 对比对象

- 基线 FBF：`results/fbf/slow-rolling-sim/_rolling_sim/20260413_162634/stitched_summary.json`
- 肥 TP 实验：`results/fbf/slow-rolling-sim-exp-fatter-tp/_rolling_sim/20260416_153251/stitched_summary.json`
- 趋势对照：
  - BPC：`results/bpc/slow-rolling-sim/_rolling_sim/20260413_144115/stitched_summary.json`
  - TPC：`results/tpc/slow-rolling-sim/_rolling_sim/20260416_104125/stitched_summary.json`
  - ME：`results/me/slow-rolling-sim/_rolling_sim/20260416_090720/stitched_summary.json`

## stitched 总览

| 策略 | 月数 | stitched_total_r | stitched_total_trades | 单笔 R |
|------|------|------------------|-----------------------|--------|
| FBF | 16 | +36.7554 | 240 | 0.153 |
| FBF exp fatter TP | 16 | +32.7287 | 272 | 0.120 |
| BPC | 16 | +754.9468 | 171 | 4.415 |
| TPC | 27 | +1420.9170 | 357 | 3.980 |
| ME | 27 | +384.1505 | 432 | 0.889 |

## 对 FBF 的解释

### 1. FBF 不是“没信号”，而是“每单不肥”

- 基线 FBF 16 个月共 **240 笔**，并不稀疏。
- 但总收益只有 **+36.8R**，单笔平均仅 **0.153R**。
- 与 BPC / TPC 相比，FBF 的问题不是密度不足，而是**吃不到大趋势 fat tail**，每笔 payoff 明显更弱。

### 2. FBF 不是只在极少数月份偶发有效

按 `results/fbf/slow-rolling-sim/_rolling_sim/20260413_162634/fast_month_*/fbf/event_backtest_fbf.json`
统计：

- **正收益月份**：10
- **负收益月份**：4
- **0 交易月份**：1

典型正月：

- `2024-02`: +9.2209R / 30 trades
- `2024-09`: +9.6312R / 5 trades
- `2024-10`: +9.0804R / 20 trades

典型负月：

- `2023-12`: -3.1757R / 3 trades
- `2024-12`: -1.7190R / 38 trades

这说明 FBF 更像：

- **大多数月份都能贡献一点收益**
- 但**很少出现趋势类那种爆发性大月**

因此它不像“强 regime 开关策略”，更像“弱互补、低爆发、较平滑”的辅助腿。

### 3. `fbf_exp_fatter_tp` 不成立

与基线相比：

- 总 R：`+32.73R < +36.76R`
- 交易数：`272 > 240`
- 单笔 R：`0.120 < 0.153`

说明把 FBF 从 `TP=2R` 拉到 `TP=3R`、延长 `time_stop`，
**没有换来更好的 fat tail 捕获，反而稀释了单笔质量**。

这与 FBF 的产品语义一致：

- 它更像抓 **failed breakout 后的回吐 / 修复段**
- 而不是抓后续大趋势扩张

## 最终建议

### 保留什么

- 保留 `config/strategies/fbf` 及 `config/prod_train_pipeline_2h_slow_fbf_only.yaml`
- 作为**辅助 / 次级策略**保留

### 归档什么

- `fbf_exp_fatter_tp` 归入 `config/strategies/bad-candidates/fbf_exp_fatter_tp`
- 不建议继续作为主版本候选

### 组合中的定位

- **主线赚钱腿**：TPC / BPC / ME
- **辅助补充腿**：FBF

FBF 的意义在于：

- 语义上提供“失败突破反打”这一条与趋势延续不同的腿
- 在部分假突破 / 回吐月份提供一定补充

但从目前结果看，它的互补性**不强到足以升级为主线**。

