# trend_scalp：翻转 reseed 实验（2026-05-19）

## 问题

从对冲开局改为 `initial_legs: TREND` 后，段内 `trend_direction` 翻转时，应：

- **A**：立即按新方向 reseed（旧默认，等同段内「反向入场」），还是  
- **B**：平掉逆势腿，**等下一 regime 段**再开仓？

## 方法

| 项 | 内容 |
| --- | --- |
| 脚本 | `scripts/experiment_dual_add_flip_reseed.py` |
| 诊断 | `scripts/diagnose_dual_add_trend.py` |
| 配置 | `config/strategies/trend_scalp/research/calibrate_roll.default.yaml` |
| 窗口 | 2022-01-01 → 2026-03-31 |
| 标的 | BTC / ETH / SOL / BNB / XRP |
| 信号 | 2h；执行 1min 回放 |
| 参数 | `--no-initial-hedge`，basket TP，`regime_only`，`fee-bps 8` |

### 变体

| 变体 | flip_action | reseed_on_flip | 含义 |
| ---- | ----------- | -------------- | ---- |
| `flat_until_next_regime` | close_offside_all | false | **采纳**：翻转平仓，等同段不再 reseed |
| `reseed_on_flip_close_offside` | close_offside_all | true | 旧默认：平仓后立即按新方向 reseed |
| `keep_offside_legacy` | keep | true | 对冲时代：保留逆势腿 |

## 结果（`sum_pnl_per_capital` / 资本桶）

| 变体 | return_pct | portfolio_cum_dd | worst_segment | trades | 段胜率 |
| ---- | ----------: | ----------------: | ------------: | -----: | -----: |
| **flat_until_next_regime** | **1273.1** | **-3.88%** | **-2.50%** | 14213 | 82.5% |
| keep_offside_legacy | 1271.8 | -9.02% | -6.52% | 15048 | 82.4% |
| reseed_on_flip_close_offside | 1234.5 | -9.26% | -6.76% | 15610 | 80.6% |

原始表：`results/dual_add_flip_reseed_2022_2026/ablation_summary.csv`  
副本：`docs/strategy/trend_scalp_翻转reseed_ablation_summary.csv`

## 结论

采纳 **`reseed_on_flip: false`**：

- 净收益更高（约 +3% vs 旧默认）  
- 组合与单段 maxDD 明显更小  
- 与「regime 确认才入场」一致  

已写入 `config/strategies/trend_scalp/archetypes/execution.yaml`；宪法与管线 slug 为 **`trend_scalp`**。
