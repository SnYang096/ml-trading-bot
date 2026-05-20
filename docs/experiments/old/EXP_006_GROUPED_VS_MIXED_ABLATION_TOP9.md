## EXP_006: Grouped vs Mixed Ablation（Top9 / 4H / OOS=2025-05~2025-10）

目标：对比 **Grouped 训练（HighCap9-only）** vs **Mixed 训练（mixed baseline）** 在同一套：
- OOS 数据窗口（`2025-05-01 ~ 2025-10-31`，每币 ~1099 根 4H bars）
- FeatureStore layer（`features_83f12ecc5e`）
- Router tuned 阈值（来自 `EXP_006_RULE_THRESHOLD_TUNING_TOP9`）
- returns_source=`rr_execution`（含 vbt fee/slippage）

输出：Rule router 的 OOS counterfactual 指标（Sharpe/DD/return/turnover）+ per-symbol 诊断。

---

## 1) 对比结论（先看核心指标）

> 结论：在这次 “Top9 + 2025-05~10 + tuned router” 的口径下，**HighCap9-only（grouped）显著优于 mixed baseline**（rule_sharpe_mean 从负转正）。

| 项 | Grouped（HighCap9-only） | Mixed baseline |
|---|---:|---:|
| `test_symbols` | 9 | 9 |
| `test_steps` | 2970 | 2970 |
| `rule_sharpe_mean` | **0.1715** | **-0.2728** |
| `rule_avg_max_dd` | 0.1175 | 0.1323 |
| `rule_avg_total_return` | -0.0010 | -0.0193 |
| `rule_avg_switch_rate` | 0.2695 | 0.3026 |
| `rule_avg_mode_entropy` | 0.8206 | 0.8746 |

> 注：这里 `rule_*` 指 **rule router action** 的表现；`pred_*` 是 BC 预测 action 的表现（本实验的重点是 rule side）。

---

## 2) Router 行为（mode 分布对比）

- Grouped mode counts：`{"NO_TRADE": 6984, "MEAN": 1659, "TREND": 1248}`
- Mixed   mode counts：`{"NO_TRADE": 6613, "MEAN": 1524, "TREND": 1754}`

观察：Grouped 更“保守”（NO_TRADE 更多、TREND 更少），但在该窗口下反而带来更好的 rule Sharpe。

---

## 3) Per-symbol 诊断（重点 ETH）

两边都输出了 counterfactual 的 `per_symbol.csv`。这里先看 `rule_sharpe`（OOS test split）：

| symbol | Grouped rule_sharpe | Mixed rule_sharpe |
|---|---:|---:|
| BTCUSDT | **3.3701** | 0.3936 |
| ETHUSDT | **-2.7035** | -0.3782 |
| SOLUSDT | **2.2173** | -2.1928 |
| BNBUSDT | 0.0257 | 1.7937 |
| XRPUSDT | 1.6045 | 1.9994 |
| ADAUSDT | -3.1589 | -2.7455 |
| AVAXUSDT | 0.3162 | -2.0333 |
| LINKUSDT | -1.0394 | 1.0381 |
| DOTUSDT | 0.9117 | -0.3301 |

要点：
- **Grouped 的收益主要由 BTC + SOL 支撑**，但 **ETH 仍是明显拖累**（更差）。
- Mixed baseline 对 ETH 更“温和”，但整体 rule_sharpe_mean 反而更差（更多 TREND 参与导致尾部/回撤更重）。

这也解释了为什么后续的低维护优先项仍然是：`ETH gating v1`（而不是立刻做 per-symbol 阈值）。

---

## 4) 复现实验：命令与产物路径

### 4.1 Grouped（HighCap9-only）OOS 全链路（已跑完）

Run dir：
- `results/exp006_group_ablation/highcap9_only_rerun3/e2e_top9_oos/`

关键产物：
- preds：`.../preds_oos/preds_<SYM>.parquet`
- mode：`.../mode_3action_tuned.parquet`
- logs：`.../logs_3action_tuned.parquet`
- e2e：`.../e2e_tuned/{shadow,counterfactual}/`

执行日志：
- `results/logs/exp006_ablation_highcap9_only_rerun3_oos_e2e_top9.log`

### 4.2 Mixed baseline OOS 全链路（为公平对比，重跑在“数据修复后”的完整窗口）

Run dir：
- `results/exp006_nnmh_highcap6_best_top10/e2e_top9_oos_fixed/`

执行日志：
- `results/logs/exp006_ablation_mixed_baseline_oos_e2e_top9_fixed.log`

---

## 5) 下一步建议（最小动作）

1) 先在该口径下，把 **ETH gating v1** 接进 rule router（shadow→online gate），验证是否能把组合 Sharpe/DD 拉上去。  
2) 如果 gating 不够，再考虑 “per-symbol rr profile” 或 “tree gate → 导出规则”。

