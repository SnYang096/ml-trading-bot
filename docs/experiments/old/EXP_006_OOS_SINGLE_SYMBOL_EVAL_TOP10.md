## EXP_006: OOS 单币诊断（Top10，Rule 3-action tuned）

目标：把 Top10 OOS 组合结果拆成逐币诊断，回答：
- 哪些币对组合表现贡献/拖累最大？
- 是否需要 per-symbol profile（执行参数）、per-symbol 阈值，或 gating（直接禁用/降仓）？
- 数据覆盖是否足够（避免“少样本高 Sharpe”的错觉）？

---

## 0) 固定条件与产物路径

- **OOS window**：`2025-05-01 ~ 2025-10-31`（4H, 240T）
- **symbols (Top10)**：`BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,DOTUSDT,LTCUSDT`
- **preds（Top10）**：  
  `results/exp006_nnmh_highcap6_best_top10/.../preds_oos_top10_retry2/`
- **tuned mode**：  
  `results/exp006_nnmh_highcap6_best_top10/e2e_top10/mode_3action_tuned.parquet`
- **logs（用于 e2e）**：  
  `results/exp006_nnmh_highcap6_best_top10/e2e_top10/logs_3action_tuned_retry2.parquet`
- **e2e（shadow + counterfactual）**：  
  `results/exp006_nnmh_highcap6_best_top10/e2e_top10/e2e_tuned_retry2/`
  - counterfactual metrics：`.../counterfactual/metrics.json`
  - per-symbol 指标：`.../counterfactual/per_symbol.csv`

补充：为了方便二次分析，本次把逐币汇总另存为：
- `results/exp006_nnmh_highcap6_best_top10/e2e_top10/single_symbol_eval_rule_summary.csv`

---

## 1) Top10 组合总览（counterfactual）

来自：`.../e2e_tuned_retry2/counterfactual/metrics.json`

- `rule_sharpe_mean = 2.5911451157910683`
- `rule_avg_total_return = 0.020381638536100054`
- `rule_avg_max_dd = 0.06311621363258191`

---

## 2) 数据覆盖检查（非常关键）

理想情况下，6 个月的 4H bar 数应接近 ~1100（每个 symbol）。
但本次 logs 覆盖差异很大：部分币只有 180~372 行，这会导致：
- 单币 Sharpe/Sortino 极不稳定（少样本噪声被放大）
- “贡献最大/拖累最大”的结论可能被误导（不能据此加仓）

### 2.1 log_rows（每币可用 bar 数）

（来自 `logs_3action_tuned_retry2.parquet`）

- `ETHUSDT/BTCUSDT/XRPUSDT`：1099（接近完整 6 个月）
- `DOTUSDT`：558
- `ADAUSDT`：186
- `SOLUSDT/LTCUSDT`：180
- `AVAXUSDT/LINKUSDT/BNBUSDT`：372/366/366

> 结论：当前 Top10 评估里，有多枚币属于“严重缺数据/缺 bars”的状态。后续做“最终上线”前，必须先把数据覆盖修到一致，否则会出现虚高/虚低。

---

## 3) 单币结果（Rule 3-action tuned）

数据源：
- `.../e2e_tuned_retry2/counterfactual/per_symbol.csv`
- `.../mode_3action_tuned.parquet`（mode 分布）

### 3.1 拖累最大的币（rule_sharpe < 0）

- `ETHUSDT`：`rule_sharpe=-1.75`，`rule_max_dd=0.135`（样本量充足，是真拖累，优先处理）
- `AVAXUSDT`：`rule_sharpe=-1.61`（样本量低，需先补数据再判断）
- `LINKUSDT`：`rule_sharpe=-0.69`（样本量低）
- `BNBUSDT`：`rule_sharpe=-0.64`（样本量低）

### 3.2 贡献最大的币（rule_sharpe 高）

注意：`SOLUSDT/ADAUSDT/LTCUSDT` 的 `rule_sharpe` 很高，但 test 样本量极低（~55），**不建议据此判断真实可交易性**。

- `SOLUSDT`：`rule_sharpe=11.63`（低样本，暂不作为结论）
- `ADAUSDT`：`rule_sharpe=8.87`（低样本，暂不作为结论）
- `LTCUSDT`：`rule_sharpe=5.78`（低样本，暂不作为结论）
- `XRPUSDT`：`rule_sharpe=2.25`（样本量充足，可作为稳健贡献候选）
- `DOTUSDT`：`rule_sharpe=1.83`（样本量中等偏低，需补数据）
- `BTCUSDT`：`rule_sharpe=0.23`（中性）

---

## 4) 需要 per-symbol profile/阈值/gating 吗？

### 4.1 优先级建议（从低维护到高维护）

1) **先做数据覆盖修复**（最高优先级）
   - 让 Top10 在同一窗口内都有接近 6 个月的 bars
   - 否则 per-symbol 阈值/执行参数调参很可能是“对噪声调参”

2) **gating（低维护、强收益/风险比）**
   - **低样本/缺数据**的 symbol：建议直接 NO_TRADE（或从 Top10 组合里暂时剔除），直到数据补齐。
   - 对 `ETHUSDT`（样本充分但表现差）：建议优先做 **gating**：
     - 更严格的 tradeability 条件（提高 `mfe_min/eff_min` 或加冷却/波动门）
     - 如果 ETH 在某段 regime 明显 bleed，优先加 detector/gate（例如 chop/vol/流动性过滤）

3) **per-symbol profile（中维护）**
   - 适用于：某些币的执行微观结构不同（波动、滑点、回撤特性）
   - 在本 repo 里可以通过 rr_execution 的 profile overrides 来做（例如 `max_holding_bars/take_profit_r`）
   - 建议先只对“样本足够且确实异常”的币做（当前最优先：`ETHUSDT`）

4) **per-symbol 阈值（高维护，慎用）**
   - 这是最后手段，因为会引入“多币多套参数”的维护地狱
   - 只有在你已经做完：
     - 数据覆盖一致
     - 通用 gating 生效
     - profile 仍不足以解决
     才考虑少量 per-symbol 阈值

---

## 5) 下一步行动清单（建议）

1) 先解决数据覆盖：让 Top10 在 `2025-05~2025-10` 都有 ~1100 bars  
2) 复跑 Top10：同一套 tuned 阈值下，重新输出 per_symbol.csv  
3) 针对 `ETHUSDT`：
   - 先做 gating（减少不该参与的段）
   - 再做 profile（执行参数差异化）
4) 最后才考虑 per-symbol 阈值（尽量避免）


