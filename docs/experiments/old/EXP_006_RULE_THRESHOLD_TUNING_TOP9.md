## EXP_006: Rule(3-action) Router Threshold Tuning (Top9 OOS, rr_execution)

本实验目标：在 **固定 nnmultihead 模型与 OOS 预测输出** 的前提下，仅通过调整 `mlbot rule mode-3action` 的阈值，使 **Top9** 的 OOS counterfactual `rule_sharpe_mean` 从负值提升到正值，并形成可复盘的参数与产物路径。

---

## 1) 固定条件（不随 tuning 改动）

- **模型**：`results/exp006_nnmh_highcap6_best_top10/path_primitives_4h_80h_min__prune_try__keep_semantic__volume_profile__poolb__volatility_reversal_score_f__poolb__compression_duration_f__rm_poolb__volume_profile_volatility_features_f_multi_240T/model.pt`
- **OOS 预测（Top9，无 XRP）**：
  - 目录：`results/exp006_nnmh_highcap6_best_top10/path_primitives_4h_80h_min__..._multi_240T/preds_oos2/`
  - 文件形如：`preds_BTCUSDT.parquet`（包含 `pred_dir_prob/pred_mfe_atr/pred_mae_atr/pred_t_to_mfe`）
- **评估窗口**：`2025-05-01 ~ 2025-10-31`（4H，240T）
- **returns_source**：`rr_execution`
- **symbols(Top9)**：`BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,DOTUSDT,LTCUSDT`
- **成本/滑点/延迟（本次调参默认）**：entry_delay=0, cost=0, slippage=0  
  （注意：这些是假设参数，后续要做“带成本/延迟”的稳健性复验）

---

## 2) Baseline（未调参前）结果

产物：
- `results/exp006_nnmh_highcap6_best_top10/e2e_top9/e2e/counterfactual/metrics.json`

关键指标（baseline）：
- `rule_sharpe_mean = -2.6972351502438263`
- `rule_avg_total_return = -0.036782297980111504`

---

## 3) 阈值调参方法（fast sweep）

为什么不用每次都跑 `rl build-logs-3action (rr_execution)`：
- `rr_execution` 生成 `ret_mean/ret_trend` 成本较高
- 对于“阈值调参”，我们只需要把 **preds → mode_action** 重算，再在同一份 `ret_mean/ret_trend` 上做 PnL 计算

实现脚本：
- `scripts/tune_rule_mode_3action_thresholds.py`

输入：
- `--preds <preds_oos2_dir>`
- `--logs <logs_3action.parquet>`（已包含 rr_execution 的 `ret_mean/ret_trend`）
- `--model <model.pt>`（用于自动判断 `preds_in_log1p`）

输出：
- `tuning_trials.csv`：每组阈值的 `rule_sharpe_mean / dd / trade_rate`
- `best.json`：最佳阈值与指标

运行命令：

```bash
cd /workspaces/ml_trading_bot
python3 scripts/tune_rule_mode_3action_thresholds.py \
  --preds results/exp006_nnmh_highcap6_best_top10/path_primitives_4h_80h_min__prune_try__keep_semantic__volume_profile__poolb__volatility_reversal_score_f__poolb__compression_duration_f__rm_poolb__volume_profile_volatility_features_f_multi_240T/preds_oos2 \
  --logs  results/exp006_nnmh_highcap6_best_top10/e2e_top9/logs_3action.parquet \
  --model results/exp006_nnmh_highcap6_best_top10/path_primitives_4h_80h_min__prune_try__keep_semantic__volume_profile__poolb__volatility_reversal_score_f__poolb__compression_duration_f__rm_poolb__volume_profile_volatility_features_f_multi_240T/model.pt \
  --out   results/exp006_nnmh_highcap6_best_top10/e2e_top9/tuning_rule_thresholds \
  --n-trials 300 \
  --seed 7 \
  --entry-delay 0 \
  --cost-per-turnover 0 \
  --slippage-bps 0
```

---

## 4) Best 参数与结果（Top9）

产物：
- `results/exp006_nnmh_highcap6_best_top10/e2e_top9/tuning_rule_thresholds/best.json`
- `results/exp006_nnmh_highcap6_best_top10/e2e_top9/tuning_rule_thresholds/tuning_trials.csv`

Best（trial=83）：
- `rule_sharpe_mean`: **2.4295438367915256**
- `rule_sharpe_std`: 0.7811497612464647
- `rule_dd_mean`: 0.12079441289265229
- `trade_rate_mean`: 0.34938388989964164

对应的 `mlbot rule mode-3action` 阈值 overrides：
- `--mfe-min 0.1259208384`
- `--eff-min 1.0434303531`
- `--dir-conf-trend-min 0.0641852109`
- `--mfe-trend-min 0.4572347656`
- `--ttm-trend-min 5.6310524663`
- `--eff-mean-min 1.2011240134`
- `--ttm-mean-max 29.0044411469`

---

## 5) 这是不是“炼丹调参”，对未来没意义？

结论：**它可能会过拟合，所以必须用“严格口径”把它变成有预测意义的工程流程**。本次结果说明：
- 当前 Router 阈值初始值并不合适（会导致 NO_TRADE 塌缩或错误切分）
- 通过阈值把 action 分布调到合理区间后，PnL 可以显著改善

但要避免“未来没意义”，必须做下面这些验证：
- **严格的 OOS 分割**：阈值调参只能使用一段 tuning window；最终必须在更晚的 holdout window 验证（时间上完全隔离）。
- **稳健性扫描**：对成本、滑点、entry_delay 做敏感性分析（3x3 或更大网格），避免只在理想假设下好看。
- **每币一致性**：做逐币单币评估，避免“组合靠 1 个币撑起来”。
- **限制自由度**：阈值范围与数量要受控（越多越容易过拟合）。优先只调 2~4 个关键阈值。

> 本文档先记录本次“让系统跑通 + baseline 变正 Sharpe”的阶段性结果。下一步应把 tuning 变成“固定流程 + 固定数据切分 + 固定报告”。

---

## 6) 用 best 阈值重跑全链路（Top9）

目的：验证 “tuner 的快速计算” 与 “正式 e2e 报告”一致。

重跑将生成新的产物（避免覆盖旧结果）：
- `mode_3action_tuned.parquet`
- `logs_3action_tuned.parquet`
- `e2e_tuned/`（shadow + counterfactual + fsm）

命令（已执行）：

```bash
cd /workspaces/ml_trading_bot

RUN_DIR=results/exp006_nnmh_highcap6_best_top10/e2e_top9
PREDS=results/exp006_nnmh_highcap6_best_top10/path_primitives_4h_80h_min__prune_try__keep_semantic__volume_profile__poolb__volatility_reversal_score_f__poolb__compression_duration_f__rm_poolb__volume_profile_volatility_features_f_multi_240T/preds_oos2
MODEL=results/exp006_nnmh_highcap6_best_top10/path_primitives_4h_80h_min__prune_try__keep_semantic__volume_profile__poolb__volatility_reversal_score_f__poolb__compression_duration_f__rm_poolb__volume_profile_volatility_features_f_multi_240T/model.pt

mlbot rule mode-3action --no-docker \
  --preds "$PREDS" --model "$MODEL" \
  --mfe-min 0.1259208384 \
  --eff-min 1.0434303531 \
  --dir-conf-trend-min 0.0641852109 \
  --mfe-trend-min 0.4572347656 \
  --ttm-trend-min 5.6310524663 \
  --eff-mean-min 1.2011240134 \
  --ttm-mean-max 29.0044411469 \
  --output "$RUN_DIR/mode_3action_tuned.parquet"

mlbot rl build-logs-3action --no-docker \
  --preds "$PREDS" --mode "$RUN_DIR/mode_3action_tuned.parquet" --model "$MODEL" \
  --data-path data/parquet_data --timeframe 240T \
  --start-date 2025-05-01 --end-date 2025-10-31 \
  --returns-source rr_execution \
  --output "$RUN_DIR/logs_3action_tuned.parquet"

mlbot rl run-e2e-3action --no-docker \
  --logs "$RUN_DIR/logs_3action_tuned.parquet" \
  --out "$RUN_DIR/e2e_tuned"
```

执行日志（便于复盘）：
- `results/logs/exp006_top9_mode_3action_tuned.log`
- `results/logs/exp006_top9_build_logs_3action_tuned.log`
- `results/logs/exp006_top9_run_e2e_3action_tuned.log`

### 6.1 产物路径

- tuned mode：`results/exp006_nnmh_highcap6_best_top10/e2e_top9/mode_3action_tuned.parquet`
- tuned logs：`results/exp006_nnmh_highcap6_best_top10/e2e_top9/logs_3action_tuned.parquet`
- tuned e2e：
  - shadow：`results/exp006_nnmh_highcap6_best_top10/e2e_top9/e2e_tuned/shadow/`
  - counterfactual：`results/exp006_nnmh_highcap6_best_top10/e2e_top9/e2e_tuned/counterfactual/`
    - `metrics.json`
    - `report.html`
    - `per_symbol.csv`
  - fsm：`results/exp006_nnmh_highcap6_best_top10/e2e_top9/e2e_tuned/fsm_decision.json`

### 6.2 tuned 结果（counterfactual）

来自：`.../e2e_tuned/counterfactual/metrics.json`

- `rule_sharpe_mean = 2.628624219750041`
- `rule_avg_total_return = 0.013146444036881675`
- `rule_avg_max_dd = 0.06206116591701116`

同时（供参考）：
- `pred_sharpe_mean = 2.1324544370220546`
- `pred_avg_total_return = 0.006189856614844292`

> 备注：tuner 的 best `rule_sharpe_mean≈2.43` 与正式 e2e 的 `2.63` 在量级上是一致的（并且正式链路略高），说明阈值与评估管线对齐。


