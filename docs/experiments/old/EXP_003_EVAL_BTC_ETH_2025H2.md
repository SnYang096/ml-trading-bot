### EXP_003 — NN Multihead（Path Primitives）BTC+ETH OOS 回测（2025H2，4H）

本报告用于回答“模型是否做了回测、Sharpe 如何、multi-symbol 表现是否正常、head 分布是否正常”，并给出**可复现命令**与**产物路径**。

---

## 1) 实验设置

- **模型**：`results/exp003_btc_eth_train_norm/path_primitives_4h_80h_min_multi_240T/model.pt`
- **特征层（FeatureStore layer）**：`features_83f12ecc5e`
- **评估区间（OOS）**：`2025-07-01` ~ `2025-11-30`
- **Symbols**：`BTCUSDT,ETHUSDT`
- **Timeframe**：`240T`（4H）
- **Router**：`rule_mode_3action`（NO_TRADE / MEAN / TREND）
- **Execution/Returns**：`rr_execution`（ATR RR 模拟，修复了“只产生一笔交易”的 bug，见下方说明）

---

## 2) 一键复现实验命令

> 该流程会产出 preds、router mode、RL logs、shadow/counterfactual 报告，以及 head 分布统计。

```bash
cd /workspaces/ml_trading_bot

EVAL_DIR=results/exp003_btc_eth_train_norm/eval_2025H2
MODEL=results/exp003_btc_eth_train_norm/path_primitives_4h_80h_min_multi_240T/model.pt
LAYER=features_83f12ecc5e

# 1) OOS 预测（从 FeatureStore 读）
python3 scripts/predict_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2025-07-01 \
  --end-date 2025-11-30 \
  --model "$MODEL" \
  --output "$EVAL_DIR/preds" \
  --features-store-root feature_store \
  --features-store-layer "$LAYER"

# 2) Rule router（3-action）
python3 scripts/rule_mode_3action.py \
  --preds "$EVAL_DIR/preds" \
  --model "$MODEL" \
  --output "$EVAL_DIR/mode_3action.parquet"

# 3) 构建 logs + RR execution returns（修复版输出）
python3 scripts/rl_build_logs_3action.py \
  --preds "$EVAL_DIR/preds" \
  --mode "$EVAL_DIR/mode_3action.parquet" \
  --data-path data/parquet_data \
  --timeframe 240T \
  --start-date 2025-07-01 \
  --end-date 2025-11-30 \
  --output "$EVAL_DIR/logs_3action_rrfix.parquet" \
  --returns-source rr_execution \
  --model "$MODEL"

# 4) Shadow eval（行为稳定性 gate）
python3 scripts/rl_shadow_eval_3action.py \
  --logs "$EVAL_DIR/logs_3action_rrfix.parquet" \
  --out "$EVAL_DIR/rl_e2e_rrfix/shadow" \
  --train_ratio 0.7

# 5) Counterfactual eval（Sharpe/DD 等）
python3 scripts/rl_counterfactual_eval_3action.py \
  --logs "$EVAL_DIR/logs_3action_rrfix.parquet" \
  --out "$EVAL_DIR/rl_e2e_rrfix/counterfactual" \
  --train_ratio 0.7

# 6) Head 分布统计（从 logs 读 head_*）
python3 scripts/report_path_primitives_preds.py \
  --preds "$EVAL_DIR/logs_3action_rrfix.parquet" \
  --out-dir "$EVAL_DIR/head_report_logs"
```

---

## 3) 关键结果（Sharpe / 回撤 / per-symbol）

来自：`results/exp003_btc_eth_train_norm/eval_2025H2/rl_e2e_rrfix/counterfactual/metrics.json`

- **Rule（Router mode）Sharpe_mean**：`1.8248`
- **Rule avg max DD**：`0.0626`
- **Rule avg total return**：`0.0725`

per-symbol（`per_symbol.csv`）：

- **BTCUSDT**：Sharpe `0.1153`，TotalReturn `0.00067`，MaxDD `0.0244`
- **ETHUSDT**：Sharpe `3.5342`，TotalReturn `0.1443`，MaxDD `0.1008`

> 注意：`pred_*` 指标在 counterfactual 报告里对应 **BC Router**（行为克隆策略）的 predicted mode，
> 当前这次跑出来 BC 在 test 端塌缩为 `NO_TRADE`，因此 `pred_sharpe_mean=0` 是“无交易/无收益”的结果，
> 不等价于“NN heads 不行”。你真正关心的“回测 Sharpe”应看 `rule_*`。

---

## 4) Router/Heads 分布是否正常？

### 4.1 Router mode 分布（OOS 期间）

日志（`eval.log`）记录的 mode 计数：

- `NO_TRADE`: 1353
- `MEAN`: 396
- `TREND`: 77

这说明**不是全程 NO_TRADE**（回测可运行），但也偏保守（大部分时间不交易）。

### 4.2 Heads 分布（多头输出统计）

统计文件：
- `results/exp003_btc_eth_train_norm/eval_2025H2/head_report_logs/preds_head_summary_per_symbol.csv`

可重点查看（每个 symbol 一行）：
- `head_dir_score`（方向分数，[-1,1]）
- `head_mfe_atr` / `head_mae_atr`（ATR 单位）
- `head_t_to_mfe`（bars）

---

## 5) 产物位置（常规报告汇总）

根目录：`results/exp003_btc_eth_train_norm/eval_2025H2/`

- **Predictions**：`preds/preds_BTCUSDT.parquet`, `preds/preds_ETHUSDT.parquet`
- **Rule mode**：`mode_3action.parquet`
- **Logs（含 head_* + ret_mean/ret_trend）**：`logs_3action_rrfix.parquet`
- **Shadow eval**（HTML + metrics/confusion/sample）：
  - `rl_e2e_rrfix/shadow/shadow_report.html`
  - `rl_e2e_rrfix/shadow/metrics.json`
- **Counterfactual eval**（HTML + metrics + per_symbol）：
  - `rl_e2e_rrfix/counterfactual/report.html`
  - `rl_e2e_rrfix/counterfactual/metrics.json`
  - `rl_e2e_rrfix/counterfactual/per_symbol.csv`
- **Head 分布统计**：
  - `head_report_logs/preds_head_summary_per_symbol.csv`

---

## 6) 重要实现备注：RR execution 返回序列修复

这次回测用的 `rr_execution` 是 `src/time_series_model/rl/execution_returns_rr.py`。
之前存在一个严重问题：模拟器只会生成第一笔交易，后续永远不再开仓，导致 `ret_mean/ret_trend` 基本全是 0，
从而 Sharpe 全是 0（看起来像“模型不交易”）。

已修复：允许多笔 sequential trades（不重叠但可以多次开平仓），因此现在 `ret_mean/ret_trend` 不再近似全 0，
`rule_sharpe_mean` 才是有效的回测结果。


