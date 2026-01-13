## 阈值“平坦高原”调参协议（Router / 规则阈值 / 止损止盈）

目标：把调参从“找尖峰”变成“找高原”——在多窗口、bootstrap、局部扰动下仍然稳。

本协议先覆盖 **Rule Router 3-action** 的阈值（你当前最关键的可控旋钮）。  
后续扩展到 Execution 的 SL/TP 阈值时，仍沿用同一套评估口径（只是 returns_source 不同）。

---

### 1) 你需要的输入（复用已有产物）

- `preds_*.parquet`：`mlbot nnmultihead predict` 的输出（包含 `pred_dir_prob/pred_mfe_atr/pred_mae_atr/pred_t_to_mfe`）
- `logs_3action.parquet`：`mlbot rl build-logs-3action` 的输出（包含 `ret_mean/ret_trend`）
- `model.pt`：用于自动判断 `preds_in_log1p`（避免“阈值口径不一致”）
- baseline thresholds：写成 JSON（7 个 key）

---

### 2) 一条命令跑“平坦高原”搜索（推荐）

下面这个脚本会：
- 围绕 baseline 做随机扰动生成 candidates
- 多个时间子窗口（walk-forward slices）评估
- bootstrap 评估稳健性
- 输出 `candidates.csv / summary.json / report.md`

示例命令（把路径换成你自己的输出目录）：

```bash
/usr/bin/python3 /workspaces/ml_trading_bot/scripts/plateau_tune_rule_router_3action.py \
  --preds /workspaces/ml_trading_bot/results/nnmh_e2e/smoke_2025M05_2025M10_reqonly_tunedthr_v1/preds \
  --logs  /workspaces/ml_trading_bot/results/nnmh_e2e/smoke_2025M05_2025M10_reqonly_tunedthr_v1/logs_3action.parquet \
  --model /workspaces/ml_trading_bot/results/nnmultihead/poolb_primitives_2023_2024/path_primitives_4h_80h_min_poolb_multi_240T/model.pt \
  --baseline-json /workspaces/ml_trading_bot/results/nnmh_e2e/smoke_2025M05_2025M10_reqonly_tunedthr_v1/router_thresholds_baseline.json \
  --out /workspaces/ml_trading_bot/results/plateau/router3action_2025M05_2025M10_v1 \
  --n-candidates 300 \
  --n-windows 6 --min-days-per-window 25 \
  --n-bootstrap 30 \
  --rel-sigma 0.05 --abs-sigma 0.01 \
  --entry-delay 0 \
  --cost-per-turnover 0.0 --slippage-bps 0.0
```

baseline JSON 文件格式（7 个 key）：

```json
{
  "mfe_min": 0.1259208384,
  "eff_min": 1.0434303531,
  "dir_conf_trend_min": 0.0641852109,
  "mfe_trend_min": 0.4572347656,
  "ttm_trend_min": 5.6310524663,
  "eff_mean_min": 1.2011240134,
  "ttm_mean_max": 29.0044411469
}
```

---

### 3) 产物怎么解读（你要看的 3 个数）

- **`win_score_p25`**：越高越好（代表“坏窗口”也没崩）
- **`plateau_frac_ge_95pct`**：越高越好（代表“高原更宽”，更不炼丹）
- **trade_rate_mean**（在 `candidates.csv`）：避免“高 Sharpe 但靠过度交易”这种伪解

---

### 4) 评分口径（写死在 summary.json，便于复盘）

每个时间子窗口里先算：
\( \text{window_score} = \text{SharpeMean} - \lambda\cdot\text{SharpeStd} - \mu\cdot\text{DDMean} \)

最终 robust_score 取：
- `mean(window_score) + mean(bootstrap(window_score))`

你可以通过 `--lambda/--mu` 调整“稳健性 vs 收益”的偏好。

