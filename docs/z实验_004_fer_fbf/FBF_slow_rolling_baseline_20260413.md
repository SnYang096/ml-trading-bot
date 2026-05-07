# FBF slow_realistic 滚动基线（2026-04-13）

## Run 标识

| 字段 | 值 |
|------|-----|
| **run_id** | `20260413_162634` |
| **模式** | `slow_realistic`（cadence 3 月结构快照 + 月度快环） |
| **窗口** | 16 个日历月：`2023-09` … `2024-12` |
| **stitched_total_r** | **+36.7554** |
| **stitched_total_trades** | **240** |
| **约胜率** | ~42%（事件回测单笔汇总；与固定 RR≈2:1 相容） |

## 结果路径

- 汇总：`results/fbf/slow-rolling-sim/_rolling_sim/20260413_162634/stitched_summary.json`
- 交易地图（拼接）：`.../20260413_162634/trading_map_stitched.html`
- 连续图：`.../20260413_162634/trading_map_continuous.html`
- 分月 ledger：`.../20260413_162634/monthly_ledger.jsonl`

## 配置要点（与本次 run 对齐）

- **流水线**：`config/prod_train_pipeline_2h_slow_fbf_only.yaml`
  - `rolling_calibration.execution_opt.enabled: false`（避免月度 sym_r 覆盖 archetype 固定 SL/TP）
  - `strategies.fbf.simple_execution`：`sl_r: 1.0`，`tp_r: 2.0`，`timeout_bars: 108`（与事件层语义对齐）
- **执行 archetype**：`config/strategies/fbf/archetypes/execution.yaml`
  - `stop_loss.type: fixed`，`initial_r: 1.0`，`take_profit.target_r: 2.0`（约 **RR 2:1**）
  - `trailing.enabled: false`，`breakeven.enabled: false`
  - `holding.time_stop_bars: 36`
- **入场**：偏稀疏 — `prefilter.yaml` / `entry_filters.yaml` 提高失败突破与结构门槛；`trend_r2_20` 制度闸收紧。

## 复现命令

```bash
mlbot pipeline run --all --config config/prod_train_pipeline_2h_slow_fbf_only.yaml --stage rolling_sim
```

（可选加速试验：`--skip-shap`，与完整慢滚不等价。）

## 后续实验备忘（未实施）

- **单笔更「肥」、仍保留事件感**：优先试 **只提高 `take_profit.target_r`**（或加 `structural_exit`），避免同步大幅放宽 `initial_r`（历史对照：宽止损实验曾显著拉低总 R）。
- **产品方向**：若放弃「波动区失败突破」主叙事，可考虑与 ME/BPC/TPC **正交** 的 **成功突破 / 延续** 家族（需新特征与 archetype，与现有三者区分）。

## Git tag

与本基线对应的 annotated tag：`fbf-slow-baseline-20260413-36R-240t`（见仓库 tag 列表）。
