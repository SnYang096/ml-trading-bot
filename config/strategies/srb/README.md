# SRB（Structural Range Breakout）

## 与 BPC / ME / TPC 的分工（趋势家族互补）

| 策略 | 核心语义 | 典型入场特征（摘要） | 方向源（本仓） |
|------|-----------|----------------------|----------------|
| **BPC** | 突破 → **回踩** → 延续 | Donchian + pullback/restore + `box_compression` 蓄势 | Donchian × MACD × EMA1200 |
| **ME** | **动量扩张** | accel + multi-TF alignment + VWAP 带 | accel/cvd/MACD × VWAP1200 |
| **TPC** | 大趋势中 **回调不破** | EMA1200 死区规避 + `tpc_pullback_depth` | MACD × EMA1200 |
| **SRB** | **关键位成功突破 → 顺势延续** | SR + 频谱 + **`tpc_score_breakout`** + **盒沿突破/扩张** + path/trend_r2 | **ROC20 × EMA1200** |

目标：**SRB 尽量吃「结构突破后的趋势段」**，不在「长期窄盒横盘」里高频刷单；与 ME/BPC/TPC 的**信号源错开**，减少同源拥挤。

## A~E 改动映射（2026-04-24）

- **A（Prefilter）**：去掉「高 `box_stability` + 极窄 `box_width`」；改为 **`box_compression_score >= 1`** + **`box_breakout_up/down` any_of** + `tpc_score_breakout` + `path_efficiency` / `trend_r2_20` + `bpc_impulse_return_atr`。
- **B（执行/加仓）**：`min_order_interval_minutes: 120`，`max_add_times: 2`，`trend_health_gate` 启用（母仓 ≥0.5R 且入场 ≤288 bar 才加仓）。
- **C（两段式）**：`srb_staged_entry_2b.enabled: true`（`archetypes/execution.yaml`）。
- **D（方向）**：`archetypes/direction.yaml` 改为 **ROC20 sign × EMA1200**，不再与 TPC 共用 MACD×EMA1200。
- **E（统计）**：用下面命令做 **rolling_sim 前后对比**（建议 `--no-adopt`）。

## 建议统计命令

```bash
# 新配置（当前工作区）
mlbot pipeline run --all --no-adopt \
  --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_srb_only.yaml \
  --stage rolling_sim --skip-shap 2>&1 | tee log.srb.turbo.trend_ride.txt

# 对比 stitched 汇总（将 <OLD_RUN_TS> 换成改前 run 目录名）
python - <<'PY'
import json
from pathlib import Path
new = Path("results/srb/turbo-rolling-sim/_rolling_sim/<NEW_TS>/stitched_summary.json")
old = Path("results/srb/turbo-rolling-sim/_rolling_sim/<OLD_TS>/stitched_summary.json")
for p in (old, new):
    d = json.loads(p.read_text())
    print(p.parent.name, "total_r=", round(d["stitched_total_r"], 2), "trades=", d["stitched_total_trades"])
PY
```

单币连续图：各 run 下 `trading_map_continuous.html`；按月：`fast_month_YYYY-MM/srb/event_backtest_srb.json` 的 `funnel` / `total_r`。

## 回退

若通过率过低或 stitched 变差：优先放宽 **`tpc_score_breakout`**、`path_efficiency_pct`、`box_compression_score` 门槛；其次将 **`srb_staged_entry_2b.enabled`** 设回 `false` 做消融。

## 2026-04-24b（统计放宽 + 骑趋势出场）

- **Prefilter**：`tpc_score_breakout` 0.30→0.18、`bpc_impulse_return_atr` 0.12→0.05、`box_compression_score` 1.0→0.88、`path_efficiency` 0.26→0.18、`trend_r2_20` 0.06→0.03，盒宽上下界略放宽（rolling 过稀时对症）。
- **Execution**：`initial_r` 6→7；`breakeven` trigger 3.5→5.5、`lock_level_r` 0→**-0.2**（多空对称：LONG SL 略低于 entry、SHORT SL 略高于 entry，减洗盘扫「贴价保本」）；`trail_r`/`trail_r_far` 加宽；`opposite_sr_buffer_atr`、`l3_structural_exit.buffer_atr` 略放宽。
