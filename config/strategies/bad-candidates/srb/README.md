# SRB（Structural Range Breakout）

> **2026-04 归档**：本策略目录已迁至 `config/strategies/bad-candidates/srb`（rolling 效果未达主腿标准，仅供对照与复现实验）。生产部署默认不再同步至 `live/highcap`；跑管线请使用 `config/prod_train_pipeline_*_srb_only.yaml`（其中 `strategies.srb.config` 已指向本路径）。

## 与 BPC / ME / TPC 的分工（趋势家族互补）

| 策略 | 核心语义 | 典型入场特征（摘要） | 方向源（本仓） |
|------|-----------|----------------------|----------------|
| **BPC** | 突破 → **回踩** → 延续 | Donchian + pullback/restore + `box_compression` 蓄势 | Donchian × MACD × EMA1200 |
| **ME** | **动量扩张** | accel + multi-TF alignment + VWAP 带 | accel/cvd/MACD × VWAP1200 |
| **TPC** | 大趋势中 **回调不破** | EMA1200 死区规避 + `tpc_pullback_depth` | MACD × EMA1200 |
| **SRB** | **关键位成功突破 → 顺势延续** | **L3 wide SR 带**（`wide_sr_dist_atr`）+ `sr_strength_max` + `srb_l3_breakout_age_decay`；订单流放 entry | **L3 breakout side → ROC20 × EMA1200** |

目标：**SRB 尽量吃「关键支撑/阻力被突破后的趋势段」**；prefilter 以 **SR 层级（L2 强度 + L3 关键带）** 为主，不用盒结构当主语义；与 ME/BPC/TPC **错开**。

## A~E 改动映射（2026-04-24）

- **A（Prefilter）**：只用 **大结构**：`sr_strength_max` + **`wide_sr_dist_atr`（贴近 L3 上/下沿）** 或 `srb_l3_breakout_age_decay`（突破后窗口）。订单流 / EMA 2b / impulse 不放 prefilter。
- **B（执行/加仓）**：模仿 BPC 的 ATR ladder：`min_current_r_by_add: [1,2,3]`、`add_size_multipliers: [1,2,3]`，不依赖再次出现 SRB signal。
- **C（两段式）**：runtime `srb_staged_entry_2b` 已关闭；2a/2b 下沉为普通特征：`srb_l3_breakout_age_decay` + `srb_l3_breakout_2b_score`。
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

若通过率过低：放宽 **`wide_sr_dist_atr`** 上界、降低 **`srb_l3_breakout_age_decay`** 阈值或延长其 feature `max_age_bars`；若过稀再略降 **`sr_strength_max`**（慎用，伤语义）。Entry 侧可下调 `srb_sr_success_breakout_score`。

## 2026-04-24b（统计放宽 + 骑趋势出场）

- **Execution**：`initial_r` 6→7；`breakeven` trigger 3.5→5.5、`lock_level_r` **-0.2**；`trail_r`/`trail_r_far` 加宽；`opposite_sr_buffer_atr`、`l3_structural_exit.buffer_atr` 略放宽。

## 2026-04-25（Prefilter 语义：SR 突破，非盒）

- Prefilter 改为 **L3 关键 SR 带 + L2 强度 + 突破频谱 + impulse**；去掉盒/路径效率/trend_r2 主过滤，与「Structural Range Breakout」命名一致。

## 方向诊断（图上 Prefilter 有脉冲但 Dir 恒为 0）

常见原因不是「规则太严」而是 **特征 dict 里根本没有 `ema_1200_position` / `roc_20`**：`row_to_features` 会丢掉 NaN，慢窗在部分 bar 上为空 → `DirectionEvaluator` 两档全失败 → `direction_value=0`，与 prefilter 是否通过无关。事件回测已对上述两列做 **按 (symbol, tf) 的因果前向填充**（`event_backtest._apply_pcm_direction_ffill`），并保留 `roc_20` **sign 兜底**规则。

## 2026-04-27（更早启动 + 空单可用性）

- **方向**：`direction.yaml` 增加 **strict → relaxed dual**（`roc_20` + `ema_1200_position`，`epsilon: -0.065`），缓解 EMA1200 滞后导致的多空「永远对不齐」。
- **sr_wide_entry_guard**：`min_distance_atr` **2.0 → 1.15**，减少贴近 L3 下沿时的 **SHORT** 误拒。
- **2a+2b**：`confirm_k` **3→2**；`ema_pos_min` / `ema_slope_min` 略降，arm 略提前。

## 2026-04-26（肥尾 / 跟趋势执行预设）

- **止损与出场**：`initial_r` 8；`breakeven` 6.5R / lock **-0.3R**；`trailing` activation **7R**，`trail_r`/`trail_r_far` **7 / 10.5**；`l3_structural_exit.buffer` **0.45**；`structural_sl` buffer **0.9**。
- **持仓**：`max_holding_bars` **480**，`time_stop_uncap_mfe_r` **1.5**。
- **加仓**：`trend_health_gate` MFE **0.35**，staleness **360** bars。
- **2a+2b**：`post_2a_max_bars` **32**，`ema_slope_bars` **3**，`ema_pos_min` **0.02**，`ema_slope_min` **0.008**，`arm_pcm_bars` **12**。
