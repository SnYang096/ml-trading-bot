# SRB QuickStrike（Structural Range Breakout 短打版）

## 与 BPC / ME / TPC 的分工（趋势家族互补）

| 策略 | 核心语义 | 典型入场特征（摘要） | 方向源（本仓） |
|------|-----------|----------------------|----------------|
| **BPC** | 突破 → **回踩** → 延续 | Donchian + pullback/restore + `box_compression` 蓄势 | Donchian × MACD × EMA1200 |
| **ME** | **动量扩张** | accel + multi-TF alignment + VWAP 带 | accel/cvd/MACD × VWAP1200 |
| **TPC** | 大趋势中 **回调不破** | EMA1200 死区规避 + `tpc_pullback_depth` | MACD × EMA1200 |
| **SRB QuickStrike** | **关键位真突破 → 同向短打推进** | **L3 wide SR 突破窗口**（`srb_l3_breakout_age_decay`）+ `sr_strength_max` + 突破频谱；订单流放 entry | **L3 breakout side** |

目标：**SRB QuickStrike 吃「关键支撑/阻力被真正突破后的短打推进」**。它和 FBF 机制相反但执行范式接近：FBF 判断假突破后反向跟随真推进；SRB QuickStrike 直接判断真突破后顺向跟随短推进。它不承担 BPC/TPC 式长趋势肥尾。

## 当前版本（2026-04-25）

- **Prefilter**：只做大结构真突破窗口：`sr_strength_max` + `spectrum_price_high_freq_ratio` + `srb_l3_breakout_age_decay`。
- **Direction**：只认 `srb_l3_breakout_side`，不再用 ROC/EMA fallback 把普通动量误识别为 SRB。
- **Entry**：`srb_sr_success_breakout_score` 确认订单流/价格同向推进。
- **Execution**：参考稳定 FBF 短打基线：`initial_r: 1.0`、`target_r: 2.0`、`time_stop_bars: 36`、`allow_add_on: false`。

## 建议统计命令

```bash
# 新配置（当前工作区）
mlbot pipeline run --all --no-adopt \
  --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_srb_quickstrike_only.yaml \
  --stage rolling_sim --skip-shap 2>&1 | tee log.srb_quickstrike.turbo.txt

# 对比 stitched 汇总（将 <OLD_RUN_TS> 换成对照 run 目录名）
python - <<'PY'
import json
from pathlib import Path
new = Path("results/srb_quickstrike/calibrate_roll.default/_rolling_sim/<NEW_TS>/stitched_summary.json")
old = Path("results/srb/calibrate_roll.default/_rolling_sim/<OLD_TS>/stitched_summary.json")
for p in (old, new):
    d = json.loads(p.read_text())
    print(p.parent.name, "total_r=", round(d["stitched_total_r"], 2), "trades=", d["stitched_total_trades"])
PY
```

单币连续图：各 run 下 `trading_map_continuous.html`；按月：`fast_month_YYYY-MM/srb/event_backtest_srb.json` 的 `funnel` / `total_r`。

## 回退

若交易过少：先小幅降低 `srb_l3_breakout_age_decay`（如 0.35→0.25）或 `srb_sr_success_breakout_score`（0.12→0.08）。若亏损集中在假突破：提高 `srb_sr_success_breakout_score`，不要扩大持仓、加仓或改成趋势肥尾执行。
