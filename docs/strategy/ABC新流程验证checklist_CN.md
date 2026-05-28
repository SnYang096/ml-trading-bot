# ABC 新流程验证 checklist（含树通道）

> 用途：拿到这条 checklist，**一项一项跑**，每项产物落到磁盘 → 人工对账 → 通过后再做下一项。  
> 配套：[`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) §2–§3；**命令口径** [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) §1（`mlbot research`，非 `quick_layer_scan.py`）。  
> **本轮实测记录**：[`ABC验证操作记录_20260526_CN.md`](ABC验证操作记录_20260526_CN.md)（部分命令为 20260526 对拍遗留）。
>
> 通用约定：
>
> - 所有命令在仓库根目录运行，`PYTHONPATH=src:scripts` 前缀已默认。
> - **不**改 live；所有产物落到 `results/` 或 `docs/decisions/`。
> - **每项都必须留一个 markdown / json 产物**，便于复盘 + watchdog 追踪。
> - "✅ 通过条件"是这一项要看的最小输出；只要满足就过，不必精雕细琢。

## 0. 环境准备

```bash
git pull
PYTHONPATH=src:scripts python -m pytest \
  tests/unit/test_calibrate_roll_validation_mode.py \
  tests/unit/test_quick_layer_scan_modes.py \
  tests/unit/test_variant_grid_engine.py \
  tests/unit/test_build_grid_segment_labels.py \
  tests/unit/test_decision_doc_templates.py \
  tests/unit/test_tree_slugs.py -x -q
```

✅ 通过条件：6 个测试全部通过；如有 fail，先修 fail 再继续。

---

## A. A1 系统（spot_accum_simple）

A1 没有 R&D 闭环，只做"环境是否仍然能买"复盘。

| # | 任务 | 命令 | ✅ 通过条件 |
|---|---|---|---|
| A1.1 | 周线 EMA200 死区复盘 | `mlbot research scan condition-set --features-parquet results/<recent>/features.parquet --label dummy --condition "deep_bear: weekly_ema_200_position<0" --output results/spot_accum_simple/quick_scan/<日期>.md` | 报告写入 `results/spot_accum_simple/quick_scan/`；`n_deep_bear` ≥ 100 |
| A1.2 | 阶梯卖出回测 | `PYTHONPATH=src:scripts python -m mlbot backtest strategy --strategy spot_accum_simple --start 2022-01-01 --end 2026-04-01` | `results/spot_accum_simple/backtest/` 有 `metrics.json`；ret/maxDD 与历史快照一致 |

---

## B. B 系统（BPC/TPC/ME/SRB）

主线 R&D 流程，每条都按"假设 → 离线扫 → 双段回测 → 决策文档 → live"。

### B.1 假设阶段 — `mlbot research`

```bash
PARQ=$(ls -t results/train_final/tpc/train_final_*/tpc/features_labeled.parquet | head -1)
DATE=$(date +%Y%m%d)

# B.1.a 单 feature plateau 是否存在
mlbot research scan feature-plateau \
  --strategy tpc --layer prefilter \
  --features-parquet "$PARQ" --label success_no_rr_extreme \
  --feature tpc_pullback_depth --operator "<=" \
  --grid 0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95 \
  --subset "tpc_semantic_chop<=0.4 AND ema_1200_position>=0.10" \
  --calendar-window 2024-01-01,2025-01-01 \
  --output results/tpc/quick_scan/pullback_in_bull_${DATE}.md

# B.1.b regime 候选对照（H/F/F'）
mlbot research scan condition-set \
  --strategy tpc --layer regime \
  --features-parquet "$PARQ" --label success_no_rr_extreme \
  --subset "tpc_semantic_chop<=0.4" \
  --condition "H: abs(ema_1200_position)>0.10" \
  --condition "F: abs(ema_1200_position)>0.12" \
  --condition "Fp: abs(ema_1200_position)>0.10 AND abs(ema_1200_slope_10)>0.002" \
  --output results/tpc/quick_scan/regime_candidates_${DATE}.md

# B.1.c IC 衰减 + baseline 对照
mlbot research ic --strategy tpc \
  --features-parquet "$PARQ" \
  --features ema_1200_position,vol_persistence,tpc_pullback_depth \
  --horizons 1,3,5,10,20,50 \
  --baseline-json config/monitoring/factor_ic_baseline_tpc_20260526.json \
  --output results/tpc/quick_scan/ic_decay_${DATE}.md
```

✅ 通过条件：3 份 md 同时存在；至少一个候选满足 `|z|>2 且 Δpp ≥ +0.5pp`，否则回到假设阶段重选。

### B.2 双段回测 — `event_backtest --variant-grid`

```bash
# 准备 grid yaml（参考已有的 config/experiments/tpc_variant_grid_smoke.yaml）
# 把候选 variant 复制到 config_experiments/<variant>_strategies/ 后：

PYTHONPATH=src:scripts python -m scripts.event_backtest --variant-grid \
  config/experiments/<your_grid>.yaml
```

✅ 通过条件：
- 每个 variant 在 `results/tpc/experiments/<variant>/` 有 `capital_report.json` + `event_trades_tpc.csv`；
- `results/tpc/experiments/EXPERIMENT_INDEX.json` 含所有 runs；
- **两段**都跑了：recent + bull 2024。

### B.3 决策文档骨架 — `_new_decision_doc.py`

```bash
PYTHONPATH=src:scripts python scripts/_new_decision_doc.py \
  --experiment-index results/tpc/experiments/EXPERIMENT_INDEX.json \
  --topic tpc_<topic>_$(date +%Y%m%d)
```

✅ 通过条件：`docs/decisions/tpc_<topic>_<日期>.md` 生成，含变体表、双段结果表、决策占位。**人手填**「by-side breakdown + 决策」后才视为完成。

### B.4 promote + watchdog baseline 刷新

```bash
cp config_experiments/<new>/tpc/archetypes/gate.yaml \
   config/strategies/tpc/archetypes/gate.yaml
cp config/strategies/tpc/archetypes/gate.yaml \
   live/highcap/config/strategies/tpc/archetypes/gate.yaml

PARQ=$(ls -t results/train_final/tpc/train_final_*/tpc/features_labeled.parquet | head -1)
# 编辑 config/monitoring/regime_watchdog_baseline.json，更新 bull_share / trigger_rate

PYTHONPATH=src:scripts python scripts/regime_watchdog.py \
  --strategies tpc --window-parquet "$PARQ" \
  --baseline-json config/monitoring/regime_watchdog_baseline.json
```

✅ 通过条件：`exit_code == 0`；`report.json` 包含 IC drift 与 PSI（Phase B 新增），无 ALERT。

### B.5 周度监控

```bash
# 加 cron（见方法论 §2.6）
PYTHONPATH=src:scripts python scripts/regime_watchdog.py \
  --strategies tpc,bpc,me,srb --window-parquet "$PARQ" \
  --baseline-json config/monitoring/regime_watchdog_baseline.json
PYTHONPATH=src:scripts python scripts/regime_drift_monitor.py \
  --strategies tpc,bpc,me,srb --window-parquet "$PARQ"
```

✅ 通过条件：两脚本均 `exit 0`；若 ALERT 触发，回到 B.1 假设阶段。

---

## C. C 系统（chop_grid / trend_scalp）

C 不走 B 的 SHAP/方向 label 工厂；走"语义代理 R&D"（[`WORKFLOW_..._CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) §2.2.1）。

### C.1 多腿基线回测

```bash
PYTHONPATH=src:scripts python scripts/chop_grid_backtest.py \
  --config config/strategies/chop_grid/research/calibrate_roll.default.yaml \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --start 2025-04-01 --end 2026-04-01 \
  --out-dir results/chop_grid/baseline_recent
```

✅ 通过条件：`grid_segments.csv` + `grid_trades.csv` + `capital_report.json` 均存在；至少 50 个 segment。

### C.2 语义代理候选 grid 回测 — `--variant-grid + engine=chop_grid`

```bash
# Phase D 新功能：variant_grid.py 通过 engine 字段分派
PYTHONPATH=src:scripts python -m scripts.event_backtest --variant-grid \
  config/experiments/chop_grid_semantic_proxy_grid.yaml
```

✅ 通过条件：所有 variant 都在 `results/chop_grid/experiments/<variant>/` 写出 `grid_segments.csv`；`EXPERIMENT_INDEX.json` 中 `engine: chop_grid` 出现。

### C.3 segment label 桥 — `_build_grid_segment_labels.py`

```bash
PARQ=$(ls -t results/train_final/chop_grid/train_final_*/chop_grid/features_labeled.parquet | head -1)

for V in baseline_recent proxy_tpc_recent proxy_chop_not_box_recent; do
  PYTHONPATH=src:scripts python scripts/_build_grid_segment_labels.py \
    --segments results/chop_grid/experiments/$V/grid_segments.csv \
    --features-parquet "$PARQ" \
    --out results/chop_grid/experiments/$V/seg_labeled.parquet
done
```

✅ 通过条件：每个 variant 都有 `seg_labeled.parquet`，列含 `seg_total_r_over_dd / seg_adverse_break_rate / seg_maker_return_per_round / seg_period_5_ok`。

### C.4 代理 × C KPI 扫描 — `mlbot research scan condition-set`

```bash
mlbot research scan condition-set \
  --features-parquet results/chop_grid/experiments/proxy_tpc_recent/seg_labeled.parquet \
  --label seg_total_r_over_dd \
  --condition "bpc_in: bpc_semantic_chop>=0.50" \
  --condition "tpc_in: tpc_semantic_chop>=0.50" \
  --condition "not_box: chop_not_box>0" \
  --output results/chop_grid/quick_scan/proxy_kpi_$(date +%Y%m%d).md
```

✅ 通过条件：报告 ≥ 1 个候选 Δ vs base ≥ 5% 且 `|z|>2`。

### C.5 决策文档骨架（c_semantic_proxy 模板）

```bash
PYTHONPATH=src:scripts python scripts/_new_decision_doc.py \
  --experiment-index results/chop_grid/experiments/EXPERIMENT_INDEX.json \
  --topic-template c_semantic_proxy \
  --topic chop_grid_proxy_$(date +%Y%m%d)
```

✅ 通过条件：`docs/decisions/chop_grid_proxy_<日期>.md` 含「entry_feature 候选 × KPI 表」与「Plateau 宽度」骨架。

---

## D. 树通道（fast_scalp / short_term_swing）

**首次跑**之前确认 slug 目录存在：

```bash
ls config/strategies/fast_scalp/ config/strategies/short_term_swing/
# 各应有 meta/features/labels/model/backtest.yaml 共 5 个
```

### D.1 IC 对齐 horizon

```bash
PARQ=$(ls -t results/train_final/tpc/train_final_*/tpc/features_labeled.parquet | head -1)
mlbot research ic --strategy fast_scalp \
  --features-parquet "$PARQ" \
  --features macd_atr,bb_width_normalized_pct,tpc_semantic_chop,cvd_short_normalized,vpin_short,atr_percentile,hurst_short \
  --horizons 1,3,5,10,20,50 \
  --output results/fast_scalp/ic_decay_$(date +%Y%m%d).md

mlbot research ic --strategy short_term_swing \
  --features-parquet "$PARQ" \
  --features ema_1200_position,ema_1200_slope_10,trend_confidence,hurst_long,bb_width_normalized_pct,macd_atr \
  --horizons 1,3,5,10,20,50 \
  --output results/short_term_swing/ic_decay_$(date +%Y%m%d).md
```

✅ 通过条件：每个 slug 至少有 5 列满足 |IC|>0.02 且 best_lag 在目标区间（fast: 1–5；swing: 10–20）。然后**手动**把通过的列写到 slug 的 `features.yaml`。

### D.2 训练 + 出 score

```bash
PYTHONPATH=src:scripts python -m mlbot train final \
  -c config/strategies/fast_scalp \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --output-dir results/train_final/fast_scalp/$(date +%Y%m%d_%H%M%S)

PYTHONPATH=src:scripts python -m mlbot train final \
  -c config/strategies/short_term_swing \
  --output-dir results/train_final/short_term_swing/$(date +%Y%m%d_%H%M%S)
```

✅ 通过条件：每个 slug 的 `<out>/predictions.parquet` 存在；`score` 列有 ≥ 5000 非 NaN 行。

### D.3 τ plateau 标定

```bash
SCORE=results/train_final/fast_scalp/<最新>/predictions.parquet
PYTHONPATH=src:scripts python scripts/regime_threshold_calibrate.py \
  --features-parquet "$SCORE" --feature score --operator ">=" \
  --grid 0.45,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.65 \
  --label forward_rr \
  --out results/fast_scalp/tau_plateau_$(date +%Y%m%d).md
```

✅ 通过条件：plateau 报告标出 mid 值；把 mid 写到 `config/strategies/fast_scalp/backtest.yaml` 的 `long_entry_threshold`（短同理）。

### D.4 双段回测验证

```bash
# 准备 grid yaml 类似 config/experiments/tpc_variant_grid_smoke.yaml
# variant: fast_scalp_recent / fast_scalp_bull
PYTHONPATH=src:scripts python -m scripts.event_backtest --variant-grid \
  config/experiments/fast_scalp_dual_period.yaml
```

✅ 通过条件：两段 maxDD% 与 ret% 比 baseline B 槽位均不显著恶化。

### D.5 决策文档（tree_slug 模板）

```bash
PYTHONPATH=src:scripts python scripts/_new_decision_doc.py \
  --experiment-index results/fast_scalp/experiments/EXPERIMENT_INDEX.json \
  --topic-template tree_slug \
  --topic fast_scalp_initial_$(date +%Y%m%d)
```

✅ 通过条件：`docs/decisions/fast_scalp_initial_<日期>.md` 含「IC 对齐 + τ plateau + PCM 槽位影响」骨架。

### D.6 PCM 配比（shadow，不进 live）

- 在 `config/constitution/constitution.yaml` 加入 `fast_scalp` / `short_term_swing` 槽位，确认总和不超过现宪法上限（**仅 diff，不 deploy**）。
- 跑一次 `mlbot run --paper` 至少 24h，看 PCM 是否能识别新 slug，看是否出现「同方向冲突」日志。

✅ 通过条件：paper 跑通；PCM 日志看到 `fast_scalp.LONG ranking score=...` 出现。

---

## E. 总体过关条件

跑完上面所有 ✅，你应该有：

| 类别 | 产物 |
|---|---|
| 报告 | `results/{tpc,chop_grid,fast_scalp,short_term_swing}/quick_scan/*.md` ≥ 5 份 |
| 回测 | `results/<slug>/experiments/EXPERIMENT_INDEX.json` 含 ≥ 4 variant |
| 决策文档 | `docs/decisions/<topic>_<日期>.md`（B/C/Tree 各至少一份） |
| 监控 | `regime_watchdog.py` 与 `regime_drift_monitor.py` exit 0；report.json 含 IC/PSI |
| live | **未改动**（除非有显式 promote） |

---

## F. 失败时的回退

| 哪一步炸 | 怎么处理 |
|---|---|
| 0 测试不过 | 先修测试再继续；测试在 `tests/unit/test_*.py`，跑 `-vv` 看 traceback |
| B.1 候选都 `|z|<2` | 改假设（feature/阈值），不要拿 noise 继续往下跑 |
| B.2 一段恶化超 5R / 2pp DD | 走 regime-conditional（见 §2.4 Pareto rule），或直接 drop |
| C.3 join 是空 | 检查 segment.csv 的 symbol 列与 features parquet 的 symbol 是否一致；调 `--tolerance 1h` 试试 |
| C.4 KPI 表都是 NaN | `seg_total_r_over_dd` 需要 `max_drawdown != 0`；用其它 KPI 列代替（`seg_pnl_per_capital`）|
| D.1 没有 5 列 IC 显著 | 扩 candidate pool；或缩窄 horizon 范围（不要扩到 H=50） |
| D.2 mlbot train 报 import 错误 | slug 的 `labels.yaml` 指向 `forward_rr_signed_label`，检查 `src/.../strategies/labels/forward_rr_signed_label.py` 是否存在 |
| D.3 plateau 为 0 宽 | τ 不存在 → 模型没学到方向 → 回 D.1 重选特征 |
