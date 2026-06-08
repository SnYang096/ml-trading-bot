# TPC macro_pullback_pct **替代** depth prefilter（2026-06-10）

## R&D 阶段进度

流程定义见 [`../LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) §标准 R&D 阶段。

| Phase | 状态 | 产物 |
|-------|------|------|
| 0 | ✅ | `tpc_macro_pullback_pct_f` + labeled parquet |
| 1 | ✅ | label scan + OHLC → [`PHASE1_REPORT.md`](PHASE1_REPORT.md) |
| 2 | ✅ | τ L15/S12 主候选；L20/S15 对照 |
| 3 | 🔄 | segment grid（`trading_map: true` 同跑分段地图） |
| 4 | ⏳ | bull 三变体对比 map：[`tpc_macro_pullback_bull_maps_grid.yaml`](tpc_macro_pullback_bull_maps_grid.yaml) |
| 5 | ⏳ | `DECISION.md` promote（须过三条杠） |

| 字段 | 值 |
|------|-----|
| 策略 | tpc |
| Grid | [`tpc_macro_pullback_replace_grid.yaml`](tpc_macro_pullback_replace_grid.yaml) |
| 变体树 | `config_experiments/tpc_macro_replace_*_strategies/`（**静态** prefilter，直接改 YAML） |
| 扫描 | `results/tpc/research/macro_pullback_scan_20260609/` |

## 背景

- `tpc_pullback_depth`（20-bar 区间分位）与「大周期价格回撤 %」不对齐；depth 下界/上界 grid 已证伪。
- 新特征 `tpc_macro_pullback_pct_long/short`（N=240 + EMA regime gate）应用 **macro 口径** 做 prefilter。
- **本实验用 macro 规则替换 prod 的 `depth<=0.85`**，不是叠在 depth 上（见作废的 [`20260609_tpc_macro_pullback_validate/`](../20260609_tpc_macro_pullback_validate/)）。

语义笔记：[B系统入场语义与执行层周期错配_CN.md](../../docs/strategy/B系统入场语义与执行层周期错配_CN.md)

## 变体

| ID | prefilter |
|----|-----------|
| **E0_prod** | 现网：`tpc_pullback_depth <= 0.85` |
| **M_replace_L15_S12** | **仅** macro `any_of`：long≥0.15 / short≥0.12 |
| **M_replace_L20_S15** | **仅** macro `any_of`：long≥0.20 / short≥0.15 |
| **M_add_L15_S12** | **depth≤0.85 AND** macro `any_of` long≥0.15 / short≥0.12（叠加，见 add grid） |

## 跑法

```bash
# 历史扫描（标定 τ，可选重跑）
PYTHONPATH=. python scripts/research/scan_tpc_pullback_lookback.py \
  --out results/tpc/research/macro_pullback_scan_20260609

# canonical 三阶段 × 3 变体 = 9 runs（replace 实验，已跑完）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260610_tpc_macro_pullback_replace/tpc_macro_pullback_replace_grid.yaml \
  --quiet-signal-logs

# macro + depth 叠加：E0 vs M_add，3×2 = 6 runs
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260610_tpc_macro_pullback_replace/tpc_macro_pullback_add_grid.yaml \
  --quiet-signal-logs
```

**Trading map（推荐：segment grid 内嵌，不另跑全窗）**

grid yaml 顶层 `trading_map: true` → 每段回测顺带出 `trading_map_tpc_event.html`（与 capital_report 同目录）。

三变体 bull 对比（仅 3 runs，已够用）：

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260610_tpc_macro_pullback_replace/tpc_macro_pullback_bull_maps_grid.yaml \
  --quiet-signal-logs
# 或：bash config/experiments/20260610_tpc_macro_pullback_replace/run_trading_maps.sh
```

## 结果

- 分段 R：`results/tpc/experiments/macro_pullback_*_20260610/<variant>/<segment>/capital_report.json`
- 分段地图：同上目录下 `trading_map_tpc_event.html`（例 bull：`.../bull_2023_2024/trading_map_tpc_event.html`）

## 决策

见 [`DECISION.md`](DECISION.md)（canonical 三阶段 Total R / maxDD / 笔数 + **bull_2023_2024** trading map 是否落在大回调区）。
