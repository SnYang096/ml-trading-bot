# chop_grid — live-aligned execution replay

**日期：** 2026-06-03  
**产物：** `results/chop_grid/experiments/exec_align_20260603/`

## 代码变更摘要

| 模块 | 变更 |
|------|------|
| [`subbar_replay.py`](../../../src/time_series_model/grid/subbar_replay.py) | `segment_execution_bounds()`；live 语义文档 |
| [`agg100ms_replay.py`](../../../src/time_series_model/grid/agg100ms_replay.py) | segment 内 aggTrades → 100ms OHLC |
| [`chop_grid_backtest.py`](../../../scripts/chop_grid_backtest.py) | 始终 subbar 窗口；默认 `execution_timeframe=1min` |
| [`calibrate_roll.default.yaml`](../../../config/strategies/chop_grid/research/calibrate_roll.default.yaml) | `grid_backtest.execution_timeframe: 1min` |

## 四段 validate（1min prod，20bps — 与对齐前一致）

| Segment | return_pct_timeline | segment_win_rate | trades |
|---------|---------------------:|-----------------:|-------:|
| bear_2022 | +3.54% | 41.9% | 327 |
| bull_2023_2024 | +5.16% | 39.5% | 782 |
| recent_range_to_bear | +2.46% | 39.2% | 542 |
| **recent_6m_oos** | **-0.67%** | 37.3% | 216 |

1min canonical 数值与 pre-align 跑批一致（本就使用右边界窗口）。

## OOS 对照（对齐后）

| 设定 | return_pct | vs 旧 reconcile |
|------|----------:|----------------|
| **2h exec · 4bps · aligned subbar** | **+6.53%** | 旧 optimistic 2h path **+12.66%** |
| 1min exec · 20bps · prod | -0.67% | -0.75%（≈同） |
| 1min · 20bps · spacing 1.25/1.2% | -0.04% | 略好于 baseline spacing |

**解读：** 2h exec 旧表 inflated ~2×；对齐后 2h+4bps 仍为正但不再代表 live。Promote 仍以 **1min+20bps** 为准。

## 决策

- [x] Research 默认 **1min exec** + 右边界 segment 窗口
- [x] 2h exec 保留为 legacy sensitivity（manifest 显式 `execution_timeframe: 2h`）
- [ ] 100ms smoke：需 `data/agg_data` aggTrades zip（CLI 已接入，无数据则 fail loud）
- [ ] spacing 放大：OOS 略改善（-0.04%），未过 promote 线；不作 archetype 默认变更

## 测试

- `tests/unit/test_subbar_replay.py`
- `tests/unit/test_chop_grid_exec_alignment.py`（SOL-style：subbar 不在 confirm 前 fill SHORT）
