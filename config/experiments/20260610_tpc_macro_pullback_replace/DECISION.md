# TPC macro_pullback replace — DECISION（待填）

> **方法学**：τ 来自 Phase 1 `scan_tpc_pullback_lookback.py`（见 [`README.md`](README.md)）。流程符合 [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) §标准 R&D 阶段。

## Phase 2 定参（2026-06-04）

见 [`PHASE1_REPORT.md`](PHASE1_REPORT.md)。

- **M_replace_L15_S12**：label bull macro≥0.15 \|z\|=3.5 + OHLC τ 支持 → **主候选**
- **M_replace_L20_S15**：long≥0.20 label \|z\|<2 → 仅作对照
- Grid 全量重跑（作废旧 partial）

## 假设

macro `tpc_macro_pullback_pct` prefilter **替代** prod `depth<=0.85` 后，在 canonical 三阶段上相对 E0_prod：

1. **bull_2023_2024** trading map 入场落在大回调区（非小震荡）
2. Total R 提升或 trade-off 可解释（R vs maxDD vs 笔数）

## 结果（2026-06-09 grid 全量）

| variant | bear_2022 R | bull R | recent R | sum R | worst maxDD | trades |
|---------|-------------|--------|----------|-------|-------------|--------|
| E0_prod | +4.47 | +18.82 | +13.44 | **+36.73** | −12.1% | 208 |
| M_replace_L15_S12 | −0.50 | −6.26 | +1.82 | −4.94 | −12.0% | 63 |
| M_replace_L20_S15 | −0.91 | −6.35 | +2.04 | −5.22 | −6.5% | 32 |

**读数**：macro 替代 depth 两变体 **sum R 均为负**，显著劣于 E0_prod；笔数大幅收缩（63/32 vs 208）。L15 bull 段尤其差（−6.26R）。

## Promote

**不 promote** `M_replace_*` → prod。macro prefilter 替代 depth 在本 grid 未过 Total R 杠；trading map 待 [`run_trading_maps.sh`](run_trading_maps.sh) 完成后人审 bull 语义（当前数字已足够否决 promote）。

## Follow-up：`M_add_L15_S12`（macro AND depth，待跑）

- 变体树：`config_experiments/tpc_macro_add_L15_S12_strategies/`
- Grid：[`tpc_macro_pullback_add_grid.yaml`](tpc_macro_pullback_add_grid.yaml) → `results/tpc/experiments/macro_pullback_add_20260610/`
- 假设：大回撤背景（macro≥τ）+ 当根深回踩（depth≤0.85）交集，相对 E0 提升 bull R且笔数不过度塌缩

| variant | bear R | bull R | recent R | sum R | worst maxDD | trades |
|---------|--------|--------|----------|-------|-------------|--------|
| E0_prod | | | | | | |
| M_add_L15_S12 | | | | | | |
