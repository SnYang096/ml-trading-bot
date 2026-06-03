# chop_grid — post-TP replenish ablation (2026-06-03)

**背景：** 研究曾用 `null`（无限补挂）；live prod 与 20260526 sweep 推荐 **`max_replenish=1`**。segment validate 20260603 在 unlimited 下 OOS **-0.75%**。

**目标：** 关闭 TP 后补挂（`max_replenish=0`），看能否接近历史较好窗口（如 20260526 proxy +38% pooled 口径需对照解读）。

## 变体

| ID | CLI | 含义 | output_dir |
|----|-----|------|------------|
| `replenish_unlimited` | `--max-replenish-per-level null` | TP 后同档无限补挂（历史研究对照） | `.../replenish_unlimited/` |
| `replenish_off` | `--max-replenish-per-level 0` | **禁用** TP 后补挂（legacy one-shot） | `.../replenish_off/` |
| `replenish_live` | _(archetype default=1)_ | **当前默认** — 每档 TP 后最多补挂 1 次 | `.../replenish_live/` |

配置语义见 [`docs/experiments/chop_grid_replenish_sweep_20260526.md`](../../../docs/experiments/chop_grid_replenish_sweep_20260526.md)。

## 跑法（单段 smoke：recent_6m_oos）

```bash
BASE=results/chop_grid/experiments/replenish_ablation_20260603
CFG=config/strategies/chop_grid/research/calibrate_roll.default.yaml
SYM=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT
FWD=(--config "$CFG" --symbols "$SYM" --timeframe 2h --execution-timeframe 1min
     --initial-capital 10000 --no-maps)

# unlimited（与 segment_validate_20260603_timeline 相同 replenish 语义）
python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/replenish_unlimited" --segments recent_6m_oos -- "${FWD[@]}"

# 关闭 TP 后补挂
python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/replenish_off" --segments recent_6m_oos -- \
  "${FWD[@]}" --max-replenish-per-level 0

# live prod 对齐
python scripts/experiment_chop_grid_market_segment.py \
  --out-root "$BASE/replenish_live" --segments recent_6m_oos -- \
  "${FWD[@]}" --max-replenish-per-level 1
```

四段全量：去掉 `--segments recent_6m_oos`，或分别改 `segment_matrix` manifest。

## 指标

Timeline 组合口径（与 [`../20260603_multileg_segment_validate/METRICS.md`](../20260603_multileg_segment_validate/METRICS.md) 一致）。对比时额外记录 segment 级 `replenish_trades`（若需可从 `grid_segments.csv` 汇总）。

## 结论

见 [`DECISION.md`](DECISION.md)。
