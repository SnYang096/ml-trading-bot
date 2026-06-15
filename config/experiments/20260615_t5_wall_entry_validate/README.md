# T5α wall entry — Phase 3 ablation

**假设**：`wall_nearest_dist_atr` 在 TPC entry 层有因果增益（scan Phase 1c）。

| 变体 | entry 改动 |
|------|------------|
| **E0_prod** | prod baseline（vol_confirm OR） |
| **W1_bull_wall2** | AND + 多单 `wall_dist≤2` |
| **W2_asym_wall** | AND + 多 `≤2` / 空 `≤2.5` |
| **W4_sym_wall2** | AND + 双向对称 `≤2`（对照） |

## 准备变体树

从 prod 复制 **`tpc/` 包**（~17 个 yaml），再 patch `entry_filters` + `features`；不复制 bpc/chop_grid 等无关策略。

```bash
PYTHONPATH=src:scripts python scripts/research/prepare_t5_wall_entry_snapshots.py
```

## Phase 3 backtest

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260615_t5_wall_entry_validate/t5_wall_entry_grid.yaml \
  --quiet-signal-logs
```

产物：`results/tpc/experiments/t5_wall_entry_20260615/<variant>/<segment>/capital_report.json`

晋升：[`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) 三条杠。
