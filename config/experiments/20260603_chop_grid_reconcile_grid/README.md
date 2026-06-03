# chop_grid — fee × exec × universe reconciliation grid

**目的：** 解释「旧实验看起来很好、当前 OOS 很差」—— 逐项隔离 **执行粒度**、**成本假设**、**币池** 与 **指标口径**（pooled vs timeline）。

## 为何旧数字容易误读

| 来源 | 典型设置 |  headline 数字 | 问题 |
|------|----------|---------------|------|
| `20260526_chop_grid_semantic_proxy` | 2h-only、6 sym、低 fee 可能 | **+38.89% ret** | `totR=+0.39` × 100，且为 **pooled**（≈6× 组合） |
| `segment_validate_20260603` prod | 1min exec、5 sym、**20bps** | **-0.75% timeline** | canonical 组合口径，成本更保守 |

本 grid 在 **同一 OOS 窗**（`recent_6m_oos` 2025-10→2026-03）上跑 2×2×2 矩阵，全部输出 **timeline `return_pct`** + pooled 对照。

## 矩阵（OOS）

| exec | fee | 5 sym | 6 sym |
|------|-----|-------|-------|
| **2h** | 4bps | `2h_4bps_5sym` | `2h_4bps_6sym` ← 最接近旧 proxy |
| **2h** | 20bps | `2h_20bps_5sym` | `2h_20bps_6sym` |
| **1min** | 4bps | `1min_4bps_5sym` | `1min_4bps_6sym` |
| **1min** | 20bps | `1min_20bps_5sym` ← **当前 prod** | `1min_20bps_6sym` |

Replenish：研究默认 **unlimited**（与 prior segment validate 一致）。补挂对照见 [`../20260603_chop_grid_replenish_ablation/`](../20260603_chop_grid_replenish_ablation/)。

## 跑法

```bash
python scripts/experiment_chop_grid_reconcile_grid.py \
  --manifest config/experiments/20260603_chop_grid_reconcile_grid/grid_oos.yaml

# 单格
python scripts/experiment_chop_grid_reconcile_grid.py --cells 2h_4bps_6sym,1min_20bps_5sym
```

产物：`results/chop_grid/experiments/reconcile_grid_20260603/oos/reconcile_summary.csv`

## 结论

见 [`DECISION.md`](DECISION.md)（跑批后更新）。
