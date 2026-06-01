# Tree IC target comparison: label vs forward_rr (small curated pools)

| 字段 | 值 |
|------|-----|
| 目录 | `20260601_1322_tree_forward_rr_ic_small_pool/` |
| 日期 | 2026-06-01 |
| 策略 | `fast_scalp` + `short_term_swing` |
| OOS | `recent_6m_oos` (2025-10-01 → 2026-03-31) |
| 决策 | [`DECISION.md`](DECISION.md) |

## 目的

1. 在小特征集合（当前 canonical ~20 列）上，用**相同 IC 工具**对比 `target=label` vs `target=forward_rr_hN`。
2. `fast_scalp` 与 `short_term_swing` 在**相同 train/OOS 数据**上并排比较。
3. 若小集合优于 wide 大池，分析是否因 math 类特征（hilbert/wpt/spectrum/evt/dtw…）导致过拟合。

## 实验矩阵

| Arm | Strategy | IC target | Slug |
|-----|----------|-----------|------|
| A1 | fast_scalp | `label` | `fast_scalp_label_ic_small` |
| A2 | fast_scalp | `forward_rr_h3` | `fast_scalp_forward_rr_ic_small` |
| B1 | short_term_swing | `label` | `short_term_swing_label_ic_small` |
| B2 | short_term_swing | `forward_rr_h20` | `short_term_swing_forward_rr_ic_small` |

训练目标：四臂均仍用 floored `label` 回归（仅 IC 选材 target 不同）。

## 跑法

```bash
PYTHONPATH=src:scripts:. python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260601_1322_tree_forward_rr_ic_small_pool/rd_loop_tree_forward_rr_comparison.yaml
```

特征族分析（跑完后）：

```bash
PYTHONPATH=src:scripts:. python scripts/research/analyze_feature_family_overfit.py \
  --manifest config/experiments/20260601_1322_tree_forward_rr_ic_small_pool/feature_family_manifest.yaml
```

## 产物

- `results/rd_loop/tree_forward_rr_ic_small_pool/`
- `results/train_final/tree_forward_rr_ic_small_pool/`
