# Chop Grid Proxy Validation

- **日期**: 2026-05-26
- **experiment_id**: `chop_grid_validation_smoke`
- **template**: `c_semantic_proxy`

## 1. 变体定义

| ID | strategies_root | 说明 |
|---|---|---|
| **baseline_segments_reuse** | `config/strategies` | _(fill)_ |

## 2. 双段回测结果

### 2.1 2024-01-01 → 2026-05-11

| 变体 | trades | totR | ret% | maxDD% | dir |
|---|---:|---:|---:|---:|---|
| baseline_segments_reuse | 1259 | +0.39 | 38.89% | 4.85% | `/home/yin/trading/ml_trading_bot/results/chop_grid/check_current_20240101_20260511` |

## 3. 语义代理 vs C KPI（fill from segment_label parquet + quick_layer_scan）

> 输入：`results/<slug>/segments/grid_segments.csv` + features parquet → 
> `scripts/_build_grid_segment_labels.py` → segment-level labeled parquet → 
> `quick_layer_scan condition-set --label seg_total_r_over_dd|seg_adverse_break_rate|seg_maker_return_per_round`.

| entry_feature 候选 | n | seg_total_r_over_dd 中位 | seg_adverse_break_rate | seg_maker_return_per_round | |z| vs base |
|---|---:|---:|---:|---:|---:|
| **baseline_segments_reuse**（_fill_）| | | | | |

## 4. Plateau 宽度（fill from quick_layer_scan feature-plateau）

| 候选 | 开带 | 关带 | plateau 宽度 | base_succ |
|---|---:|---:|---:|---:|
| _(fill: 每个 entry_feature 的 max_semantic_chop_* plateau)_ | | | | |

## 5. 决策

- [ ] Promote `entry_feature` = ___ + plateau mid ___
- [ ] 拒绝理由 ___
- [ ] shadow 一个季度后再 live-deploy（同 §2.2.1 流程）
