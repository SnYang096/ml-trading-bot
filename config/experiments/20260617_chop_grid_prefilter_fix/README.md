# chop_grid Prefilter Fix — 回测 Bug 修复 + 特征增强实验

## 背景

回测 `_lookup()` 只传递 6 个特征 key 给引擎，而 live 系统传递全部 feature-bus 列。
导致回测中 box_pos_60 prefilter 和 stable-box 阻断**完全失效**，回测结果失真。

### P0 Fix（已完成）

| 文件                                    | 改动                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `scripts/diagnose_crf_edge.py`          | `build_symbol_dataset` 现在保留所有 windowed box 列 (60/120/240/480/1200) |
| `scripts/backtest_multileg_timeline.py` | `_lookup` 现在传递 DataFrame 全部列（与 live `**raw` 一致）               |

### 影响

| 字段                | 修复前（默认值）  | 修复后（真实值）   |
| ------------------- | ----------------- | ------------------ |
| `box_pos_60`        | `0.5`（永远通过） | 真实值（可能拒绝） |
| `box_stability_120` | `None` → `False`  | 真实值（可阻断）   |
| 传递 keys 数量      | 6                 | 60+                |

## Phase checklist

| Phase | 内容                   | 状态                                   |
| ----- | ---------------------- | -------------------------------------- |
| A0    | 基线回测（P0 fix 后）  | ✅ 完成 — 全 segment 盈利，maxDD ≤9.21% |
| A1    | 多 Box 共识过滤        | 待定                                   |
| A2    | compression_score 入场 | 待定                                   |
| A3    | Wall 静态信号          | 待定                                   |
| A4    | 组合最佳配置           | 待定                                   |
| B     | Shadow 实盘验证        | 待定                                   |
| C     | 实盘上线               | 待定                                   |

## A0 — 基线回测

验证 P0 fix 后的真实 chop_grid 表现。

```bash
bash config/experiments/20260617_chop_grid_prefilter_fix/run_a0_baseline.sh
```

**Segments:** bear_2022, bull_2023_2024, recent_range_to_bear, recent_6m_oos

**产物:**
```
results/chop_grid/experiments/prefilter_fix_20260617/a0_baseline/
├── preload.pkl
├── bear_2022/summary.json
├── bull_2023_2024/summary.json
├── recent_range_to_bear/summary.json
├── recent_6m_oos/summary.json
└── joint/summary.json
```

## 结构

```
20260617_chop_grid_prefilter_fix/
├── README.md
├── DECISION.md
├── variant_grid_a0.yaml
├── run_a0_baseline.sh
└── variants/
    └── baseline/
        ├── meta.yaml
        └── archetypes -> live/highcap chop_grid archetypes (symlink)
```
