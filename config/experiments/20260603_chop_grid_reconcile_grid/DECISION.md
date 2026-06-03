# chop_grid reconcile grid — OOS 判决

**日期：** 2026-06-03  
**窗：** `recent_6m_oos`（2025-10-01 → 2026-03-31）  
**产物：** `results/chop_grid/experiments/reconcile_grid_20260603/oos/reconcile_summary.csv`

## 结果矩阵（timeline `return_pct`）

| cell_id | exec | fee | n | **timeline** | pooled（旧口径） | trades | seg_win% |
|---------|------|-----|--:|-------------:|-----------------:|-------:|---------:|
| **2h_4bps_6sym** | 2h | 4 | 6 | **+12.90%** | 77.4% | 389 | 71.6% |
| 2h_4bps_5sym | 2h | 4 | 5 | +12.66% | 63.3% | 322 | 71.4% |
| 1min_4bps_6sym | 1min | 4 | 6 | +6.38% | 38.3% | 275 | 55.9% |
| 1min_4bps_5sym | 1min | 4 | 5 | +5.80% | 29.0% | 220 | 54.6% |
| 2h_20bps_6sym | 2h | 20+fund20 | 6 | +4.30% | 25.8% | 389 | 57.2% |
| 2h_20bps_5sym | 2h | 20+fund20 | 5 | +4.10% | 20.5% | 322 | 56.2% |
| 1min_20bps_6sym | 1min | 20+fund20 | 6 | -0.36% | -2.1% | 275 | 39.2% |
| **1min_20bps_5sym** | 1min | 20+fund20 | 5 | **-0.75%** | -3.7% | 220 | 37.3% |

## 为何「原来看起来很好」

1. **口径：** 20260526 proxy 的 **+38.89%** 是 pooled / `totR×100` 风格；本窗最接近的旧设定 **2h+4bps+6币 pooled=77%**，但 **timeline 只有 +13%** — 策略并未「6 个月 +77% 组合」。
2. **执行：** 同一 fee 下 **2h → 1min** 约 **减半** edge（例：4bps 5币 12.7% → 5.8%）。
3. **成本：** **4bps → 20bps + funding 20** 再砍一半以上；prod 栈（1min+20+funding）OOS **-0.75%**。
4. **窗长：** 旧 proxy 用 **16 个月** bull/range；OOS 仅 **6 个月** 近期 chop 环境，本身更难。

**结论：策略在 OOS 上并非全面失效；当前 prod 研究 profile（1min 执行 + 20bps 全成本）把 edge 磨没。**

## 决策

- [ ] segment validate 报告应并列 **2h/4bps 对照列** 与 prod 列，避免只看 prod 误判「策略坏了」
- [ ] promote 前：决定 live 用 2h 执行还是接受 1min+真实 fee 的更低期望
- [ ] 长窗（2024-01→2026-05）复跑 `2h_4bps_6sym` 对齐 20260526 proxy 窗
