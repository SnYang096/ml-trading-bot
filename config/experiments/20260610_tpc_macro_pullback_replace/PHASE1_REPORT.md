# TPC macro_pullback — Phase 1 报告（2026-06-04）

## 数据

| 项 | 值 |
|----|-----|
| Parquet | `results/train_final/tpc/train_final_20260604_rd_rerun/tpc/features_labeled.parquet` |
| 特征集 | `features_scan_phase1.yaml`（含 macro + soft_phase + atr） |
| 标的 | BTCUSDT, SOLUSDT |
| 窗 | 2022-01-01 — 2026-04-01 |
| 扫描 | `results/rd_loop/tpc_macro_label_20260610/quick_scan/` |
| OHLC 对照 | `results/tpc/research/macro_pullback_scan_20260609/` |

## Label scan 要点

### Bull（`ema>=0.10`, `chop<=0.40`, n=1683）

| 扫描 | 关键格点 | \|z\| | 解读 |
|------|----------|------|------|
| `macro_long_plateau` | long≥0.12 | 4.51 | succ 66.1% vs base 52.2% |
| `macro_long_plateau` | long≥0.15 | 3.50 | succ 67.2% |
| `macro_long_plateau` | long≥0.20 | 0.63 | **无显著 plateau** |
| `macro_vs_depth` | macro_L15 | 3.50 | Δpp **+15.0** vs base |
| `macro_vs_depth` | prod depth≤0.85 | 1.56 | Δpp −0.6（label 上非瓶颈） |

### Bear（`ema<=-0.10`, n=1468）

| 扫描 | 关键格点 | \|z\| | 解读 |
|------|----------|------|------|
| `macro_short_plateau` | short≥0.12 | 0.19 | 边缘 |
| `macro_short_plateau` | short≥0.25 | 3.97 | 高阈样本少（n=176） |

### OHLC 扫描对照

- 推荐 macro τ：**long≥0.16, short≥0.12**（full 窗）
- 实验变体 **L15/S12** 与 label bull≥0.15 一致；**L20/S15** 的 long≥0.20 **label 不支持**

## Phase 2 定参

| 变体 | 决策 |
|------|------|
| **M_replace_L15_S12** | **保留** — label + OHLC 均支持 |
| **M_replace_L20_S15** | **保留作对照** — label long≥0.20 无显著性 |
| E0_prod | 基线不变 |

**不改**变体树阈值；grid 按现有 `tpc_macro_pullback_replace_grid.yaml` 全量重跑。

## Phase 3

- 作废旧 partial runs，重跑 3×3 segment grid + trading map。
