# TPC 斜率分符号 × 分日历窗 label 验证

- **Parquet**: `train_final_20260523_122438_rr_extreme/tpc/features_labeled.parquet`
- **Filter**: `tpc_semantic_chop<=0.4`
- **问题**: 负斜率是否在「牛」「熊」两段日历窗都好？

## 结论（label）

**否。** 负斜率 **不是** 牛熊两窗都好；两段符号 **相反**：

| 日历窗 | 更好的一侧 | slope_down Δpp | slope_up Δpp | H_and_slope_up Δpp |
|---|---|---:|---:|---:|
| **2024-01→2025-01**（bull 样本） | **正斜率** | **-6.52** \|z\|=2.61 | **+2.11** | **+2.19** |
| **2025-04→2026-04**（recent） | 略偏 **正斜率** | +9.86 | +9.26 | **+10.06**（最好） |

全样本曾见 `slope_down +1.60pp`：是 **两窗平均掩盖** 了 bull 窗内负斜率更差。

→ **不**建议生产 regime 改为「仅负斜率」或「对称 |slope|」（F' backtest 已弱于 H）。

## 表：2024 日历 bull 子样本（n=5882）

| condition | n | succ_in | Δpp |
|---|---:|---:|---:|
| H | 2258 | 54.56% | -0.88 |
| slope_down | 370 | 48.92% | **-6.52** |
| slope_up | 1503 | 57.55% | **+2.11** |
| H ∧ slope_down | 369 | 49.05% | -6.39 |
| H ∧ slope_up | 1475 | 57.63% | **+2.19** |

## 表：2025-04→2026-04 recent 子样本（n=4670）

| condition | n | succ_in | Δpp |
|---|---:|---:|---:|
| H | 1775 | 65.13% | +8.15 |
| slope_down | 953 | 66.84% | +9.86 |
| slope_up | 554 | 66.25% | +9.26 |
| H ∧ slope_down | 951 | 66.77% | +9.79 |
| H ∧ slope_up | 540 | 67.04% | **+10.06** |

Recent 上负、正斜率 **都** 优于纯 H；**正斜率略优**，且 H∧slope_up 样本更少、label 最高。

## 与 event_backtest 的关系

- **H（生产）** / **Fp（\|slope\|）** 双段 R 已跑：Fp totR < H。
- 本扫说明：即便 recent label 喜欢 slope，**bull 窗负斜率 hurt** → 单腿负斜率 regime 难双段 Pareto。
- 若仍要 R 验证：可跑 `tpc_regime_slope_up_only` vs `slope_down_only` grid（非必须）。

## BPC 要不要同样验证？

**要分开看，不能照搬 TPC 结论。**

| | TPC | BPC |
|---|---|---|
| `\|ema\|>=0.10` | bull label 混合 | **bull -5.7pp，bear +3.3pp**（已 rd_loop） |
| 2024 日历窗 slope_down | 差 | n≈0（子样本极少） |
| Regime 已做 | gate H | box→EMA；**ema_slope** grid 略优未 promote |

BPC 应用 **bear / position 分侧** 做 regime，而不是 TPC 式「全样本负斜率更好」。斜率分窗应用 **全量 parquet 日期范围** 重扫（当前 BPC parquet 在 2025-04→2026-04 日历窗 n=0）。
