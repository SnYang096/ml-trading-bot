# BPC Entry v2 实验（对标 TPC E）

- **日期**: 2026-05-27
- **Grid**: `config/experiments/bpc_entry_v2_grid.yaml`
- **Scan**: `results/rd_loop/bpc_entry/quick_scan/`
- **决策**: **不 promote** — 维持生产 `bpc_orderflow_continuation`

## 变体

| ID | entry_filters |
|---|---|
| **baseline** | `bpc_cvd_absorption>=1` + `bpc_recovery_strength>=0.25` |
| **v2** | OR(`bpc_vol_compression>=0.5` ∧ `bpc_vol_ratio<=1.08`, `vp_absorption_score<=0.11`) |

Gate / regime / prefilter：与 `config/strategies` 一致。

## Event backtest（6 币）

| 窗 | baseline | v2 | Δ totR |
|---|---:|---:|---:|
| 2024 bull | 29 / **+15.52** | 25 / +6.21 | **-9.31** |
| 2025–26 recent | 17 / **+5.89** | 16 / **-4.18** | **-10.07** |

→ v2 两段均显著差于 baseline；与 TPC E（未 adopt）同型结论。

## Label scan（chop<=0.40）

- `bpc_vol_compression>=0.5`：succ +3～5pp（\|z\|≈5）— label 支持 v2
- `bpc_vol_ratio<=1.08`：succ +4～5pp（\|z\|≈5）
- **label ≠ R**：须以本表 backtest 为准

## 复现

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/rd_loop_bpc_entry.yaml
```
