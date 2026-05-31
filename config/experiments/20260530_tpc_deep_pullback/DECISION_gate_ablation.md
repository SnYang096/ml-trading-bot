# TPC gate locked 规则复验（20260531）

固定 **E2_or entry** + **prod prefilter**（不比较 prefilter）。对照 prod gate 里 0530 未单独验证的 locked 规则。

## 离线 plateau（`tpc_gate_locked_revalidate.yaml`）

- `skip_locked: false` 全规则扫描
- 单特征 lift：`vol_persistence`、`vol_leverage_asymmetry`
- 产出：`results/rd_loop/tpc_gate_locked_revalidate/quick_scan/`

## event_backtest ablation（`tpc_gate_ablation_grid.yaml`）

| variant | 改动 |
|---------|------|
| G0_gate_prod | 当前 prod gate（chop + bull vol×2） |
| G1_no_bull_vol_gates | 关闭 vol_persistence + vol_leverage bull |
| G2_no_chop_gate | 关闭 semantic_chop |
| G4_no_vol_persist_bull | 仅关闭 vol_persistence bull |
| G5_no_vol_lev_bull | 仅关闭 vol_leverage bull |

窗口：2023–2025 全段 + 2024H1 bull 子窗。

## 结论

TODO（grid + rd_loop 完成后填写）
