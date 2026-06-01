# TPC regime 复验 + gate 扩展扫（20260601）

回应五项：

1. **Regime**：`|ema_1200_position|` 阈值 vs `ema_1200_slope_10` vs 其它宏观候选（label + IC）
2. **vol_leverage**：现网中间带 → 候选 **bull 低尾单边 deny**（`vla < τ`）
3. **evt_var_99**：prod 窄带（当前 disabled）condition-set + lift
4. **path_efficiency**：`>0.15 deny` 是否该去掉（condition-set 对照）
5. **其它 gate 候选**：`bb_width`、`vol_clustering`、`atr_percentile`、`trend_r2_20`

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260601_tpc_regime_gate_extend/rd_loop_tpc_regime_gate_extend.yaml
```

Phase 2（regime / gate 变体 backtest）见 `tpc_regime_gate_extend_phase2_grid.yaml`（scan 后再跑）。
