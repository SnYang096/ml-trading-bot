# TPC 现网 gate 有用性验证（20260531）

**问题**：`gate.yaml` 里 locked 的 chop / bull vol 规则未走 0530 deep_pullback 同口径 scan；不宜用 IC 定阈。

**流程（严格 ① → ②）**：

1. **Phase 1 — label scan only**（本目录 `rd_loop_tpc_gate_validate.yaml`）
   - KPI：`success_no_rr_extreme`（与 `train_final_*_rr_extreme` parquet 一致）
   - `feature-plateau`：单特征 + \|z\|（对齐 0530 depth/PE）
   - `condition-set`：现网 deny 带（chop / vol bull 宽带）
   - `gate-plateau`：整规则 lift（`label_col: success_no_rr_extreme`，`skip_locked: false`）
   - **不跑** event_backtest

2. **Phase 2 — 因果复验**（`tpc_gate_validate_phase2_grid.yaml`，读完 Phase 1 再跑）
   - 固定 E2_or entry + prod prefilter
   - 仅 2–3 个 variant（G0 prod gate vs scan 建议）

```bash
# Phase 1
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260531_tpc_gate_validate/rd_loop_tpc_gate_validate.yaml

# Phase 2（人工 unlock 后）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260531_tpc_gate_validate/tpc_gate_validate_phase2_grid.yaml \
  --quiet-signal-logs
```

产物：`results/rd_loop/tpc_gate_validate/quick_scan/` + `DECISION.md`
