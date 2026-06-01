# TPC gate 单调单边 condition-set（20260602）

**问题**：中间带 deny 已证伪/过杀；是否应改为 **单调单边** gate？  
**方法**：`condition-set` + `feature-plateau`（`success_no_rr_extreme`），不跑 backtest。

**判读**（与 0531 一致）：

- 若 **deny 区** `succ_in` **低于** `succ_out`，且 **Δpp ≤ −0.5pp、|z| ≥ 2** → 单调 deny **有 label 价值**，可进 Phase 2 event_backtest。
- 若 hit 侧 **更好**（Δpp > 0）→ 该方向 **反了**，不能作 deny。
- 中间带仅作对照，不 promote。

```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260602_tpc_gate_monotonic_validate/rd_loop_tpc_gate_monotonic.yaml
```

产物：`results/rd_loop/tpc_gate_monotonic/quick_scan/report.html`
