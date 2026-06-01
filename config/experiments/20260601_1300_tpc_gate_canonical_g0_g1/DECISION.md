# TPC Gate 最终干净判决 — DECISION

**实验**: `20260601_1300_tpc_gate_canonical_g0_g1`  
**Grid**: `tpc_gate_g0_vs_g1_canonical.yaml`  
**原则**: [`../LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md)  
**结果目录**: `results/tpc/experiments/gate_canonical_final/`  
**运行**: 2026-06-01，exit 0，BTC+ETH，canonical 三阶段（`config/market_segment.yaml`）

---

## 分阶段结果（G0 prod vs G1 关 bull vol×2）

| Segment | Variant | Total R | maxDD | CAGR | Trades |
|---------|---------|---------|-------|------|--------|
| bear_2022 | G0 | -0.12 | 4.0% | -0.64% | 30 |
| bear_2022 | G1 | -0.12 | 4.0% | -0.64% | 30 |
| bull_2023_2024 | G0 | 6.71 | 6.3% | 3.17% | 44 |
| bull_2023_2024 | G1 | **7.88** | 6.5% | 3.78% | 48 |
| recent_range_to_bear | G0 | 4.33 | 3.4% | 3.26% | 29 |
| recent_range_to_bear | G1 | **9.09** | 4.0% | 7.09% | 34 |

**G1 − G0（按段）**

| Segment | Δ Total R | maxDD 变化 |
|---------|-----------|------------|
| bear_2022 | 0 | 持平 |
| bull_2023_2024 | **+1.17 R** | +0.2pp（略差） |
| recent_range_to_bear | **+4.76 R** | +0.6pp（略差） |
| **三阶段合计** | **+5.93 R** | 可接受 |

---

## Promote 结论（三条杠）

| 条件 | 判定 |
|------|------|
| 总 R 明显提升 | ✅ bull、recent 显著更好；bear 中性 |
| maxDD 不恶化 | ⚠️ bull/recent 略升 0.2–0.6pp，相对 +5.93R 可接受 |
| 逻辑可解释 | ✅ bull-only vol 中间带在事件回测中 overkill |

**决定：lock G1 形态并写入 prod**

- `config/strategies/tpc/archetypes/gate.yaml`：仅保留 `gate_tpc_semantic_chop_high`；**物理删除** vol_persistence / vol_leverage / EVT 规则
- `live/highcap/config/strategies/tpc/archetypes/gate.yaml`：已同步
- `features_gate.yaml`：移除仅服务于已删 gate 的 vol_* 白名单项

---

## 已证伪、不再进入 prod 的 gate 形态

- bull vol_persistence / vol_leverage **中间带**（G0 形态）
- vol_leverage 极低尾单边（G6/G10）
- vol_persistence 全 regime（G9）
- EVT 窄带（G7，已删）

Phase 1 label/IC 对 vol_* 的 lift **不**构成 promote 依据；以本 grid 及 0530–0602 系列 event_backtest 为准。

---

## 后续

- 无需再跑 G10 / monotonic Phase 2
- 监控：漂移见 `docs/strategy/漂移监控_mlbot_monitor_CN.md`；gate 层仅 chop 一条硬规则
