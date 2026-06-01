# TPC Gate 最终干净判决 — DECISION（待跑完后填写）

**实验**: 20260601_1300_tpc_gate_canonical_g0_g1  
**Grid**: `tpc_gate_g0_vs_g1_canonical.yaml`  
**原则**: 见 `../LAYER_PROMOTION_CRITERIA.md`

---

## 运行摘要（跑完后自动填充）

| Segment              | Variant | Total R | maxDD (R) | CAGR    | Trades | Win% | Tail Contrib | Notes |
|----------------------|---------|---------|-----------|---------|--------|------|--------------|-------|
| bear_2022            | G0      |         |           |         |        |      |              |       |
| bear_2022            | G1      |         |           |         |        |      |              |       |
| bull_2023_2024       | G0      |         |           |         |        |      |              |       |
| bull_2023_2024       | G1      |         |           |         |        |      |              |       |
| recent_range_to_bear | G0      |         |           |         |        |      |              |       |
| recent_range_to_bear | G1      |         |           |         |        |      |              |       |

**Delta 总结**（G1 - G0）：
- bull_2023_2024: ...
- recent_range_to_bear: ...
- bear_2022: ...（预期中性）

## Promote 结论（按 LAYER_PROMOTION_CRITERIA.md 三条杠）

- [ ] G1 在 bull + recent 总 R 明显提升，maxDD 未恶化，逻辑清晰（两条 bull vol 中间带 overkill） → **推荐 lock G1 形态**
- [ ] 所有已 disabled 的 vol_* / EVT gate 规则在最终 archetypes/gate.yaml 中**物理删除**（不留历史 baggage）
- [ ] 同步 live/highcap 配置
- [ ] 更新主策略文档及本 DECISION.md 完整版

（历史证据已在 0530~0602 系列实验中充分展示，此次仅做 canonical 窗口最终确认。）

---

**跑完后删除本占位内容，替换为完整分析 + 最终 patch 链接。**
