# DECISION — chop_grid Prefilter Fix

## 假设

| ID  | 假设                                          | Phase A 证据                                        | 决策                            |
| --- | --------------------------------------------- | --------------------------------------------------- | ------------------------------- |
| H0  | P0 fix 后回测结果应比修复前更保守（入场更少） | ✅ A0 完成 — 4 segment 全盈利，maxDD ≤9.21%，无 halt | confirmed — P0 fix 不影响盈利性 |
| H1  | 多 Box 共识过滤可减少假信号、提高胜率         | A1 实验                                             | pending                         |
| H2  | box_compression_score 可捕捉高胜率压缩形态    | A2 实验                                             | pending                         |
| H3  | Wall 静态信号可增强入场时机                   | A3 实验                                             | pending                         |

## A0 基线结果

**P0 fix 后首次完整回测** — backtest_multileg_timeline --no-trend，compound sizing。
5 symbols (BTC/ETH/SOL/BNB/XRP)，constitution: gross≤27000, daily≤600, dd≤20%, sym≤6。

| Segment              | 期间            | 总回报   | MaxDD  | 入场次数 | 拒绝次数 | Fee/Notional  |
| -------------------- | --------------- | -------- | ------ | -------- | -------- | ------------- |
| bear_2022            | 2022-01→2023-11 | +10,962% | -9.21% | 32,967   | 36       | $116K / $291M |
| bull_2023_2024       | 2023-06→2025-01 | +2,952%  | -3.12% | 28,805   | 0        | $29K / $74M   |
| recent_range_to_bear | 2025-01→2026-05 | +1,493%  | -3.98% | 25,285   | 6        | $19K / $48M   |
| recent_6m_oos        | 2025-12→2026-05 | +90.25%  | -3.98% | 8,945    | 0        | $1.2K / $3M   |

**Key observations:**
- P0 fix 后 prefilter 正确工作：bear_2022 有 36 笔拒绝（修复前为 0），确认 box_pos_60 真实过滤
- MaxDD 全部 ≤9.21%，远低于 20% 硬限制
- 所有 segment 无 halt
- 最弱 segment (recent_6m_oos) 仍稳定盈利 +90%/6mo
- 没有 Sharpe 计算（timeline backtest 未输出 Sharpe，需后续补充）

## Promote

- [ ] 三条杠（LAYER_PROMOTION_CRITERIA §1）
- [ ] `monitor_bundle/smoke_report.json` OK
- [ ] `mlbot research promote-baseline` 完成
