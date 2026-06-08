# BPC lookback + box-retest — DECISION（待填）

## 假设

拉长 BPC `lookback_breakout`（120/240）并在 L120 上叠加 **box 突破 + depth 下界**，相对 B0_prod：

1. 减少 trading map 上「贴顶追高」入场
2. 更多落在压缩区突破后的回测/延续段
3. canonical 三阶段 trade-off 可接受（R / maxDD / 笔数）

## 结果（跑完后填）

| variant | bear R | bull R | recent R | sum R | worst maxDD | trades |
|---------|--------|--------|----------|-------|-------------|--------|
| B0_prod | | | | | | |
| B_L120 | | | | | | |
| B_L240 | | | | | | |
| B_L120_retest | | | | | | |

## Promote

未看图 + 未过 LAYER_PROMOTION_CRITERIA 前三阶段，**不 promote prod**。
