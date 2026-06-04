# TPC entry semantic validate — 决策记录

**实验 ID:** `tpc_entry_semantic_validate_20260604`  
**状态:** 待跑

## 假设

| ID | 假设 | 变体 |
|----|------|------|
| S50 | 仅 **depth > 0.5** 的深回踩才开仓，可消除 SOL 等「贴极值追」 | `S50_depth_gt50` |
| S51 | S50 + 价格可 **略低于 EMA1200**（prefilter ema≥-0.10，direction inner=-0.10） | `S51_depth_gt50_ema_near` |
| E1 | 浅下界 0.15 即可显著减追高，不必等到 0.5 | `E1_depth_ge15` |
| E2 | `bars_since_local_high` 防贴顶，与 depth 带互补 | `E2_anti_chase` |
| E3 | 高 path_efficiency 延续区 deny，与 BPC 划界 | `E3_gate_pe` |
| E4 | turbo 宽 ladder 执行在相同入场下是否改善 R/DD | `E4_turbo_exec` |

## Promote 检查（LAYER_PROMOTION_CRITERIA）

- [ ] canonical 三阶段：**Total R ↑** 且 **maxDD 不恶化**
- [ ] 全窗 highcap：SOL/BTC 追高类交易减少且总 R 不崩
- [ ] 逻辑可解释（非单段过拟合）

## 结果表（跑完后填写）

| 变体 | bear_2022 R | bull_2023_2024 R | recent R | sum R | maxDD | 备注 |
|------|-------------|------------------|----------|-------|-------|------|
| E0_prod | | | | | | |
| S50_depth_gt50 | | | | | | |
| S51_depth_gt50_ema_near | | | | | | |
| E1_depth_ge15 | | | | | | |
| E2_anti_chase | | | | | | |
| E3_gate_pe | | | | | | |
| E4_turbo_exec | | | | | | |
