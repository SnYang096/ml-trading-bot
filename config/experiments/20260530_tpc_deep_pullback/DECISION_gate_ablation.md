# TPC gate locked 规则复验（20260531）

固定 **E2_or entry** + **prod prefilter**。产物：`results/tpc/experiments/gate_ablation/`

## 分段口径

此前 grid 用手写日期（含 ad-hoc 2024H1），**未**读 `config/market_segment.yaml`。  
canonical 分段见该文件；variant-grid 现已支持 `segment:` / `segment_matrix`（`scripts/event_backtest/market_segment.py`）。

按分段重跑 G0/G1：`config/experiments/20260601_tpc_gate_validate/tpc_gate_G0_G1_by_market_segment.yaml`  
→ `results/tpc/experiments/gate_by_segment/{G0,G1}/<segment_id>/`

## 全窗 2023–2025（BTC+ETH，= bull_2023_2024 窗）

| variant | 改动 | trades | totR | CAGR | maxDD |
|---------|------|--------|------|------|-------|
| G0 | prod gate | 44 | +6.71R | 3.17% | **−6.32%** |
| **G1** | 关 bull vol×2 | 48 | **+7.88R** | 3.78% | −6.52% |
| G2 | 关 chop | 63 | +8.54R | 3.81% | −6.72% |
| G4 | 仅关 vol_persist bull | 46 | +6.04R | 2.82% | −4.67% |
| G5 | 仅关 vol_lev bull | 49 | +4.97R | 2.27% | −9.07% |

→ **G1 最优 totR**（+1.17R vs G0）；**G4** DD 更好但 R 略低；**G5 单独关 vol_lev 变差**（勿只关 lev）。  
→ **G2** 多 R 但笔数 +43%、DD 更差 — 不单关 chop promote。

## 2024H1 bull 子窗

| variant | trades | totR | maxDD |
|---------|--------|------|-------|
| G0 | 15 | +2.82R | −4.58% |
| **G1** | 14 | **+7.58R** | **−3.79%** |

→ bull 子窗上 **关 bull vol 明显更优**（与「vol 过杀牛市」一致）。

## Promote 草案

- `gate.yaml`：**disabled** `vol_persistence` + `vol_leverage` bull 中间带（保留 chop；EVT 仍关）。
- 等 0601 **G6/G7/G9** 后再决定是否改 vol_lev 形状或加 EVT。
