# TPC Regime ADX — Phase 1/2: 假设扫描 + 定参

**实验 ID**: 20260610_tpc_regime_adx_phase1  
**目的**: 验证 ADX 作为 regime 自适应退出指标的统计有效性，确定最优参数。

## Phase 1 结论

| 指标 | IC(20b) | ADX>阈_fwd20 | ADX≤阈_fwd20 | 分离度 |
|------|:-------:|:----------:|:----------:|:-----:|
| ADX(14) | 0.008 | +0.10% | +0.10% | 0bps ❌ |
| **ADX(50)** | **0.043** | **+0.81%** | **-0.01%** | **82bps ✅** |
| ADX(100) | 0.034 | +0.85% | +0.06% | 79bps ⚠️ |
| EMA1200_pos | 0.036 | -0.00% | +0.11% | -11bps ❌ |
| SemanticChop | -0.018 | +0.05% | +0.23% | -18bps ❌ |

**Winner: ADX(50, 120T)**
- 最高 IC(20b)=0.043
- 最佳 regime 分离度：趋势市 fwd20=+0.81%，震荡市 fwd20=-0.01%
- ADX>25 占比 12-15%（合理，不频繁切换）
- ADX(100) 分离度类似但 >25 仅 2-8%，太严格

## Phase 2 定参

| 参数 | 值 | 理由 |
|------|:--:|------|
| regime 指标 | ADX(50) | IC 最高，分离度最大 |
| bull 阈值 | ADX > 25 | 经典 Wilder 阈值，不过拟合 |
| bear 阈值 | ADX ≤ 20 | 5 点死区防抖 |
| EMA 辅助 | ema_1200_pos > 0.10 AND | 防止单边下跌时 ADX 高但不应做多 |
| 特征周期 | 50 bar (120T) | ≈4 交易日，匹配 TPC 持仓 |

## Phase 3 计划

Grid 对比:
- E9: trailing only (baseline)
- E13: structural only  
- E21: ema_1200_position threshold 0.18
- E22: ADX(50) > 25 → structural, else trailing

## 数据

- BTCUSDT 120T, 18441 bars, 2022-01 → 2026-04
- 训练数据: train_final_20260604_rd_rerun
- Phase 1 输出: `phase1_scan.json`
