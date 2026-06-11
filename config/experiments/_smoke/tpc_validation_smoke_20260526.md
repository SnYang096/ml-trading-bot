# Tpc Validation Smoke

- **日期**: 2026-05-26
- **experiment_id**: `tpc_variant_grid_smoke`
- **template**: `default`
- **决策**: Promote **H** (fill after review)

## 1. 变体定义

| ID | strategies_root | 说明 |
|---|---|---|
| **H_recent_smoke** | `config/strategies` | _(fill)_ |
| **A_baseline_bull_2024** | `config/strategies` | _(fill)_ |
| **B_bull_2024** | `config/experiments/_smoke/variants/B_gate_only_chop_strategies` | _(fill)_ |
| **H_bull_2024** | `config/experiments/_smoke/variants/H_bull_conditional_vol_strategies` | _(fill)_ |
| **B_gate_only_chop** | `config/experiments/_smoke/variants/B_gate_only_chop_strategies` | _(fill)_ |
| **H_recent** | `config/experiments/_smoke/variants/H_bull_conditional_vol_strategies` | _(fill)_ |
| **BE_combo** | `config/experiments/_smoke/variants/BE_combo_strategies` | _(fill)_ |
| **C_chop_plus_evt** | `config/strategies` | _(fill)_ |
| **D_regime_strict** | `config/strategies` | _(fill)_ |
| **E_entry_v2** | `config/strategies` | _(fill)_ |

## 2. 双段回测结果

### 2.1 2024-01-01 → 2025-01-01

| 变体 | trades | totR | ret% | maxDD% | dir |
|---|---:|---:|---:|---:|---|
| A_baseline_bull_2024 | 159 | +17.64 | 1.74% | 8.64% | `/home/yin/trading/ml_trading_bot/results/tpc/experiments/A_baseline_bull_2024` |
| B_bull_2024 | 175 | +16.94 | -0.04% | 13.52% | `/home/yin/trading/ml_trading_bot/results/tpc/experiments/B_bull_2024` |
| H_bull_2024 | 168 | +16.30 | 2.87% | 7.57% | `/home/yin/trading/ml_trading_bot/results/tpc/experiments/H_bull_2024` |

### 2.2 2025-04-01 → 2026-04-01

| 变体 | trades | totR | ret% | maxDD% | dir |
|---|---:|---:|---:|---:|---|
| B_gate_only_chop | 178 | +59.83 | 21.82% | 5.23% | `/home/yin/trading/ml_trading_bot/results/tpc/experiments/B_gate_only_chop` |
| H_recent | 172 | +47.06 | 16.76% | 7.48% | `/home/yin/trading/ml_trading_bot/results/tpc/experiments/H_recent` |
| BE_combo | 159 | +46.36 | 16.52% | 6.25% | `/home/yin/trading/ml_trading_bot/results/tpc/experiments/BE_combo` |
| C_chop_plus_evt | 102 | +36.06 | 13.47% | 8.24% | `/home/yin/trading/ml_trading_bot/results/tpc/experiments/C_chop_plus_evt` |
| D_regime_strict | 145 | +49.34 | 16.70% | 8.76% | `/home/yin/trading/ml_trading_bot/results/tpc/experiments/D_regime_strict` |
| E_entry_v2 | 143 | +45.72 | 18.32% | 5.83% | `/home/yin/trading/ml_trading_bot/results/tpc/experiments/E_entry_v2` |

### 2.3 2026-03-01 → 2026-03-15

| 变体 | trades | totR | ret% | maxDD% | dir |
|---|---:|---:|---:|---:|---|
| H_recent_smoke | 11 | -0.72 | -0.72% | 0.72% | `/home/yin/trading/ml_trading_bot/results/tpc/experiments/_smoke_grid_H` |

## 2.3 按 side 分解（placeholder）

| 变体 | LONG totR | SHORT totR |
|---|---:|---:|
| _(fill from event_trades CSV)_ | | |

## 3. 离线 label / IC（placeholder）

- quick_layer_scan condition-set / ic-decay

## 4. 决策

- [ ] Promote variant: ___
- [ ] Reject reason: ___
