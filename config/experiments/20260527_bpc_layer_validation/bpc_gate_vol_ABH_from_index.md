# Bpc Gate Vol Abh

- **日期**: 2026-05-27
- **experiment_id**: `bpc_gate_vol_ABH_20260527`
- **template**: `default`

## 1. 变体定义

| ID | strategies_root | 说明 |
|---|---|---|
| **A_bull_2024** | `config/strategies` | _(fill)_ |
| **B_bull_2024** | `config_experiments/bpc_B_vol_off_strategies` | _(fill)_ |
| **H_bull_2024** | `config_experiments/bpc_H_bull_vol_strategies` | _(fill)_ |
| **no_breakout_bull_2024** | `config_experiments/bpc_no_breakout_strategies` | _(fill)_ |
| **A_recent** | `config/strategies` | _(fill)_ |
| **B_recent** | `config_experiments/bpc_B_vol_off_strategies` | _(fill)_ |
| **H_recent** | `config_experiments/bpc_H_bull_vol_strategies` | _(fill)_ |
| **no_breakout_recent** | `config_experiments/bpc_no_breakout_strategies` | _(fill)_ |

## 2. 双段回测结果

### 2.1 2024-01-01 → 2025-01-01

| 变体 | trades | totR | ret% | maxDD% | dir |
|---|---:|---:|---:|---:|---|
| A_bull_2024 | 27 | +16.85 | 7.02% | 5.93% | `/home/yin/trading/ml_trading_bot/results/bpc/experiments/A_bull_2024` |
| B_bull_2024 | 29 | +17.56 | 7.73% | 5.65% | `/home/yin/trading/ml_trading_bot/results/bpc/experiments/B_bull_2024` |
| H_bull_2024 | 28 | +16.81 | 6.98% | 5.97% | `/home/yin/trading/ml_trading_bot/results/bpc/experiments/H_bull_2024` |
| no_breakout_bull_2024 | 28 | +17.47 | 6.75% | 5.94% | `/home/yin/trading/ml_trading_bot/results/bpc/experiments/no_breakout_bull_2024` |

### 2.2 2025-04-01 → 2026-04-01

| 变体 | trades | totR | ret% | maxDD% | dir |
|---|---:|---:|---:|---:|---|
| A_recent | 25 | -1.22 | 2.74% | 5.87% | `/home/yin/trading/ml_trading_bot/results/bpc/experiments/A_recent` |
| B_recent | 17 | -4.18 | 2.15% | 3.03% | `/home/yin/trading/ml_trading_bot/results/bpc/experiments/B_recent` |
| H_recent | 14 | -5.59 | 0.36% | 3.78% | `/home/yin/trading/ml_trading_bot/results/bpc/experiments/H_recent` |
| no_breakout_recent | 25 | -1.22 | 2.74% | 5.87% | `/home/yin/trading/ml_trading_bot/results/bpc/experiments/no_breakout_recent` |

## 2.3 按 side 分解（placeholder）

| 变体 | LONG totR | SHORT totR |
|---|---:|---:|
| _(fill from event_trades CSV)_ | | |

## 3. 离线 label / IC（placeholder）

- quick_layer_scan condition-set / ic-decay

## 4. 决策

- [ ] Promote variant: ___
- [ ] Reject reason: ___
