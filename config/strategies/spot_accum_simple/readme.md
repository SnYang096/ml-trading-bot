# spot_accum_simple

简化版 A 层现货吸筹（生产默认）。原版 `spot_accum` 已归档至 `config/strategies/bad-candidates/spot_accum`。

## 标的与预算（constitution `spot.accumulation`）

| Symbol | Budget % | USDT @10k |
| ------ | -------- | --------- |
| BTC    | 50%      | 5,000     |
| BNB    | 25%      | 2,500     |
| SOL    | 25%      | 2,500     |

## 买入

- **深熊**：`weekly_ema_200_position < 0`（周线收盘在周线 EMA200 下方）
- **节奏**：每个 UTC 日最多 1 笔 deploy leg（`max_deploy_legs_per_day: 1`，`min_order_interval_minutes: 1440`）
- **Deploy decay**（相对该 symbol 预算已部署比例）：

| deployed | speed |
| -------- | ----- |
| 0–30%    | 1.0x  |
| 30–60%   | 0.7x  |
| 60–80%   | 0.4x  |
| >80%     | 0.2x  |

## 卖出（盈利倍数阶梯）

达到 **持仓市值 / 成本 ≥ min_profit_multiple**（可按 symbol 配置，默认 5x）后，每个 UTC 日最多减仓一次。

- **基础**：每天卖出剩余仓位的 `base_daily_sell_fraction`（默认 5%）
- **加速**（默认 `type: power`）：

```text
speed_mult = min(max_speed_multiplier, (mtm_multiple / trigger_multiple) ** exponent)
daily_sell_fraction = base_daily_sell_fraction * speed_mult
```

示例（trigger=5x，base=5%，exponent=0.75，max=4x）：

| 浮盈倍数 | speed | 当日卖出占剩余 |
| -------- | ----- | -------------- |
| 5x       | 1.0   | 5%             |
| 10x      | ~1.68 | ~8.4%          |
| 20x      | ~2.83 | ~14%           |

可选 `acceleration.type: exponential` 用 `exp(k * (mtm/trigger - 1))` 替代幂律。

## 回测

```bash
python scripts/event_backtest.py --strategy spot_accum_simple --symbols BTCUSDT,BNBUSDT,SOLUSDT --days 365
```

需 constitution `spot.strategies` 包含 `spot_accum_simple`（与 `spot_accum` 共用 `accumulation` 预算块）。

## Note

1. 没钱了熊市还在只能充钱 
2. 牛市卖出5倍可能低也可能高，这个只能人肉调，但总的来说比自动化买不满提早退达不到目标要好
