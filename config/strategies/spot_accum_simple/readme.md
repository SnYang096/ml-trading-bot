# spot_accum_simple

简化版 A 层现货吸筹（生产默认）。原版 `spot_accum` 已归档至 `config/strategies/bad-candidates/spot_accum`。

## 标的与预算（constitution `spot.accumulation`）

| Symbol | Budget (USDT @12.5k anchor) |
| ------ | --------------------------- |
| BTC    | 5,000                       |
| BNB    | 2,500                       |
| SOL    | 2,500                       |
| ETH    | 2,500                       |

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

## 配置在哪里（不必对齐 B/C 四层管线）

| 关心什么 | 改哪里 | 是否在 archetype 四层 |
| -------- | ------ | --------------------- |
| 三币预算、每笔 USDT 大小、全局 deploy 上限 | `config/constitution/constitution.yaml` → `spot.accumulation` / `risk_limits` | 否（宪法） |
| 周线熊才买 | `archetypes/prefilter.yaml`（`weekly_ema_200_position < 0`） | prefilter |
| `deploy_decay`、5× 阶梯卖 | `archetypes/execution.yaml` | execution |
| 固定只做多 | `archetypes/direction.yaml` | direction |
| 入场时机（形态） | `archetypes/entry_filters.yaml` → **空**；用日频 deploy 节奏代替 | 无 entry 规则 |
| 买入/卖出计算公式 | `src/time_series_model/live/spot_accum_simple.py` | 代码（decay、阶梯卖） |
| gate | 空 | 未使用 |

**不必**为 spot 增加 `research/calibrate_roll.yaml`：A 层无 ML、无月度阈值 rolling；回测走 `event_backtest.py`。

买入门控在 **`prefilter.yaml`**：默认 `weekly_ema_200_position < 0`（周线 EMA200 下方）。

### `weekly_ema_200_position` 数据从哪来（与 tick warmup 无关）

| 问题 | 答案 |
|------|------|
| 是否用 `prepare_warmup_ticks.sh` 那 6 个月合约 tick？ | **否**。那是 USD-M 订单流/1m archive，给 VPIN、`atr_percentile` 等。 |
| 周线 EMA200 用什么？ | **Binance Vision 现货 1d**（默认自 2017-01-01）→ `live/highcap/data/macro/spot_weekly_ema200/<SYMBOL>.parquet`。 |
| 谁写入 bus？ | **`quant-feature-bus`**：每根 120T 特征算完后，用 seed **覆盖** `weekly_ema_200_position`（公式：当前 bar `close` 对 seed 里 ffilled 的周线 EMA）。 |
| spot 进程读什么？ | **`quant-spot-accum` 只读 feature bus**，不直接读 macro 目录。 |
| 没 seed / seed 过期会怎样？ | 列为 **NaN** → prefilter 不通过 → **不 deploy**（fail-closed）。 |

准备 seed：`python scripts/prepare_spot_weekly_ema_seed.py` 或 `bash live/scripts/prepare_spot_macro_seed.sh`；生产 **`quant-macro-seed-prepare.timer`**（每日刷新）。更新 seed 后需 **重启 feature-bus**（或等 ~15min）再 **重启 spot-accum**。

完整机制（150 天 archive 与 macro 两线对比、示意图）：[`docs/deployment/FEATURE_BUS_DATA_PIPELINE_CN.md`](../../docs/deployment/FEATURE_BUS_DATA_PIPELINE_CN.md) § `weekly_ema_200_position` 与 `spot_weekly_ema200`。

**何时下单（实盘 `run_spot_accum_live.py`）**

- 触发：feature-bus **每出一根新 2h bar**（不是固定 UTC 0 点）；bar 时间戳多为 UTC 整点 2h 网格（00:00、02:00…）。
- 新限价单：默认仅 **伦敦 08:00–11:00**（`execution.deploy_schedule`，可改）。
- 订单类型：`entry_order.type: limit`，价 = 收盘价 × (1 − 25bps)；**24h** 未成交撤单（`pending_max_age_hours`）。
- **回测** `event_backtest.py` 不读 `deploy_schedule`，在熊市区按日频 deploy 模拟。

**日志：每根 2h bar 一条 eligibility（默认开启）**

```text
[SOLUSDT] spot-eligibility ts=... NO_NEW_BUY weekly_ema=-0.0523 below_wk_ema200=True ...
  reasons=deploy_schedule:outside_deploy_window ...
```

关闭：`export MLBOT_SPOT_ELIGIBILITY_LOG=false`  
更细：`export MLBOT_SPOT_CHAIN_DEBUG=true`  
实盘下单：`export MLBOT_SPOT_SHADOW_MODE=false`

`entry_filters.yaml` 目前为空；择时窗口在 `deploy_schedule`，不是 entry 特征。

## 回测

```bash
PYTHONPATH=. python scripts/event_backtest.py \
  --strategy spot_accum_simple \
  --symbols BTCUSDT,BNBUSDT,SOLUSDT \
  --start-date 2022-01-01 --end-date 2026-05-01 \
  --data-path data/parquet_data \
  --constitution-yaml config/constitution/constitution.yaml
```

需 constitution `spot.strategies` 包含 `spot_accum_simple`（与 `spot_accum` 共用 `accumulation` 预算块）。

## Note

1. 没钱了熊市还在只能充钱 
2. 牛市卖出5倍可能低也可能高，这个只能人肉调，但总的来说比自动化买不满提早退达不到目标要好
