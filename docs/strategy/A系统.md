# A 系统策略总结（`spot_accum_simple`）

生产里的 **A 层**就是这套现货吸筹策略：简化版 `spot_accum_simple`（原版 `spot_accum` 已归档）。和 B 系统（BPC/TPC/ME swing）分工相反——**A 负责「确保在车上」**，不怕错过牛市没仓位；B 负责「每次上车都值得」，宁可少做。

---

## 定位与架构

| 维度   | 内容                                                                  |
| ------ | --------------------------------------------------------------------- |
| 策略名 | `spot-accum-simple-120T` / archetype `SpotAccumSimpleLong`            |
| 周期   | `120T`                                                                |
| 标的   | BTCUSDT、BNBUSDT、SOLUSDT（long-only，无 ML）                         |
| 特征   | 仅 `weekly_ema_200_position_f`、`atr_percentile_f`                    |
| 过滤   | prefilter / entry_filters / gate **全空**——纯规则，无打分、无训练标签 |

宪法里挂在 `spot.strategies`，与 B 的 PCM 槽位池分离，用独立现货账户与预算块 `spot.accumulation`。

---

## 资金与预算（@ 10k USDT 锚定）

| 币种 | 预算占比 | USDT  |
| ---- | -------- | ----- |
| BTC  | 50%      | 5,000 |
| BNB  | 25%      | 2,500 |
| SOL  | 25%      | 2,500 |

- 目标部署：`target_deploy_pct: 1.0`（满仓锚定）
- 每币 20 档：`BTC 250/笔`，`BNB/SOL 125/笔`
- 不允许加仓（`allow_add_on: false`，`add_position` disabled）

---

## 买入（深熊 DCA）

**入场条件**：`weekly_ema_200_position < 0`（周线收盘在周线 EMA200 下方 = 深熊）

**特征来源（实盘）**：该列由 **feature-bus** 提供；bus 内数值来自 Vision **现货日 K 长历史** 生成的 `macro/spot_weekly_ema200` seed，**不是** `prepare_warmup_ticks` 的 6 个月合约 tick。Publisher 算特征时虽只读 archive **约 150 天** 1m bars，但对本列会用 seed **覆盖**。详见 [`docs/deployment/FEATURE_BUS_DATA_PIPELINE_CN.md`](../deployment/FEATURE_BUS_DATA_PIPELINE_CN.md) § `weekly_ema_200_position` 与 `spot_weekly_ema200`。

**节奏**：
- 每 symbol 每 UTC 日最多 1 笔 deploy（`max_deploy_legs_per_day: 1`，`min_order_interval_minutes: 1440`）
- 限价单，`limit_offset_bps: 25`

**Deploy decay**（已部署占该币预算比例越高，单笔越小）：

| 已部署 | 速度倍率 |
| ------ | -------- |
| 0–30%  | 1.0×     |
| 30–60% | 0.7×     |
| 60–80% | 0.4×     |
| >80%   | 0.2×     |

含义：熊市持续吸筹，越买越慢，避免过早打满预算；预算耗尽后只能继续充钱（readme 里写的运维现实）。

---

## 卖出（盈利倍数阶梯）

**触发**：持仓市值 / 成本 ≥ `min_profit_multiple`（默认 **5×**，三币均为 5×）

**节奏**：触发后每 UTC 日最多减仓一次

**公式**（`type: power`）：
```text
speed_mult = min(4.0, (mtm_multiple / 5) ** 0.75)
daily_sell_fraction = 5% × speed_mult   # 占剩余仓位
```

| 浮盈倍数 | 速度  | 当日卖出占剩余 |
| -------- | ----- | -------------- |
| 5×       | 1.0   | 5%             |
| 10×      | ~1.68 | ~8.4%          |
| 20×      | ~2.83 | ~14%           |

无止损、无 trailing、无 time stop——退出完全由 `spot_simple_profit_ladder` 结构化处理。

---

## 与 B 系统的对比（设计哲学）

|            | A（spot_accum_simple）  | B（swing）              |
| ---------- | ----------------------- | ----------------------- |
| 核心逻辑   | 确保在车上              | 确保每次上车都值得      |
| 最怕       | 错过牛市没仓位          | 低置信度入场稀释 edge   |
| 对「错过」 | 不能接受                | 必须接受                |
| 信号复杂度 | 极简（周线熊 + 阶梯卖） | 多层 gate / 入场 / 执行 |

---

## 运维与调参要点（来自 readme）

1. **熊市没钱**：策略会继续发买信号，只能靠外部充值。
2. **5× 阶梯**：牛市卖得快慢是人工权衡（`base_daily_sell_fraction`、`exponent` 等），宁可卖不满/晚退，也不要自动化过早清仓导致「达不到长期吸筹目标」。

---

## 回测命令

```bash
python scripts/event_backtest.py --strategy spot_accum_simple --symbols BTCUSDT,BNBUSDT,SOLUSDT --days 365
```

基线证据在 `live/highcap/evidence/spot_accum_simple/` 与 `results/120T/spot_accum_simple/`。

---

**一句话**：A 系统 = 三币现货、周线深熊定投吸筹 + 5× 起跳的幂律加速阶梯减仓，规则极简、无 ML，资金与 B 系统分离，哲学是「牛市必须在车上」。
