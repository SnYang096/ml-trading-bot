# BPC 加仓杠杆与 ATR 速查

## 结论摘要

- `fixed_multiplier`：按固定比例加仓（简单、稳定），但不保证精确卡到目标总杠杆。
- `target_leverage_gap`：按“离目标杠杆差值”补仓（更适合精确控总杠杆）。
- 本系统初始仓位不是固定 20x，也不是默认 20x 直接使用；仓位由风险反算得到。
- 代码中 **`compute_slot_size_from_risk` 的 `max_leverage` 在 `trade_executor` 里因分支不同而不同**：`constitution_risk` 路径为 **3.0**，`risk_per_trade_usd` 路径为 **10.0**（见 `src/order_management/trade_executor.py`）；`slot_sizing` 默认 cap 亦为 **3.0**（见 `src/time_series_model/portfolio/slot_sizing.py`）。文档前面按「宪法风险 1% + 3x cap」的直觉表仍适用主路径。

## 关键公式

- `atr_pct = ATR / price`
- `stop_pct = stop_atr * atr_pct`
- `L0 (初始杠杆) ≈ min(risk_frac / stop_pct, 3.0)`
  - 常见 `risk_frac = 0.01`（本金 1% 风险）
- `fixed_multiplier` 下总仓位倍数：
  - `total_mult = 1 + sum(add_size_multipliers)`
  - 当前配置 `[0.5, 0.35, 0.25, 0.20, 0.15]` -> `total_mult = 2.45`
  - `L_final ≈ L0 * 2.45`

## 2ATR / 4ATR 一般是多少

这取决于 `atr_pct`。先给换算关系：

- `2ATR = 2 * atr_pct`
- `4ATR = 4 * atr_pct`

下表给常见区间的直觉（按百分比）：

| ATR/Price (`atr_pct`) | 2ATR 止损宽度 | 4ATR 止损宽度 |
|---|---:|---:|
| 0.15% | 0.30% | 0.60% |
| 0.25% | 0.50% | 1.00% |
| 0.50% | 1.00% | 2.00% |
| 1.00% | 2.00% | 4.00% |

## 在 risk=1% 下的杠杆直觉

用 `L0 ≈ min(0.01 / stop_pct, 3.0)`：

| stop_pct | L0 (初始) |
|---:|---:|
| 0.30% | 3.0x (原本 3.33x，被 3x 上限裁剪) |
| 0.50% | 2.0x |
| 1.00% | 1.0x |
| 2.00% | 0.5x |

再乘 `fixed_multiplier` 总倍数 2.45：

| L0 | L_final (固定加仓后) |
|---:|---:|
| 3.0x | 7.35x |
| 2.0x | 4.90x |
| 1.0x | 2.45x |
| 0.5x | 1.23x |

> 结论：`fixed_multiplier` 可以到 5x 附近，但不是“必然=5x”。  
> 如果目标是“尽量严格贴近 5x 上限”，优先用 `target_leverage_gap`。

## 关于 Binance 20x

- Binance 账户可能有默认杠杆设置，但本系统当前路径没有自动调用 `set_leverage()` 去设成 20x。
- 实际有效杠杆主要由“下单数量反算 + 风险约束 + 交易所保证金规则”共同决定。

## 两种加仓模式（实现位置）

实现见 **`src/time_series_model/core/constitution/add_position_rules.py`**（及调用方）。

**`fixed_multiplier`**

- 只按 `add_size_multipliers` 缩放加仓规模。
- 不主动用 `target_leverage_by_add` 去对齐目标杠杆。
- 行为稳定、可预期。

**`target_leverage_gap`**

- 先按「目标杠杆与当前杠杆的缺口」换算倍数，再被 `max_total_leverage`、`max_add_leverage_step`、`max_add_notional_frac`、`min_add_notional_usd` 等约束夹住。
- 更偏风控化，会随当前仓位状态自适应。

### `fixed_multiplier` 与总杠杆倍数

当前示例数组 `[0.5, 0.35, 0.25, 0.20, 0.15]` 的和为 **1.45**，故相对初始仓位的总名义倍数约为 **1 + 1.45 = 2.45**（与上文「总倍数 2.45」一致）。

在 **`L_final ≈ L0 × 2.45`** 关系下，若初始约 2x 则全加完后约 4.9x；初始约 2.2x 则约 5.39x。因此 **不保证** 恰好落在某固定倍数（如 5x），除非外层另有硬约束。

