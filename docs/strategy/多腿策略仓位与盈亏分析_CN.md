# 多腿策略仓位与盈亏分析

> 基于 `segment_dd_target: 0.072`，本金 $10,000 USDT

## 1. 仓位计算

### Chop Grid（均值回归网格）

| 参数 | 值 |
|------|-----|
| `segment_dd_target` | 0.072 |
| `max_loss_per_grid` | 0.03 (3%) |
| `max_levels_per_side` | 3 |
| 总格数 | 2 × 3 = **6 格** |

$$\text{unit\_notional} = \frac{10000 \times 0.072}{0.03 \times 6} = \frac{720}{0.18} = \textbf{4{,}000 USDT/格}$$

- **单段最大亏损**：4000 × 0.03 × 6 = **720 USDT**（= 7.2% 本金）
- **满仓 gross notional**：4000 × 6 = **24,000 USDT**（2.4x 杠杆）

### Trend Scalp（趋势跟踪）

| 参数 | 值 |
|------|-----|
| `segment_dd_target` | 0.072 |
| `max_loss_per_segment` | 0.02 (2%) |
| `max_gross_exposure_units` | 4 |

$$\text{unit\_notional} = \frac{10000 \times 0.072}{0.02 \times 4} = \frac{720}{0.08} = \textbf{9{,}000 USDT/腿}$$

- **单段最大亏损**：9000 × 0.02 × 4 = **720 USDT**（= 7.2% 本金）
- **满仓 gross notional**：9000 × 4 = **36,000 USDT**（3.6x 杠杆）

---

## 2. 理论最坏亏损（几乎不可能发生）

公式假设**所有仓位同时成交且同时打到止损**：

**Chop Grid**：6 格全部成交，每格都亏 `max_loss_per_grid` (3%)
$$4000 \times 0.03 \times 6 = 720 \text{ USDT}$$

**Trend Scalp**：4 腿全部成交，每腿都亏 `max_loss_per_segment` (2%)
$$9000 \times 0.02 \times 4 = 720 \text{ USDT}$$

---

## 3. 四层保护机制

### 第 1 层：Regime 退出（最常见）

| 策略 | 触发条件 | 效果 |
|------|----------|------|
| Chop Grid | `chop < exit_chop_below` (0.25) | 整段平仓 |
| Trend Scalp | `trend_conf < exit_trend_below` (0.50) 或 `chop > max_hold_chop` | 整段平仓 |

市场条件一变差，regime 模型就会先于任何止损触发退出。**绝大多数 segment 的亏损远小于理论值。**

### 第 2 层：Risk Stop / Catastrophic SL

**Chop Grid** — `_risk_stop()` 检查的是**整个 grid 的 MTM**：
```
当 MTM ≤ -max_loss_per_grid × level_notional 时触发
= -0.03 × 4000 = -120 USDT
```
也就是说整个 grid 亏 **120 USDT**（1.2% 本金）就会强制平仓，远不到 720。

**Trend Scalp** — 每腿有交易所侧 catastrophic SL：
```
SL 距离 = max(tp_dist × 8, ATR × 8)
```
以 BTC 为例，ATR ≈ 2%，SL 距离 ≈ 16%，单腿最大亏 ≈ 9000 × 16% = 1440 USDT。但 regime 退出通常远早于此。

### 第 3 层：Protection Orders（交易所侧硬止损）

每个成交的仓位都有交易所挂单保护：

| 策略 | TP | SL |
|------|-----|-----|
| Chop Grid | entry ± tp_distance | entry ± spacing × (levels+1) |
| Trend Scalp | entry ± tp_dist | entry ± max(tp_dist×8, ATR×8) |

即使代码崩溃、网络断开，交易所侧的止损单仍然生效。

### 第 4 层：Constitution Kill Switch

```yaml
max_drawdown_pct: 0.20  # 从峰值回撤超 20% 停止一切新开仓
```

---

## 4. 实际最大亏损估算

| 保护层 | Chop Grid 单段 | Trend Scalp 单段 |
|--------|---------------|-----------------|
| **理论极端**（全部同时止损） | -720 USDT (7.2%) | -720 USDT (7.2%) |
| **Risk Stop / Regime 退出** | **~-120 USDT (1.2%)** | **~-180~360 USDT (1.8~3.6%)** |
| **Catastrophic SL（兜底）** | ~-480 USDT (4.8%) | ~-1440 USDT (14.4%) |
| **Kill Switch（全局）** | -2000 USDT (20%) | -2000 USDT (20%) |

---

## 5. 收益估算

`segment_dd_target` 定义的是**最坏情况亏损**，不是收益。收益取决于策略的 win rate 和 risk-reward：

### Chop Grid（均值回归网格）

每格在 spacing 区间内低买高卖，典型单次网格利润 ≈ spacing × unit_notional：
- 如果 spacing ≈ 0.5%（`atr_mult: 0.50`），单次网格成交利润 ≈ 4000 × 0.5% = **~20 USDT/格**
- 一个完整 segment 6 格全部成交并平仓 ≈ **~120 USDT**（1.2% 本金）
- 实际中不会每格都成交，典型一个 segment 赚 **40-80 USDT**（0.4-0.8%）

### Trend Scalp（趋势跟踪）

TP 目标通常较小但方向性更强：
- 如果 tp ≈ 0.3-0.5%，单腿利润 ≈ 9000 × 0.4% = **~36 USDT/腿**
- 一个 segment 通常 1-2 腿盈利出场 ≈ **36-72 USDT**（0.36-0.72%）
- 趋势好的时候加仓到 3-4 腿，单次 segment 可达 **100-150 USDT**（1-1.5%）

### 汇总

| | 单次 segment 预期收益 | 最坏亏损 | 风险收益比 |
|---|---|---|---|
| **Chop Grid** | ~40-80 USDT | -720 USDT | 约 1:10 ~ 1:18 |
| **Trend Scalp** | ~36-150 USDT | -720 USDT | 约 1:5 ~ 1:20 |

> ⚠️ 这些是**单次 segment** 的估算。实际月收益取决于开仓频率、win rate、市场环境。`segment_dd_target` 从 0.036/0.016 翻倍到 0.072 意味着仓位翻倍，盈亏都放大 2 倍（chop）或 4.5 倍（trend）。如果之前 trend 的 0.016 对应单次 ~8-33 USDT，现在 0.072 就是 ~36-150 USDT，但最坏亏损也从 160 USDT 涨到 720 USDT。

---

## 6. 结论

`segment_dd_target: 0.072` 的 720 USDT 是 sizing 公式的**预算上限**，用来反推仓位大小。实际单段最大亏损被 risk stop 限制在 **~120 USDT (chop)** 或 regime 退出 **~180-360 USDT (trend)**，大约是理论值的 **1/6 ~ 1/2**。

720 USDT 需要所有仓位同时成交且同时打到止损，在有 regime + risk stop + protection orders 三层保护的情况下基本不会发生。
