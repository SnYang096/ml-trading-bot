# C 系统总结

C 系统是 **120T（2h）多腿库存（multi-leg）组合**，挂在宪法 `multi_leg` 账户，与 A（现货吸筹）、B（PCM 趋势 swing）**账户与执行模型均分离**。核心职责：**在 B 系统不愿做的市场里赚钱**——震荡用网格收割回摆，非 chop 趋势段用顺势加仓；与 TPC 等 trend swing **天然互补**。

---

## 定位与架构

| 维度 | 内容 |
|------|------|
| 策略 | `chop_grid`（震荡网格）、`trend_scalp`（趋势 regime 小波段；原 `dual_add_trend`） |
| 周期 | `120T` |
| 账户 | `multi_leg` 独立合约账户（`run_multi_leg_live.py`），非 PCM trend 槽位 |
| 管线 | **不走** 单仓位 `TradeIntent` / 标准 `event_backtest`；专用 `grid_backtest` / `dual_add_backtest` 库存引擎 |
| 互斥 | **同一 symbol 同时只能跑其中一个**（`chop_grid` ↔ `trend_scalp`） |

```text
Portfolio 分工（与 A/B 对照）：
  fattail / 长期在车上  → A（spot_accum_simple）
  高置信 trend swing    → B（TPC / BPC / ME / SRB，PCM）
  震荡 mean-reversion     → C.chop_grid
  非 chop 趋势段 scalp    → C.trend_scalp
```

设计哲学（与 B 对照）：

| | B（trend swing） | C（multi-leg） |
|--|------------------|----------------|
| 赚钱条件 | 有趋势、高置信 pullback/突破 | chop 来回扫 **或** 趋势延续可加仓 |
| 亏钱条件 | 震荡、假突破 | 单边趋势突然启动（网格一侧被套） |
| 与 TPC | 争抢趋势段 | chop_grid **对冲** TPC 震荡亏损 |
| 执行形态 | 单笔 intent + 结构止损 | 多层库存 + regime 段退出 |

---

## 资金与风控（宪法 `multi_leg`）

| 参数 | 值 | 含义 |
|------|-----|------|
| `unit_notional` | 200 USDT | 每格 / 每腿名义（@ 10k 锚定约 2%/腿） |
| `equity_usdt` | 10,000 | 离线预算锚定 |
| `max_drawdown_pct` | 12% | 多腿账户硬停 |
| `max_gross_notional_pct` | 20% | 组合总 gross 上限 |
| `max_net_notional_pct` | 10% | 组合净敞口上限 |
| `max_symbol_gross_notional_pct` | 8% | 单币 gross 上限 |
| `max_symbol_net_notional_pct` | 4% | 单币净敞口上限 |

粗算敞口上限（与注释一致）：
- `chop_grid`：`max_open_levels_total=4` → 约 800 USDT/symbol gross
- `trend_scalp`：`max_gross_exposure_units=4` → 约 800 USDT gross footprint

标的：`trend_scalp` 默认 BTC/ETH/SOL/BNB/XRP；`chop_grid` `symbol_include` 为空（沿用全局 universe）。

---

## 子策略一：`chop_grid`（震荡中性网格）

**语义**：不预测涨跌；在 **广义震荡（semantic chop）** 里于现价上下挂限价，靠价格来回吃网格间距。

### Regime（`archetypes/prefilter.yaml`）

| 项 | 配置 |
|----|------|
| 入场特征 | `bpc_semantic_chop` |
| 开网格 | `entry_chop_min: 0.50` |
| 关网格 | `exit_chop_below: 0.32` |
| 结构过滤 | `box_pos_60` 须在 **0.35–0.65**（箱体中部震荡区） |
| `exclude_box_prefilter` | `false`（当前候选含 box 结构；README 亦讨论 `chop_not_box` 变体） |

退出语义：chop 失效 → 撤未成交单 + **强制平掉所有库存**（taker + slippage）。

### 网格执行（`archetypes/execution.yaml`）

```text
spacing = max(atr_mult * ATR, min_pct * price)
默认: atr_mult=1.00, min_pct=1.0%
max_levels_per_side: 2（每边最多 2 层）
max_open_levels_total: 4
order_type: limit
```

方向：**无单一方向**。下方买单成交 → LONG 库存，目标上一格止盈；上方卖单 → SHORT 库存，目标下一格止盈。每层独立成交、独立止盈。

### 主要风险

- 趋势突然启动 → 一侧网格连续成交，库存单边堆积，靠 `force_exit_on_regime_loss` 强平
- 触价 ≠ 实盘成交（排队、部分成交）
- 强平成本高于正常网格 maker 止盈；含 funding

### 研究快照（README / 诊断脚本）

- `semantic_chop` / `chop_not_box`：多周期回测 **5/5 为正**
- `0.50 ATR` 间距在 `chop_not_box` 下 PnL 最强；`0.75/1.00 ATR` 换手更低
- 稳定窄 box 更适合 CRF 边界 fade，**不宜与 chop 网格混参**（`box_prefilter` 网格 4/5 为正、2024 有大亏）

---

## 子策略二：`trend_scalp`（regime 趋势小波段）

**语义**：非无限马丁、非中性网格。趋势段 **先开顺势一腿（`initial_legs: TREND`）**，仅沿当前趋势方向加仓；趋势掉头 **`close_offside_all`** 清逆势腿，**不在同一段内立即反向 reseed**（`reseed_on_flip: false`，见 `EXPERIMENT_FLIP_RESEED.md`）。

### Regime（`archetypes/prefilter.yaml`）

| 项 | 配置 |
|----|------|
| 入场特征 | `trend_confidence` |
| 开仓 | `entry_min: 0.7` |
| 段结束 | `exit_below: 0.4` |
| chop 约束 | 入场 `max_semantic_chop_entry: 0.25`；持仓 `max_semantic_chop_hold: 0.4` |
| box | `exclude_box_prefilter: true`（稳定 box 留给 CRF 类策略） |

**适用阶段**：非 chop、非稳定 box 的 **趋势延续**；放在 chop 内会频繁掉头、手续费吞噬利润。

### 库存与执行（`archetypes/execution.yaml`）

```text
初始: TREND（单腿顺势，非多空对冲开局）
加仓: add_mode=trend, max_adds_per_side=3
间距: 0.75 * ATR（段内冻结 ATR）
敞口: max_gross_exposure_units=4, max_net_exposure_units=2
止盈: basket 模式, fee_aware（含往返 fee buffer）
订单: marketable_limit, max_slippage_bps=5
翻转: close_offside_all（实证优于保留逆势腿）
段风控: risk_stop_mode=regime_only, max_loss_per_segment=0.02
保护: catastrophic STOP（8×ATR 或 8×TP 距离取大）
```

止盈距离（研究默认）：

```text
net_target = max(0.6 * ATR, 0.12% * price)
tp_distance = fee_buffer + net_target
fee_buffer = 2 * fee_bps * price   # 研究侧 fee_bps=4/side → 8bps round trip
```

### 加仓是否有用（2024 Q1 消融，README）

相对 `max_adds=0`，允许顺势加仓显著提升 `return_pct` 并常收窄 `worst_segment`；`max_adds` 2→3 边际变小（样本与执行粒度敏感，需本地复跑验证）。

### 全周期证据快照（2022–2026，6 币，fee-aware）

- 段数 3,118；交易 15,791；交易胜率 91.1%；段胜率 76.7%
- 费后净 PnL 5.85（资本桶口径）；最差段 -2.13%；risk stop 2.76%
- 费用约占 gross 35% → **必须 fee-aware TP**

---

## C 系统内部：为何二选一

`chop_grid` 与 `trend_scalp` **edge source 相反**：

| 策略 | 赚钱 | 亏钱 |
|------|------|------|
| chop_grid | 无方向震荡、边界模糊 chop | 单边趋势突破 |
| trend_scalp | 趋势延续、低 chop | 震荡、频繁翻转 |

同一 symbol 若同时开网格又开趋势腿，会在同一行情里 **自相矛盾**（一边收割震荡、一边赌趋势），故宪法强制 **per-symbol 互斥**。

---

## 与 A / B 的协作

```text
A：年级持仓、fattail、无结构止损
B：趋势 swing、拒高 chop（semantic_chop > 0.4）、PCM 单仓
C：专门吃 B 不要的 regime
     chop_grid  ↔  TPC 在震荡市天然对冲
     trend_scalp ↔ 与 TPC 部分重叠（都要趋势），但执行模型不同（多腿库存 vs 单仓结构止损）
```

- B 文档共识：**「震荡收益 → chop_grid」**；不应让 B 用宽止损去抓 fattail。
- C 不替代 A 的长期在车上；不替代 B 的高置信 swing，而是 **填补组合曲线的另一段 regime**。

---

## 运维心智

```text
定期可动（慢变量）：
  regime 阈值 — semantic_chop / trend_confidence 开平仓带
  box 是否纳入（chop 的 exclude_box、trend_scalp 的 exclude_box=true）

异常才查：
  spacing / max_levels / max_adds / flip_action 消融结果

几乎不动：
  多腿引擎契约（inventory + risk 块语义）
  与 BPC 四层 YAML 形态不必强行统一（见 chop_grid prefilter 注释）
```

**现状**：两策略仍为 **研究/多腿专用管线**；接入通用实盘前需 first-class 多腿成交、funding、强平与 OMS 对齐（README Current Status）。

研究入口示例：

```bash
# chop_grid 滚动校准
mlbot pipeline run --config config/strategies/chop_grid/research/calibrate_roll.default.yaml

# trend_scalp 滚动校准
mlbot pipeline run --config config/strategies/trend_scalp/research/calibrate_roll.default.yaml
```

---

## 一句话

**C 系统 = 独立 multi-leg 账户下的 `chop_grid`（高 chop 中性网格）与 `trend_scalp`（低 chop regime 小波段）二选一 per symbol；专吃震荡与可加仓趋势段，与 A/B 账户分离，与 B 的 trend swing 在 regime 上互补；维护重点是 chop/trend regime 阈值与费用感知止盈，而非并入 PCM 四层管线。**
