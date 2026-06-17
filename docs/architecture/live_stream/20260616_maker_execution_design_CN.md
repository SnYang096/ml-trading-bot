# Maker-only 执行方案设计（trend_scalp / 多腿）2026-06-16

> **目标**：把 trend_scalp 的**入场、加仓、出场**全部改成 maker（被动限价）成交，
> **不使用 taker 兜底**（宁可错失也不吃单），只保留灾难级 STOP_MARKET 作为安全网。
> 利用 2h 信号节奏 + 1m 盘中波动，被动挂单有足够长的驻留窗口成交。
>
> **状态**：设计 + Phase 1（mock 费用模型）已落地；Phase 2/3（engine maker 出场 / 入场）实现中。
> 关联：`config/experiments/20260613_multileg_sizing_validate/BACKTEST_LIVE_ALIGNMENT_CN.md` §6/§8。

---

## 1. 问题与量化

`recent_6m_oos`（2025-12 → 2026-05，10k 本金，joint chop+trend，maker 2bps / taker 5bps）费用构成：

| 费用桶 | USDT | 占比 | 来源 |
|---|---|---|---|
| `open_taker` | 13,165 | 53% | 入场 + 加仓（`marketable_limit` IOC，吃单） |
| `reduce_taker` | 10,731 | 43% | 出场（basket 市价平 / SL / TP-market，吃单） |
| `reduce_maker` | 974 | 4% | chop_grid 的限价 TP（已是 maker） |
| `open_maker` | 0 | 0% | trend 入场无被动挂单 |
| **合计** | **24,870** | | 6 个月 ≈ 本金 2.5× |

- 出场全 maker 的理论上限省费 `est_maker_tp_savings ≈ 6,438`（reduce notional 21.46M ×(5−2)bps）。
- 入场/加仓是更大的桶（53%）；全 maker 还能再省一块（量级与出场相当）。
- 综合：若入场+出场都做到 maker，理论上费用可从 ~24.8k 降到 ~10k 量级（仅剩偶发 taker 安全网为 0、加上漏单减少的交易量），**省费可观**。

---

## 2. 设计原则

1. **全 maker、无 taker 兜底**：入场/加仓/TP/regime 出场/flip 平仓都用 post-only 限价；挂不上就等、再不行就**漏单**（miss），绝不转市价。
2. **唯一例外 = 灾难级 STOP_MARKET**：`protection_stop_mode: catastrophic` 的止损保持 STOP_MARKET（吃单）。它是极少触发的安全网，改成 maker 会在崩盘时无法成交→爆仓。**这条是安全红线，不纳入 maker 化**（若要改需单独评估）。
3. **复用现有机制**，避免重写引擎契约：
   - 出场 = reduce-only 限价单走 `place_protection`(`order_type: limit`) 通道；成交由已实现的 `_handle_protection_fill` 移除该腿（与 chop_grid 同构）。
   - 入场/加仓 = 现有 `pending_orders` + 跨 bar re-peg。
4. **利用 2h×1m 结构**：engine 按 2h 出 action，mock 在两次 2h 之间用每根 1m bar 撮合挂单（live 是真实盘口）。一个被动挂单天然有 ~2h（=120 根 1m）的驻留成交窗口；配合每 2h re-peg 追价，成交率应当很高。

---

## 3. 订单类型策略（`maker_execution: true`）

| 订单 | 现状（taker） | maker 模式 |
|---|---|---|
| 初始入场 | `marketable_limit` IOC，submit=close×(1±slip) | **post-only LIMIT @ close**，不加滑点；跨 bar re-peg；超 `maker_entry_max_wait_bars` 或信号失效 → 撤单漏单 |
| 趋势加仓 | 同上 | 同上（被动 LIMIT @ 阶梯价或 close）；漏单不补 taker |
| TP 出场（basket/per_leg） | `market_exit` | **reduce-only post-only LIMIT @ TP 价**（常驻=maker，价到即成） |
| regime 出场 | `market_exit` | reduce-only post-only LIMIT @ close；每 bar re-peg 追价；不转 taker |
| trend flip 平反向腿 | `market_exit` | 同 regime 出场 |
| 灾难止损 | STOP_MARKET | **不变**（安全网，吃单，极少触发） |

---

## 4. Peg / re-peg 算法（被动、永不穿越盘口）

**挂价**（回测只有 OHLC，用 `close` 作为 touch 代理；live 用 best bid/ask）：
- BUY maker：limit price = `close`（或 best_bid）。买限价挂在市价下方/平齐 → 被动；当后续某根 1m `low ≤ price` 时成交（= 价格下来找到我 = maker）。
- SELL maker：limit @ `close`（或 best_ask）；当 `high ≥ price` 成交。

**re-peg**：每个 engine bar（2h），若挂单未成交且仍需要 → 撤单 + 按新 `close` 重挂，被动追价，使挂单始终贴近 touch。

**post-only 保证**：若 re-peg 价格会立即穿越盘口（marketable），交易所会拒 post-only。回测 mock 用「只有当 bar 在 maker 方向触达挂价时才成交」来诚实建模——不会在下单瞬间以更差价成交。

---

## 5. 等待 / 漏单策略（不用 taker）

- **入场/加仓**：挂单有效期 `maker_entry_max_wait_bars`（2h bar 数，默认 2）。每 bar re-peg **仅当入场信号仍有效**（`trend_conf ≥ entry_trend_min`、`chop ≤ max_entry_chop`、方向未变）。信号失效或超期 → 撤单漏单（不转 taker）。
- **出场（TP/regime/flip）**：默认无最大等待——我们要平，但**只以 maker 方式**。每 bar re-peg 追价。
  - TP 挂在有利侧（常驻），价到即成，成交率高。
  - regime/flip 用 close 挂被动单 + 追价，盘中波动通常能成交。
  - **尾部风险**：单边急行情下 regime 出场可能延迟成交——由灾难级 STOP_MARKET 兜底极端情形（仅防爆仓，不防小亏）。文档明确接受此风险。

---

## 6. 回测（mock）改动

1. **maker/taker 费率模型**（✅ 已落地）：`maker_fee_bps`/`taker_fee_bps`；按 (开/平 × maker/taker) 拆分 `fee_breakdown`/`notional_breakdown`；回测 summary 输出 + `est_maker_tp_savings_usdt`。
2. **maker 意图标记（待做）**：当前用 `reduce_only && limit` 推断 maker，不能区分「被动入场限价」与「marketable IOC」。改为挂单显式带 `post_only`/`maker` 标记，由 engine/adapter 设置，mock 按该标记计费。这样入场 maker、出场 maker 都能正确归类。
3. **诚实被动成交**：mock 的 limit 撮合已是「触达即成」（buy: low≤price；sell: high≥price），对 post-only 入场（submit=close，不加滑点）天然要求一次回踩才成交，符合 maker 语义。marketable 入场（submit=close×(1+slip) 高于市价）则维持现有「上方即成 = 近似 taker」。

---

## 7. 引擎改动

### 7.1 配置 schema（`order_model` / `take_profit`）
```yaml
order_model:
  maker_execution: true            # 主开关（opt-in，默认 false → 现有 taker 行为）
  entry_order_type: limit          # maker 模式下被动限价（替代 marketable_limit）
  add_order_type: limit
  maker_entry_max_wait_bars: 2     # 入场/加仓挂单最大等待（2h bar）；超时漏单
  maker_repeg_bars: 1              # 每 N 个 bar re-peg 一次
take_profit:
  exit_order_type: limit           # TP/regime/flip 出场走 reduce-only 限价
```

### 7.2 数据结构
`DualAddOrder` 增 `post_only: bool`、`order_kind`（entry/add）、`repeg_count`。
出场限价复用 `DualAddPosition.protection_order_ids`（与 chop 同构）。

### 7.3 关键方法
- `_place_order`：maker 模式 → `order_type="limit"`、`post_only=True`、`submit_price=close`（不加滑点）、`tif=GTC`。
- `_repeg_pending_entries()`（新）：每 `on_bar` 调用；对未成交、仍需要、信号有效的入场/加仓挂单 → cancel + 按新 close 重挂；超期/信号失效 → cancel 漏单。
- 出场（`_target_exits` / `_exit_all` / `_handle_trend_flip`）：maker 模式 → 不发 `market_exit`，改发 reduce-only 限价（`place_protection` take_profit-limit @ 出场价），**保留 inventory 直到成交**；成交由 `_handle_protection_fill` 移除腿。
- `_repeg_exit_limits()`（新）：regime/flip 出场挂单每 bar 追价（cancel_protection + 重挂）。
- 灾难 SL：`_protection_actions` 的 stop 分支不变（STOP_MARKET）。

### 7.4 出场契约变化（重点）
现状 `_exit_all` 立即清空 inventory；maker 模式下**出场是异步的**：挂限价 → 等成交 → `_handle_protection_fill` 移除腿。inventory 在成交前保留（这是正确的：未平就还在仓）。段生命周期 `_begin_closing` 在「全部腿挂出限价」时进入 CLOSING，但 `_maybe_deactivate_if_fully_closed` 仅在 inventory 真正清空后置 IDLE。

---

## 8. 实盘改动

- adapter 已支持 limit/post_only：`_place_protection`(take_profit, `order_type=limit`) → reduce-only LIMIT；`_place_entry` 支持 limit + tif。
- 入场 post-only：`_place_entry` 增传 `post_only`（Binance `timeInForce=GTX`）。
- 出场限价：复用 `place_protection` take_profit-limit；regime/flip 出场同样走该通道（reduce-only limit @ close）。
- reconcile：未成交 maker 挂单由现有 reconcile/`actions_ensure_protection` 维护；re-peg 由 engine 在每个 60s/2h 周期发 cancel+place。

---

## 9. 实验计划与指标

对照回测（同段、同 seed）：
1. **baseline**：现状（marketable 入场 + market 出场），maker2/taker5 计费。
2. **maker-exit**：仅出场 maker。
3. **maker-only**：入场+出场全 maker。

指标：
- `total_fees_usdt` 与 `fee_breakdown`（maker/taker 占比）
- **fill rate / miss count**（入场漏单数、出场成交延迟 bar 数）
- `return_pct` / `max_drawdown_pct`（漏单与追价的盈亏影响）
- 成交量（notional）变化

段：先 `recent_6m_oos`（promote 门禁段），再 `bear_2022` / `recent_range_to_bear` 验证极端段的漏单/尾部风险。

---

## 10. 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| 入场漏单 | 急行情下被动单不成交 → 错过趋势 | re-peg 追价 + 接受漏单（用户已确认）；统计 miss 率评估 |
| 出场延迟 | 单边行情下 regime 出场 maker 不成交 → 持仓更久 | 灾难 STOP_MARKET 兜底极端；统计成交延迟 |
| 追价逆选择 | re-peg 追价可能在更差价成交（仍 maker） | 入场限 `max_wait_bars`；记录成交价 vs 信号价滑点 |
| 回测乐观 | mock「触达即成」可能高估 maker 成交率 | 用保守撮合（要求严格 < / >，见实现）；live shadow 验证 |
| 段生命周期 | 异步出场使 CLOSING 态延长 | inventory 清空才置 IDLE；`max_segment_bars` 仍限制总时长 |

---

## 11. 落地阶段

- **Phase 1（✅）**：mock maker/taker 费率模型 + 回测费用拆分 + 量化。
- **Phase 2**：engine maker **出场**（TP 常驻限价 + regime/flip 追价限价），`exit_order_type: limit`，opt-in。
- **Phase 3**：engine maker **入场/加仓**（被动限价 + re-peg + 信号复核 + 漏单），`maker_execution: true`。
- **Phase 4**：mock 显式 post_only 计费标记；对照回测三组；记录结论到实验文档。
- **Phase 5**：live shadow（adapter post_only / reduce-only limit），对账验证后灰度。

---

## 12. 对照回测结论（2026-01-01→01-15，trend-only，maker2/taker5，equity 10k）

> ⚠️ 单一 2 周窗口，仅作方向性判断；正式 promote 前需跑 4 canonical 段。

| 变体 | return% | halt | realized PnL | total fee | open(mk/tk) | reduce(mk/tk) |
|---|---|---|---|---|---|---|
| **taker baseline** | -7.17 | 否 | -717 | 599 | 0 / 299 | 0 / 299 |
| **maker-only（入场+出场）** | **-19.62** | **是** | -1962 | 232 | 116 / 0 | 116 / 0 |
| **hybrid（maker 入场 + taker 出场）** | **-2.43** | 否 | -243 | 429 | 123 / 0 | 0 / 307 |

**核心发现：**

1. **maker 计费链路正确**：maker-only 全部走 maker，总费 599→232（**-61%**）；notional 与 taker 基本持平（~58 万），说明 2h 节奏 + 1m 撮合下**成交率不是问题**。
2. **maker-only 反而巨亏**：realized PnL -1962 vs taker -717，且触发 dd>20% 熔断。根因 = **出场**：单边逆行情中 reduce-only 限价挂在 touch 不成交，re-peg 追价 → 在更差价平仓，比 taker 即时止损亏更多。TP（顺向）出场无此问题，问题集中在 regime/flip 风险出场。
3. **入场不是问题**：`maker_repeg_bars` 0 与 1 结果逐字相同 → 被动入场在下一根 1m 即按 touch 成交（≈ taker 价但免滑点、且漏掉的多是逆选择单），入场改 maker 反而**改善**了入场质量。
4. **hybrid 最优**：maker 入场 + taker 出场 → return -2.43%（**优于 taker 基线**），无熔断；open fee 299→123（**-59%**），出场保持 taker 即时风控。总费 599→429（-28%）。

**建议落地形态：`maker_execution: true`（入场被动 maker，错失不转 taker）+ `exit_order_type: market`（出场 taker 即时）。**
出场全 maker（用户初始诉求「出场也不用 taker」）经数据证伪——出场无法「错失」，只能追价，逆行情下伤害 > 省下的费用。若仍要 maker 出场，应仅限 TP（顺向）出场用 maker、regime/flip 风险出场保持 taker（后续可加细分开关）。

配置：`config/experiments/20260616_maker_execution_validate/variants/{trend_maker, trend_maker_entry}`。
