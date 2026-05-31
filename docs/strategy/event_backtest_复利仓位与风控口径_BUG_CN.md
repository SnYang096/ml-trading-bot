# event_backtest 复利仓位 Bug + 风控口径说明

> **状态**：已知 Bug（待修复）  
> **发现日期**：2026-05-28  
> **影响范围**：`scripts/event_backtest` 下 TPC / BPC / ME 等趋势策略的 **capital_report CAGR、名义敞口、与实盘机器人对比**  
> **勿与**：TradingView / Yin2maBoxTrendBot 等 **外部复利回测** 混谈（见文末对照）

---

## 1. Bug 摘要

### 现象

连续 event_backtest（例：TPC prod，`2024-01-01` → `2025-01-01`，6 币，`--initial-capital 10000`）：

| 指标 | 回测输出 | 实盘 / Yin2ma 量级预期 |
|------|----------|------------------------|
| CAGR（capital_report） | **~5%** | 同策略复利 lots 下可为 **数倍~数十倍**（取决于 Report 初始本金） |
| median 名义 / 权益 | **~0.16x**（~$1,660 / $10k） | 单 BTC 首仓常见 **~0.39x~0.80x**（0.06~0.12 lots） |
| 仓位 sizing 锚点 | **全程固定** `$10,000 × 1% = $100` / R | 应随 **净值（Equity）复利** 放大 |

**结论**：在 `$10k`、**不复利 sizing** 的前提下，**5% 就是字面 5%**——不是「分母设太大」记错，而是 **模拟器没有把盈利滚入下一笔的 risk budget**；与 Yin2ma Excel（lots 从 0.12 → 1.37）**不是同一套记账**。

### 根因（代码）

1. **`_risk_per_slot_usdt` 只在回测开始时设一次**，不随 `_equity` 更新：

   - `scripts/event_backtest/backtester.py`：` _risk_usdt_per_unit = _initial_cash * _risk_per_slot`，写入各 simulator 的 `_risk_per_slot_usdt`。
   - 循环内 `_equity += pnl_usd` 会更新权益曲线，但 **从不回写** `_risk_per_slot_usdt` / `_account_risk_equity`。

2. **`capital_report.py` 明示非复利 sizing**：

   - assumptions：`"compounding": "Equity is additive per trade with fixed initial_capital sizing."`
   - 权益曲线 = `initial_capital + cumsum(pnl_usd_realized)`（PnL 复利展示），但 **下一单开仓量仍按初始 1% risk**。

3. **与宪法设计意图不一致**：`config/constitution/constitution.yaml` 写明 `risk_usd = equity × risk_per_slot`；event_backtest 当前用的是 **`initial_cash × risk_per_slot`（冻结）**。

### 附带问题（R 口径，部分已修）

- **加仓 leg 独立止损平仓**时，`pnl_r` 仍可能按价格路径虚高（见 commit `7ee148bb` 母仓路径；add-leg `sl` 仍有偏差）。
- **turbo 月度 `stitched_total_r`** 不可当绝对盈利；与本次 Bug 不同层。

### 预期修复方向

| 项 | 状态 |
|----|------|
| Sizing 锚点：`risk_usd = current_equity × risk_per_slot` | **已修**（默认 `compound_sizing=True`） |
| CLI `--no-compound-sizing` | **已加**（legacy 对照） |
| `_account_risk_equity` / spot budget 同步 | **已修** |
| capital_report assumptions 文案 | **已更新** |
| 加仓 leg `pnl_r` 全路径 | 待修 |
| turbo `stitched_total_r` | 勿用 |

### 相关路径

- `scripts/event_backtest/backtester.py` — `_risk_usdt_per_unit`、`_equity`
- `scripts/event_backtest/simulator/position.py` — `_estimate_entry_notional_usdt`、`_risk_budget_usdt`
- `scripts/capital_report.py` — assumptions / equity curve
- `config/constitution/constitution.yaml` — `slots.risk_per_slot`
- 对照实盘账单：`Yin2maBoxTrendBot` History xlsx（583 笔，Net **$103,849**；lots 复利）

---

## 2. 量化风控悖论：名义 39% vs 风险 1%（Volatility-Based Position Sizing）

这是一个非常经典且精妙的「量化风控悖论」：

**为什么名义本金（敞口）已经占到了总资金的 39%，但 4ATR 止损却能把总资金的损失控制在区区 1% 左右？**

答案：**「名义价值」不等于「最大风险」**。在带止损的衍生品交易中，决定亏多少的不是开了多少货（名义价值），而是 **开仓价到止损价的距离（波动空间）**。

下面用回测真实量级（BTC ~$65k、0.06 lots、账户 $10k）做纯数学拆解。

### 2.1 核心推导：1% 风险如何被锁死

假设账户 **$10,000**，开仓 **0.06 手 BTC**（名义 **~$3,900**，约 **39%** 权益）。

1. **最大亏损金额（策略写死）**

   ```
   $10,000 × 1% = $100
   ```

2. **分配到 0.06 BTC 上的最大价格 adverse  move**

   ```
   $100 ÷ 0.06 ≈ $1,666.67 / BTC
   ```

3. **与 4ATR 闭环**

   当时 **4×ATR ≈ $1,666**（单 ATR ≈ **$416**）。  
   BTC $65,000 时，$1,666 跌幅 ≈ **2.56%**。  
   触及 4ATR 止损时：

   ```
   0.06 × $1,666.67 ≈ $100  →  刚好账户 1%
   ```

### 2.2 底层公式：风险反算仓位

机器人逻辑不是「我要买多少钱」，而是：

```
Lots = (Equity × Risk%) / (4ATR 价格绝对宽度 × 每手标的价值)
```

等价于仓库宪法注释：

```
risk_usd = equity × risk_per_slot
qty      = risk_usd / (stop_loss_r × ATR)
notional = qty × price
```

因此 **名义占比** 与 **账户风险 1%** 随 ATR **动态调配**：

| 场景 | 4ATR 空间 | 为亏满 $100 所需手数 | 名义 / $10k | 扫损仍亏 |
|------|-----------|----------------------|-------------|----------|
| **A：低波动** | ~$800（~1.2%） | ~0.125 手 | **~81%** | **1%** |
| **B：高波动** | ~$3,200（~5%） | ~0.031 手 | **~20%** | **1%** |

这正是 **Volatility-Based Position Sizing**：以 ATR 定手数，以权益比例锁单边风险。  
**Yin2ma 账单**（Buy-983 金字塔、0.06 lots 档位）与这一模型一致。

### 2.3 与 event_backtest Bug 的关系

- **风控公式本身**在 `position.py` / 宪法里 **方向正确**（risk 反算 notional）。
- **Bug 在于 `Equity` 项被冻结为 `initial_cash`**，导致：
  - 盈利后仍用 $100/R 开仓 → **名义敞口长期偏低**（median ~0.16x）；
  - CAGR 被 **系统性低估**（例：bull 2024 仅 ~5%）；
  - 与 **复利 lots** 的实盘 / TradingView 回测 **不可比**。

修复后，在 **compound sizing + 正确 equity** 下，低 ATR 阶段名义可达 **~0.4x~0.8x**（与 Yin2ma 早期 0.12 lots 同量级），且 **单笔止损仍 ~1% 权益**——二者不矛盾。

---

## 3. 外部回测对照（避免再混谈）

| | Yin2maBoxTrendBot (Excel) | ml_trading_bot event_backtest (当前) |
|--|---------------------------|--------------------------------------|
| 净利润 | **$103,849** / 583 笔 | TPC bull 2024：**$504** / 176 笔 |
| Lots | **复利**（0.12 → 1.37） | **固定** initial 1% risk |
| 「5%」含义 | Report 初始本金 ~**$2M** 时分母稀释 | **$10k 固定分母 + 固定 sizing** 的字面 5% |
| 若 $10k 复利 | 总收益可达 **~10×+** | **未模拟**（本 Bug） |

---

## 4. 待办（跟踪）

- [x] backtester：`_equity` 变化时同步 `_risk_per_slot_usdt` / `_account_risk_equity`（`scripts/event_backtest/sizing.py`）
- [x] CLI / capital_report：`--no-compound-sizing` 与报表口径说明
- [ ] 加仓 leg `pnl_r` 与 `pnl_usd_realized` 全路径对齐
- [ ] 回归：$10k + compound vs Yin2ma 首仓名义 band

---

*文档版本：2026-05-28 · 与 `feat/research-tools-refactor` 分支 event_backtest 行为一致*
