# A 层多子账户扩展规划

> **状态**：战略规划（2026-06-12）  
> **上位文档**：[ABC三层收益结构_战略框架_CN.md](ABC三层收益结构_战略框架_CN.md) · [产品路线图_TODO优先级_CN.md](产品路线图_TODO优先级_CN.md) · [牛市Beta账本调仓与币本位取舍_CN.md](牛市Beta账本调仓与币本位取舍_CN.md)

---

## 1. 当前 A 层结构

```
A 层（Beta 容器）
└── A·Spot（现货账户）
    └── spot_accum_simple（周线 EMA200 DCA）
```

**现状**：
- 仅 1 个现货账户（`spot` scope）
- 策略单一：`spot_accum_simple`（周线 EMA200 深熊 DCA + 利润倍数阶梯卖出）
- 账户隔离：与 B（trend）、C（multi_leg）物理分离
- 风控独立：`max_gross_notional_pct: 1.00`、`max_daily_deploy_pct: 1.00`

**代码位置**：
- 配置：`config/constitution/constitution.yaml` §5 Spot accumulator
- 余额查询：`src/mlbot_console/services/exchange_balances.py`（`spot` scope）
- 策略：`config/strategies/spot_accum_simple/`

---

## 2. 建议扩展结构

```
A 层（Beta 容器 — 慢 regime 参与）
├── A·Spot（现货）— ✅ 已有
│   └── spot_accum_simple
├── A·Futures（U 本位合约）— 🆕 rolling_trend
│   └── rolling_trend（组合级杠杆滚仓，TPC 信号源）
├── A·CN（A 股）— 🆕 T3
│   └── 慢周期高性价比（月线趋势 + 低估值）
└── A·Coin-M（币本位）— 🆕 T4（可选，后置）
    └── BTC 币本位 beta 容器
```

---

## 3. 子账户详细规划

### 3.1 A·Spot（现货）— 已有

| 属性 | 值 |
|------|-----|
| **账户类型** | Binance Spot |
| **结算货币** | USDT（买入现货持币） |
| **策略** | `spot_accum_simple` |
| **时间尺度** | 周～月（慢 DCA + 慢出场） |
| **核心语义** | 现货持币、抓右尾、无杠杆 |
| **优先级** | — |

**优势**：
- 无爆仓风险
- 牛市持币享受价格 beta
- 与 B/C 合约账户物理隔离

**局限**：
- 无法加杠杆（牛市 beta 暴露上限 = 本金）
- 无「抵押品随币价升值」的复利效应（对比币本位）

---

### 3.2 A·Futures（U 本位合约）— 🆕 P1

| 属性 | 值 |
|------|-----|
| **账户类型** | Binance Futures USDT-M |
| **结算货币** | USDT |
| **策略** | `rolling_trend`（组合级杠杆滚仓） |
| **时间尺度** | 周～月（TPC 信号驱动） |
| **核心语义** | 杠杆化趋势 beta，与现货 A 语义一致但工具不同 |
| **优先级** | **P1**（路线图 T1） |

**为何独立账户**：
1. **风控隔离**：杠杆爆仓风险不应污染现货 A
2. **宪法分层**：`max_gross_leverage` 需独立配置（现货 = 1.0，合约 = 2.0~3.0）
3. **NAV 追踪**：便于 T2 调仓告警区分「无杠杆 beta」vs「杠杆 beta」

**代码现状**：
- 配置：`config/strategies/rolling_trend/`（README + archetypes + features.yaml + meta.yaml）
- 模拟：`scripts/rolling_trend_simulate.py`
- **缺失**：live runner、独立 API key、宪法 `rolling` 段

**实现路径**：
1. 新增 `rolling` scope 到 `exchange_balances.py`
2. 宪法新增 `rolling:` 段（独立 `equity_usdt`、`max_gross_leverage`）
3. 开发 `scripts/run_rolling_live.py`（参考 `run_multi_leg_live.py`）
4. CMS 卡片：rolling 账户净值、杠杆、回撤

**风险**：
- 高杠杆爆仓语义需与宪法 `max_gross_leverage` 分层
- TPC 信号源订阅与组合杠杆维护逻辑

---

### 3.3 A·CN（A 股）— 🆕 P3

| 属性 | 值 |
|------|-----|
| **账户类型** | A 股券商（待定：tushare / akshare / 付费 API） |
| **结算货币** | CNY |
| **策略** | 慢周期高性价比（月线趋势 + 低估值 + 低换手） |
| **时间尺度** | 月～季（慢变量筛选） |
| **核心语义** | 跨市场 beta 分散，宏观配置 |
| **优先级** | **P3**（路线图 T3） |

**为何归 A 层**：
- 慢周期、长持仓、宏观配置，完全符合 A 层「slow regime participation」定位
- 与 crypto A 层语义一致（抓 beta），只是市场不同

**代码现状**：
- `src/market_heat/` 仅覆盖 crypto（sector_registry + heat_calculator）
- **无** A 股数据源接入
- **无** 券商 API 集成

**实现路径（分两阶段）**：

*Phase A — 研究与可视化*
1. A 股板块/行业 heatmap（数据源选型）
2. 与 crypto `market_heat` 同语义（HOT/WARM/COLD）或独立 dashboard
3. 慢变量筛选 → 观察清单（非自动下单）

*Phase B — 与 ABC 接线（可选，后置）*
1. macro regime 一维输入 T2 调仓告警
2. 小仓位 ETF/个股 pilot（合规与券商 API 单列评估）

**风险**：
- T+1、涨跌停、做空受限
- 与 crypto 24/7 运维模型不同
- 合规与数据成本

---

### 3.4 A·Coin-M（币本位）— 🆕 P4（可选）

| 属性 | 值 |
|------|-----|
| **账户类型** | Binance Futures COIN-M（`dapi`） |
| **结算货币** | BTC/ETH 等（币本位） |
| **策略** | BTC 币本位 beta 容器 |
| **时间尺度** | 周～月 |
| **核心语义** | 持币复利（牛市抵押品升值 + 多头盈利） |
| **优先级** | **P4**（路线图 T4，后置） |

**为何归 A 层**：
- 币本位 beta 容器，语义更接近 A 层的「持币复利」，而非 B/C 的 alpha 引擎
- 文档明确不推荐 B/C 全量迁移币本位（见 [币本位取舍 doc](牛市Beta账本调仓与币本位取舍_CN.md)）

**为何后置**：
1. **工程量大**：`dapi` 全栈缺失（符号体系 `BTCUSD_PERP`、PnL 单位、宪法 `equity_usdt`、SL/reconcile 均需重做）
2. **运维复杂**：多币种 B/C 需分币抵押池（BTC 仓用 BTC、ETH 仓用 ETH），无法像 USDT 池统一调度
3. **替代方案已覆盖**：现货 A + U 本位 rolling 已能覆盖大部分 beta 需求
4. **熊市风险**：币价下跌 → 抵押品贬值 + 多头亏损，双重打击

**前置条件**：
- T1 rolling + A 现货 + T2 调仓已运行 ≥1 个 regime 周期
- 验证「持币复利」是否真的优于「现货 A + U 本位 rolling」

**实现路径（若仍要做）**：
1. **仅 BTC** 币本位子账户（`dapi`），语义 = beta 容器
2. `binance_api` 抽象：`MarketKind.USDT_M | COIN_M`
3. PnL / 宪法 / CMS 币本位或统一折算 USDT 显示

---

## 4. 关键原则

### 4.1 A 层 = Beta 容器，不区分工具

A 层的核心是 **payoff 语义统一（慢、长、凸性）**，而非工具统一：
- 现货（A·Spot）：无杠杆 beta
- 合约（A·Futures）：杠杆 beta
- A 股（A·CN）：跨市场 beta
- 币本位（A·Coin-M）：持币复利 beta

### 4.2 物理隔离

每个子账户独立：
- API key
- 风控参数（宪法段）
- NAV 追踪
- CMS 卡片

便于 T2 调仓告警按子账户维度监控。

### 4.3 Regime 驱动权重

牛市：
- 提高 A·Spot + A·Futures 占比
- A·CN 可作为跨市场分散

熊市：
- 收缩 A·Futures（降杠杆或暂停）
- A·Spot 降权（不新开或极低频试错）
- A·CN 视宏观 regime 调整

### 4.4 币本位非必需

现货 A 已能抓 beta，U 本位 rolling 已能加杠杆。币本位的「抵押品升值」是锦上添花，不是雪中送炭。

---

## 5. 与 T2 调仓告警的关系

T2 调仓告警需监控各 A 层子账户的 NAV 占比：

```yaml
# config/monitoring/rebalance_targets.yaml（示意）
a_layer:
  target_nav_pct:
    spot: 0.40      # 现货 A
    futures: 0.30   # rolling_trend
    cn: 0.20        # A 股（若上线）
    coin_m: 0.10    # 币本位（若上线）
  tolerance_pct: 0.05  # ±5% 容忍带
```

告警逻辑：
1. 定时拉各子账户 equity
2. 算占比 vs 目标带
3. 对比 `abc_macro_regime_score` 或 TPC `bull_share`
4. 偏离超阈值 → `WATCH` / `REBALANCE_SUGGEST` → CMS + TG

---

## 6. 实现优先级与依赖

| 子账户 | 优先级 | 依赖 | 工作量 |
|--------|--------|------|--------|
| A·Spot | — | — | 已有 |
| A·Futures | **P1** | T1 rolling live | 中（2-4 周） |
| A·CN | **P3** | T3 Phase A heatmap | 大（数据 + 合规 + 回测脚手架） |
| A·Coin-M | **P4** | T1 + A 现货 + T2 已运行 ≥1 个 regime 周期 | 很大（4-8 周+） |

---

## 7. 显式不做 / 延后

| 项 | 原因 |
|----|------|
| A·Coin-M 作为 B/C **全量**币本位迁移 | 战略错层 + 工程量大；见币本位取舍 doc |
| A·CN 与 A·Futures **同优先级**抢资源 | A 股合规与数据未验证，不应挡 crypto beta live |
| A·Futures 未验证直接 live | 违反 experiments R&D workflow Phase 1 |

---

## 8. 配置锚点（落地时改这些）

| 子账户 | 主要路径 |
|--------|----------|
| A·Spot | `config/constitution/constitution.yaml` §5 spot、`src/mlbot_console/services/exchange_balances.py`（`spot` scope） |
| A·Futures | `config/strategies/rolling_trend/`、`scripts/rolling_trend_simulate.py`、新增 `rolling` scope |
| A·CN | 新建 `src/market_heat_cn/` 或扩展 `market_heat`、`docs/market_heat/` |
| A·Coin-M | `src/order_management/binance_api.py`（`dapi`） |

---

## 9. 一句话结论

> **A 层应扩展为多子账户结构**：现货（已有）+ U 本位合约（rolling_trend，P1）+ A 股（T3，P3）+ 币本位（T4，P4 可选）。核心是 **payoff 语义统一（beta）**，工具/市场分散。

---

*维护：新增 A 层子账户时更新 §2 结构与 §6 优先级，并注明依赖变更。*
