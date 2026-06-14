# A 层多子账户扩展规划

> **状态**：战略规划（2026-06-12）  
> **上位文档**：[ABC三层收益结构_战略框架_CN.md](ABC三层收益结构_战略框架_CN.md) · [产品路线图_TODO优先级_CN.md](产品路线图_TODO优先级_CN.md) · [牛市Beta账本调仓与币本位取舍_CN.md](牛市Beta账本调仓与币本位取舍_CN.md)

---

## 1. 当前 A 层结构

```
A 层（Beta 容器）
└── A·Spot（现货账户）
    ├── spot_accum_simple（主仓：周线 EMA200 深熊 DCA + 利润倍数阶梯卖出）
    └── profit_satellite（卫星：利润 1% / 周 → Binance 24h 热榜 Top1，见 §3.1.1）
```

**现状**：
- 仅 1 个现货账户（`spot` scope）
- 主仓：`spot_accum_simple`（周线 EMA200 深熊 DCA + 利润倍数阶梯卖出）
- 卫星：`profit_satellite` **设计稿**（§3.1.1），未 live
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
│   ├── spot_accum_simple（主仓）
│   └── profit_satellite（卫星，利润周冲强势币 — 优先于 rolling live）
├── A·Futures（U 本位合约）— ⏸ 后置
│   └── rolling_trend（组合级杠杆滚仓；见 §3.2，非 A 卫星 MVP 首选）
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

### 3.1.1 A·Spot 卫星 — `profit_satellite`（利润周冲强势币）【设计 · 优先 MVP】

> **动机**：比 `profit_ladder` 多档梯子、`rolling_trend` 合约滚仓更简单；与「定投主仓 + 手动追叙事」相比，多了 **利润池纪律 + 客观选币 + regime 门控**。  
> **与 B·TPC 浮盈加仓**：TPC `add_position` 是 **同标的、per-trade** 金字塔；卫星是 **跨标的、只用利润、每周至多一笔**。

#### 规则（草案）

| 项 | 约定 |
|----|------|
| **资金来源** | 现货主仓 **利润池**（浮盈或已实现利润，可配置）；**不动本金** |
| **部署比例** | 每周 `profit_pool × 1%`（`deploy_frac=0.01`）买入 **1 个**候选币（现货市价/限价） |
| **选币** | **Binance Spot** `ticker/24hr` 全市场 → 筛 `*USDT` → 按 24h `priceChangePercent` **Top1**（见探针 §实测） |
| **频率** | 每周 1 次（建议 UTC 周一或周日收盘后） |
| **Regime 门控** | 仅当 T2 `composite=risk_on`（或 `abc_macro_regime_score ≥ 3`）执行；`risk_off` 跳过并禁止新开 |
| **上限** | 卫星持仓合计 ≤ 总 NAV **10–15%**；单币 ≤ **5%**；meme/超小市值另设 **2%** 硬顶 |
| **出场** | 无固定倍数止盈亦可；至少：**父级主仓 regime 转 off → 先减卫星**；可选 `market_heat.state=COLD` 或持仓跌破周线结构时清仓卫星 |

**利润池计算（示意）**：

```text
profit_pool = max(0, spot_principal_market_value - spot_cost_basis)
weekly_deploy_usdt = min(profit_pool * 0.01, tier_cap_remaining, single_coin_cap)
```

#### 选币：Binance Spot 24h 热榜（P0 已探针，一次请求）

**原则**：不对全 USDT 逐币拉日 K；用公开 **`GET /api/v3/ticker/24hr`**（不传 `symbol`）一次拿全市场，本地筛 USDT + 排序。权重 80，适合周频。

**探针（已跑通 2026-06-14）**：

```bash
python scripts/profit_satellite_probe.py --limit 10 --profit-pool-usdt 5000
```

| 项 | 约定 |
|----|------|
| 接口 | `https://api.binance.com/api/v3/ticker/24hr?symbolStatus=TRADING` |
| 排序 | `priceChangePercent` 降序 |
| 过滤 | `quoteVolume ≥ 1M USDT`；排除稳定币对、`UP/DOWN/BULL/BEAR` 杠杆代币 |
| 窗口 | **24h 滚动**（非自然日、非 7 日） |

**实测 Top1（2026-06-14 UTC）**：`ZKCUSDT` +30.12%；利润池 5000 USDT × 1% → **50 USDT** 名义。完整 Top10 见 `config/strategies/profit_satellite/README.md`。

**代码**：`src/market_momentum/binance_spot_24h.py` · `scripts/profit_satellite_probe.py`

#### 备选：第三方 7 日 momentum（后置）

若需 **周涨幅** 而非 24h，可再接 CoinGecko（仓库已有先例），不作为 P0：

```http
GET /api/market/momentum?days=7&limit=20
```

| 参数 | 说明 |
|------|------|
| `days` | 回看天数，`7` ≈ 当周/近一周；允许 `1\|7\|30` |
| `limit` | 返回条数，默认 20 |

**响应骨架（示意）**：

```json
{
  "ok": true,
  "data": {
    "provider": "coingecko",
    "as_of": "2026-06-12T08:00:00Z",
    "days": 7,
    "rows": [
      {
        "symbol": "SOL",
        "pair_hint": "SOLUSDT",
        "return_pct": 18.4,
        "market_cap_usd": 85000000000,
        "volume_usd_24h": 1200000000,
        "rank": 1
      }
    ]
  }
}
```

**第三方数据源（备选 CoinGecko）**：

| 提供方 | 用途 | 仓库锚点 |
|--------|------|----------|
| **CoinGecko** | `/coins/markets` 按 `price_change_percentage_7d`（或 `30d`）排序；带市值/成交量过滤 | `src/market_heat/sync_sectors.py`、`config/market_cap/market_cap.yaml` |
| 备选 | CoinMarketCap / 其他付费 API | 未接入；需单独评估配额与 ToS |

**过滤（在第三方结果上叠加，非扩币种扫描）**：

1. `market_cap_usd ≥` 阈值（如 50M，与 `sync_sectors` 一致）  
2. `volume_usd_24h ≥` 阈值（去假拉盘）  
3. 映射到 **Binance Spot 可交易** `*USDT`（`pair_hint` 可解析性检查）  
4. 可选：Top1 同时要求 `market_heat.state != COLD`（慢趋势滤镜，接口仍用现有 `run_heat_update`，与 momentum **并列**）

**明确不做**：

- ❌ Binance `fetch_ohlcv` 对全市场 USDT 对算 `days` 日涨幅  
- ❌ 把 `market_heat`（周线 EMA50 趋势）当作「当周涨幅榜」唯一来源  

#### 执行形态（分阶段）

| 阶段 | 交付 |
|------|------|
| **P0** | `profit_satellite_probe.py` + 每周 TG/笔记 Top5；**人工**下单 |
| **P1** | 利润池会计 + regime 门控 + 半自动「确认后下单」 |
| **P2** | 全自动每周 `profit_pool×1%` 买 Top1（仍受上限约束） |

#### 配置锚点（规划）

| 用途 | 路径 |
|------|------|
| 卫星策略说明 | `config/strategies/profit_satellite/README.md` |
| 比例/上限/regime | `config/strategies/profit_satellite/satellite.yaml` |
| Binance 24h 探针 | `scripts/profit_satellite_probe.py`、`src/market_momentum/binance_spot_24h.py` |
| Momentum 第三方（备选） | `config/market_momentum/providers.yaml`（待建，coingecko 7d） |
| CMS 路由 | `src/mlbot_console/routers/market.py` → `GET /api/market/momentum`（备选，待建） |
| 与 T2 联动 | `config/monitoring/rebalance_targets.yaml` `risk_on` 时允许卫星；`risk_off` 告警清仓 |

---

### 3.2 A·Futures（U 本位合约）— ⏸ 后置（原 P1 rolling）

| 属性 | 值 |
|------|-----|
| **账户类型** | Binance Futures USDT-M |
| **结算货币** | USDT |
| **策略** | `rolling_trend`（组合级杠杆滚仓） |
| **时间尺度** | 周～月（TPC 信号驱动） |
| **核心语义** | 杠杆化趋势 beta，与现货 A 语义一致但工具不同 |
| **优先级** | **⏸ 后置**（现货 `profit_satellite` 更简单且已覆盖大部分「利润追强势」需求；若 simulate 证明显著增量再 live） |

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
- 提高 A·Spot 主仓占比；`profit_satellite` 在 risk_on 下每周利润 **1%** 追 24h 热榜 Top1
- A·Futures（若上线）再叠加杠杆 beta
- A·CN 可作为跨市场分散

熊市：
- **关闭** `profit_satellite` 新开；优先清卫星
- 收缩 A·Futures（降杠杆或暂停）
- A·Spot 主仓降权（不新开 deploy；`spot_accum` 深熊逻辑照旧）
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

| 子账户 / 模块 | 优先级 | 依赖 | 工作量 |
|---------------|--------|------|--------|
| A·Spot 主仓 | — | — | 已有 |
| A·Spot `profit_satellite` | **P1** | T2 risk_on 门控 + Binance 24h 探针已通 | 小（利润池会计 + 半自动下单） |
| A·Futures rolling | **⏸ 后置** | profit_satellite 跑通 ≥1 周期 + rolling simulate 增量 | 中（2-4 周） |
| A·CN | **P3** | T3 Phase A heatmap | 大（数据 + 合规 + 回测脚手架） |
| A·Coin-M | **P4** | A 现货 + T2 已运行 ≥1 个 regime 周期 | 很大（4-8 周+） |

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
| A·Spot 主仓 | `config/constitution/constitution.yaml` §5 spot、`config/strategies/spot_accum_simple/` |
| A·Spot 卫星 | `config/strategies/profit_satellite/`、`scripts/profit_satellite_probe.py` |
| A·Futures | `config/strategies/rolling_trend/`、`scripts/rolling_trend_simulate.py`、新增 `rolling` scope |
| A·CN | 新建 `src/market_heat_cn/` 或扩展 `market_heat`、`docs/market_heat/` |
| A·Coin-M | `src/order_management/binance_api.py`（`dapi`） |

---

## 9. 一句话结论

> **A 层应扩展为多子账户结构**：现货主仓（已有）+ **现货利润卫星 `profit_satellite`（P1，Binance 24h 热榜 + 利润 1%/周）** + U 本位合约 rolling（后置）+ A 股（P3）+ 币本位（P4 可选）。核心是 **payoff 语义统一（beta）**；卫星只用利润、一次 ticker/24hr 请求。

---

*维护：新增 A 层子账户时更新 §2 结构与 §6 优先级，并注明依赖变更。*
