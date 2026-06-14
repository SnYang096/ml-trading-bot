# profit_satellite（A·Spot 利润卫星）

> **状态**：设计 + **P0 探针已跑通**（2026-06-14）  
> **上位文档**：[A层多子账户扩展规划_CN.md](../../docs/strategy/A层多子账户扩展规划_CN.md) §3.1.1

## 规则（测试版）

| 项 | 约定 |
|----|------|
| 资金来源 | 现货主仓 **利润池**（不动本金） |
| 部署比例 | 每周 `profit_pool × 1%`（`deploy_frac: 0.01`）买入 **Top1** |
| 选币 | **Binance Spot** `GET /api/v3/ticker/24hr` → 筛 `*USDT` → 按 `priceChangePercent` 降序 |
| 过滤 | `symbolStatus=TRADING`；`quoteVolume ≥ 1M USDT`；排除稳定币对、杠杆代币（`*UP/*DOWN/*BULL/*BEAR`） |
| 频率 | 每周 1 次（UTC 周一或周日收盘后） |
| Regime | 仅 T2 `risk_on` 执行（未接 live 前人工判断） |

```text
profit_pool = max(0, spot_market_value - spot_cost_basis)
weekly_deploy_usdt = min(profit_pool * 0.01, tier_cap_remaining, single_coin_cap)
```

## 探针命令

```bash
# 表格 Top10 + 示例 deploy（利润池 5000 USDT → 本周 50 USDT）
python scripts/profit_satellite_probe.py --limit 10 --profit-pool-usdt 5000

# JSON（可管道给 TG / CMS）
python scripts/profit_satellite_probe.py --limit 20 --profit-pool-usdt 5000 --format json
```

代码：`src/market_momentum/binance_spot_24h.py` · `scripts/profit_satellite_probe.py`

## 2026-06-14 实测快照（UTC 13:53）

探针：`--limit 10 --profit-pool-usdt 5000`（`deploy_frac=0.01` → **50 USDT**）

| # | symbol | 24h% | quoteVol (USDT) |
|---|--------|------|-----------------|
| 1 | ZKCUSDT | +30.12% | 9,468,707 |
| 2 | SYNUSDT | +26.73% | 5,508,312 |
| 3 | BANANAS31USDT | +23.36% | 7,800,302 |
| 4 | MITOUSDT | +22.66% | 8,799,159 |
| 5 | OPGUSDT | +20.91% | 14,577,369 |
| 6 | MEGAUSDT | +15.69% | 30,432,157 |
| 7 | JASMYUSDT | +12.29% | 2,836,770 |
| 8 | MEMEUSDT | +10.87% | 2,791,711 |
| 9 | ATUSDT | +10.29% | 2,619,207 |
| 10 | CHIPUSDT | +8.86% | 7,287,884 |

**本周 Top1（若人工跟单）**：`ZKCUSDT`，名义约 **50 USDT**（利润池 5000 × 1%）。

> 24h 滚动窗口，非 7 日周涨幅；小市值/memecoin 常上榜，上线前建议叠加 `market_heat != COLD` 或市值硬顶。

## 配置

见 `satellite.yaml`：`deploy_frac`、`min_quote_volume_usdt`、`regime_gate` 等。

## 分阶段

| 阶段 | 交付 |
|------|------|
| **P0** ✅ | 探针 + 文档快照；每周 TG 列 Top5，**人工**下单 |
| **P1** | 利润池会计 + T2 regime 门控 + 半自动确认下单 |
| **P2** | 全自动 `profit_pool×1%` 买 Top1（仍受 NAV 上限约束） |
