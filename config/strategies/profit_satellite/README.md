# profit_satellite（A·Spot 利润卫星）

> **状态**：**已放弃 live**（2026-06-14）· 探针归档  
> **上位文档**：[A层多子账户扩展规划_CN.md](../../docs/strategy/A层多子账户扩展规划_CN.md) §3.1.1  
> **替代主攻**：[产品路线图 §T5](../../docs/strategy/产品路线图_TODO优先级_CN.md) — B 层订单墙 / 清算 scan

2026-06-14 战略决定：**不立项** P0 周报运营、P1 利润池/regime/半自动、P2 全自动。  
下列内容为历史探针记录，**不接入 live / CMS**。

---

## 归档：规则（测试版）

| 项 | 约定 |
|----|------|
| 资金来源 | 现货主仓 **利润池**（不动本金） |
| 部署比例 | 每周 `profit_pool × 1%`（`deploy_frac: 0.01`）买入 **Top1** |
| 选币 | **Binance Spot** `GET /api/v3/ticker/24hr` → 筛 `*USDT` → 按 `priceChangePercent` 降序 |

## 归档：探针命令

```bash
python scripts/profit_satellite_probe.py --limit 10 --profit-pool-usdt 5000
```

代码：`src/market_momentum/binance_spot_24h.py` · `scripts/profit_satellite_probe.py`

## 2026-06-14 实测快照（UTC 13:53）

| # | symbol | 24h% |
|---|--------|------|
| 1 | ZKCUSDT | +30.12% |

（完整 Top10 见 git 历史 / 探针输出）
