# 4H 方向类系统：HighCap / Alt / Meme 分组 Token 列表（建议 Universe）

本文件提供一份“够用、可扩展”的数字货币交易 universe 分组，用于：
- 训练/评估：多 symbol 覆盖不同 micro-structure 与 regime（减少“全跟 BTC”导致的有效样本折扣）
- 生产配置：`market_profile` 映射（用于 execution 参数化、风控与监控分组）

> 说明：以下是工程建议清单，不构成投资建议。实际可交易标的以你选择的平台（Binance/OKX/Hyperliquid/dYdX）可用的永续合约为准。

---

## 分组准则（务实）

- **HighCap**：高流动性、盘口更厚、冲击成本更可控、报价连续性更强（更适合当“基准环境”）
- **Alt**：中等流动性、多叙事轮动、波动结构与主流币不同（用于提升 regime 多样性）
- **Meme**：跳跃过程更强、爆量/断流更常见、滑点与冲击更大（用于压力测试与执行专门化）

建议你用这三类作为 `market_profile` 的一级划分（而不是每币一套模型/策略）。

---

## 推荐清单 A：起步版（约 40 个；适合 4H 阶段 0/1）

### HighCap（约 14）

- BTC
- ETH
- SOL
- BNB
- XRP
- ADA
- AVAX
- LINK
- DOT
- LTC
- BCH
- TRX
- TON
- ATOM

### Alt（约 18）

- NEAR
- APT
- SUI
- SEI
- INJ
- ARB
- OP
- UNI
- AAVE
- MKR
- COMP
- FTM
- MANA
- SAND
- RUNE
- RNDR
- STX
- ICP

### Meme（约 8）

- DOGE
- SHIB
- PEPE
- WIF
- BONK
- FLOKI
- BOME
- BRETT

---

## 推荐清单 B：扩展版（约 80–120；适合离线 RL / 更强稳健性验证）

### HighCap（补充候选）

- ETC
- FIL
- XLM
- XTZ
- EOS

### Alt（补充候选）

- DYDX
- GMX
- LDO
- CRV
- SNX
- IMX
- GALA
- ENA
- TIA
- JUP
- JTO
- PYTH
- ZK
- STRK
- BNB 生态/DeFi：CAKE、XVS（若你交易所支持永续）

### Meme（补充候选）

- POPCAT
- MEW
- PONKE
- TURBO
- NEIRO（若你交易所支持永续）

> 备注：Meme 的可交易集随平台与时间变化非常快；扩展清单建议以“交易所永续可用 + 成交量稳定”为硬门槛。

---

## 建议的 `market_profile` 名称与映射格式

建议 `market_profile` 只用 3 个值：`highcap` / `alt` / `meme`（简洁可审计）。

示例 JSON（可直接用于 `mlbot rl build-logs-3action --symbol-profiles-json ...`）：

```json
{
  "BTCUSDT": "highcap",
  "ETHUSDT": "highcap",
  "SOLUSDT": "highcap",
  "ARBUSDT": "alt",
  "OPUSDT": "alt",
  "WIFUSDT": "meme",
  "PEPEUSDT": "meme"
}
```

---

## 实操提醒（避免“有效样本折扣”）

- **不要只堆 Alt**：很多 Alt 在相关性升高时会一起跟 BTC，增加的数据量不等于增加信息量。
- **按组取样**：HighCap/Alt/Meme 三组都要有，才能覆盖执行难度与 regime 多样性。
- **先用 20–50 个跑通闭环**：等 Router/Execution/闸门稳定后，再扩到 80–200。


