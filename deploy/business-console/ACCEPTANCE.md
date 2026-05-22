# Business Console 线上验收清单

## 1. 环境与密钥检查
- [ ] 确认 `/opt/quant-engine/.env` 包含 `BINANCE_API_KEY`, `MULTI_LEG_BINANCE_FUTURES_API_KEY` 等密钥。
- [ ] 访问 `http://<server-ip>:8800/api/health`，检查 `exchange_credentials` 是否显示各 scope 配置状态为 `true`。

## 2. 账户总览页面 (Account Overview)
- [ ] 访问 `/account`，确认页面加载成功。
- [ ] **KPI 卡片**：总账权益、总钱包余额不为 0。
- [ ] **账户层汇总**：
  - [ ] 确认连接状态提示为绿色“已连接 3/3 个币安账户”。
  - [ ] 列表显示 Trend, Spot, Multi-leg，且钱包余额、权益等数据不为 0。
- [ ] **现货持仓明细 (Spot)**：
  - [ ] 显示持仓占比彩色条形图。
  - [ ] 表格列出当前持仓资产（如 BTC, ETH），且“现价”和“市值”计算正确（基于 Parquet 或 API）。
  - [ ] 底部显示“与本地母仓市值差额”，数值应在合理误差范围内。
- [ ] **策略汇总**：
  - [ ] 列表显示各策略的已实现盈亏、浮盈、已平仓、未平仓数量。
  - [ ] 浮盈（Unrealized PnL）不再全是 0，能根据最新价格动态计算。
- [ ] **交易所对账**：
  - [ ] 页面底部展示 A·Spot, B·Trend, C·Multi-leg 三个对账面板。
  - [ ] 若无差异，显示绿色“✓ 交易所与本地数据一致”。
  - [ ] 若有差异，表格列出差异类型（如 数量不符、交易所缺单 等）及具体差额。
