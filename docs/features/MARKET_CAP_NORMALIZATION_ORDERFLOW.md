# MarketCap 归一化订单流：为什么“除以成交量”可能错，如何在本项目里验证？

本文是对你提供的观点（用 **市值 MarketCap** 替换 **成交额/成交量 Volume** 作为订单流归一化分母）做“落地版”解读，并给出本项目中的实现与验证路径。

## 1) 这篇文章的核心逻辑是否有道理？

总体上 **有道理**，但要分“研究目标”和“资产/数据结构”：

- **文章要解决的问题**：把“订单流信号强度”从“流动性热度(Volume)”的干扰中解耦出来，使得信号更接近“知情交易强度”的尺度。
- **关键假设**：
  - **知情交易者的规模**更像按“资产容量/市值”缩放（大盘承载大资金）。
  - **噪声交易者的活跃**更像按“当期成交量/注意力”缩放（热门/高换手更吵）。
- **因此**：用 Volume 做分母相当于把信号乘上 \(1/\text{Turnover}\)，会把低换手资产放大、把高换手资产压扁，引入横截面异方差；用 MarketCap 作为分母更接近把信号“还原到同一尺度”。

在传统股票横截面因子里，这个观点尤其常见且容易成立（市值与容量/冲击成本/机构可交易规模强相关，换手率横截面差异极大）。

## 2) 在 Crypto 上是否也成立？

**可能成立，但需要更谨慎**：

- Crypto 的“市值”本质上是 \( \text{Price} \times \text{CirculatingSupply} \)，并非基本面估值；但它仍然刻画了“容量/承载规模”。
- Crypto 的成交量/换手同样高度异方差（并且更受交易所结构/做市/活动影响），所以 **用市值做尺度对齐**仍有潜在价值，尤其在：
  - **多币种训练（multi-symbol）**：把不同币种的订单流强度映射到可比尺度。
  - **长时间跨度**：减少“成交量热度周期”对订单流强度的误导。

不过：如果你只做 **单一币种**、且订单流 proxy 本身已经是相对尺度（比如 zscore/quantile），增益可能更小。

## 3) 本项目中的落地实现（我们做了什么）

我们新增了一个特征节点：`market_cap_normalized_orderflow_f`，输出以下列：

- `market_cap_usd`: 日频市值（来自 CoinGecko 历史 market cap）
- `dollar_volume_over_mcap`: \( \text{close} \times \text{volume} / \text{mcap} \)
- `turnover_over_mcap`: \( |\text{close} \times \text{volume}| / \text{mcap} \)
- `net_buy_usd_over_mcap`: “净买入金额 proxy” / mcap  
  - 优先用 `buy_qty - sell_qty`（如果 parquet 里有）
  - 否则用 `cvd_change_1` 或 `diff(cvd)`（如果有）
- `abs_net_buy_usd_over_mcap`: \(|\text{net_buy_usd}| / \text{mcap}\)

对应实现代码在：
- `src/features/time_series/market_cap_features.py`
- 配置接入在 `config/feature_dependencies.yaml` 的 `market_cap_normalized_orderflow_f`

并且我们把它作为候选组加入了四个策略的 group YAML：
- `config/feature_groups_sr_reversal_semantic.yaml`
- `config/feature_groups_sr_breakout_semantic.yaml`
- `config/feature_groups_compression_breakout_semantic.yaml`
- `config/feature_groups_trend_following_semantic.yaml`

组名：`market_cap_norm`

## 4) 市值数据如何获取与更新（免费接口）

我们使用 **CoinGecko** 的公共 API（无需 key 的模式也能用，但有频率限制）。

配置文件：
- `config/data/market_cap.yaml`

更新命令（会写入 `data/market_cap/<SYMBOL>.parquet`）：

```bash
python3 scripts/update_market_cap.py \
  --config config/data/market_cap.yaml \
  --write-manifest
```

说明：
- 默认会从 `config/data/market_cap.yaml` 读取 `universe_yaml` + `universe_set`，
  自动把该 universe 的所有 token（如 BTC/ETH/SOL/...）拼成 `BTCUSDT/ETHUSDT/...` 并更新市值数据。
- 默认会把 **自动 search 解析出的 `coingecko_id` 固定写回**到 `config/data/market_cap.yaml`（并生成备份），避免下次再猜测导致冲突。
- 如果你还没下载某个币种的 OHLC parquet（例如 `SOLUSDT`），即使市值更新好了，多币种训练也会因 OHLC 缺失而失败。

## 5) 如何验证它是否“真有用”（推荐实验方式）

推荐使用我们 Layer B 的 `feature-group-search`：

- **单策略验证**（例如 `sr_reversal`）：
  - baseline（不加新组）
  - step1 candidate：`market_cap_norm`
  - 多 seed（1..5）观察 Sharpe_mean、trades_mean 是否稳定提升

- **多币种稳定性验证**（关键！）：
  - `--symbol BTCUSDT,ETHUSDT,(SOLUSDT...)`
  - 看它是不是在多币种下更稳定、更能跨币种泛化

如果该组在四个策略中呈现“只对某类策略有效”，那也正常：它更多是“尺度对齐/稳健化”工具，而不是对所有策略都提供同方向 alpha。

## 6) 与架构文档的关系

这属于 `EXPERIMENT_LOOP_ARCHITECTURE.md` 中：
- Layer B（Feature Search）的“可解释特征工程 + 分组验证”
- 以及 “跨资产/跨时间尺度对齐（cross-sectional normalization）” 的实践案例


