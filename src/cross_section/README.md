# Cross-Section / Pairs Trading 研究线

**动机**：现有策略 (me/bpc/tpc/srb) 在牛熊大行情能吃到 fat-tail，但 (fer/msr/crf) 在震荡区难盈利。
本模块探索 **脱离单边方向依赖** 的稳定 alpha 来源：pairs trading、cross-sectional long/short、
多因子合成。

此模块独立于 `time_series_model`（后者聚焦单币时间序列建模），因为多因子横截面是完全不同的建模范式。

## 实验索引

| Exp | 内容 | 规模 | 状态 |
|---|---|---|---|
| [exp01](./exp01_correlation/) | 相关性 + 协整 + XS 动量基线 | 6 主流币 | ✅ pairs trading 不成立；XS 动量 Net SR 0.75 |
| [exp02](./exp02_multi_factor/) | 多因子 + 板块中性 L/S | 57 币 / 10 板块 | ✅ 基础设施 |
| [exp03](./exp03_ic_and_grid/) | 因子 IC 分析 + grid search | 57 币 × 12 因子 × 3 horizon | ✅ 定位可用因子 |
| [exp04](./exp04_small_account/) | 小资金版（1w USD, 少持仓、长周期） | 20 流动性币 | ✅ 针对 $10k 账户 |
| [exp04 batch](./exp04_small_account/run_batch.py) | 全样本 3 preset × 5 period + regime attribution | 2023–2026Q1 | ✅ 回答 "mom_only 1.81 是否特例" |
| [exp05](./exp05_regime_ic/) | target-horizon IC (14d) + regime-conditional factor rotation | 20 币 × 12 因子 × 5 regime | ✅ regime_weights.yaml 生成 |
| [exp05 v2](./exp05_regime_ic/run_walk_forward_oos.py) | walk-forward OOS 权重（无 look-ahead） | 默认 180d 窗；可选 ALL-only / 降 refit 频率 | ✅ `walk_forward/` |
| [exp07](./exp07_paper_trading/) | offline walk-forward paper trading simulator | $10k virtual, 14d hold | ✅ 支持 `--use-regime-switch` |

## 数据依赖

- 价格：`data/parquet_data/<SYMBOL>_<YYYY-MM>.parquet`（聚合 tick，重采样到 K 线）
- Funding：`data/funding_rate/parquet/<SYMBOL>_<YYYY-MM>_funding_rate.parquet`

## 已发现的关键 insight

1. **币圈简单 pairs trading 不成立**（exp01）：主流币对协整 p 值都 > 0.05，半衰期 > 1000h。
2. **板块中性化 ≠ 免费午餐**（exp02, exp03 grid）：单边牛市里 sector_neutral 常反而减收益，
   但多年样本下应提升稳定性（需要 2023+2024+2025 联合验证）。
3. **因子 IC 随 horizon 变化巨大**（exp03 vs exp04）：low_vol 在 24h horizon IC 最好，
   但在 14-day 持仓时 momentum-only 却能拿到 Sharpe 1.81（2024 样本）——
   **IC 必须在目标交易 horizon 上测量**，这是 exp05 的主题。
4. **换手成本是主要 alpha 杀手**（exp02）：gross SR 0.74 扣费后变 0.20。
   必须保持持仓周期 >= 3 天才能守住大部分 alpha。
5. **mom_only 1.81 是 2024 牛市特例**（exp04 batch, full sample）：
   Full Net SR 仅 **0.52**，2025 混合期 **-0.82**。不能作为实盘 preset。
6. **14d horizon 下 low_vol 最强**（exp05 Part 1）：IR 0.31–0.38，远超 momentum (0.09) 和 reversal (-0.02)。
   exp03 用 1d horizon 选错了因子。
7. **regime-specific 因子差异显著**（exp05 Part 2）：
   - bear: 纯 low_vol (IR 0.7)
   - range_reversal (空头拥挤+震荡): 所有因子 IR > 1，low_vol 可达 3.5（但样本少）
   - bull_normal: rev_24h 在此 regime 翻正 (+0.07)，全样本 -0.02
8. **IC-weighted static combo 是目前最稳的上线候选**（exp05 Part 3）：
   Full sample Net SR **1.01**（35% AnnRet），胜 regime_switch (0.79)、胜 mom_only (0.21)。
   regime 信号的 alpha 被换因子噪声抵消；实盘前应先上 static IC combo。

## 下次可做

- **exp05 v2**：`run_walk_forward_oos.py` 默认 **180d** 训练窗；可调 `--refit-every-n`、`--all-weights-only`、`--hold-bars`
- **exp06**：vol regime filter（BTC realized vol > 某阈值时降杠杆）
- **exp08**：三角协整 / VECM（多资产联合共整合，超越简单 pair）
- **exp09**：多 timeframe 组合（短 lookback 在震荡、长 lookback 在趋势）
- **exp07 v2**：paper engine 加期中止损 + CCXT 真实价源轮询

## 与现有单币策略的关系

横截面 L/S 产生的是**板块/币种相对强弱的 alpha**，和单币时序模型（me/bpc/crf 等）
应该是**不相关收益源**，理想情况下可以组合运行（资金各分一部分）做整体 Sharpe 提升。
