# 实验 1：相关性 / 协整 / XS 动量基线

**目标**：判断"配对交易 (pairs trading)" 或 "横截面多空 (cross-sectional long/short)" 是否能在主流币
上提供脱离 fat-tail 大行情的稳定盈利。

**数据源**：`data/parquet_data/<SYMBOL>_<YYYY-MM>.parquet`（逐笔聚合 tick），重采样为 1H K线 close。

## 脚本

`analyze_cross_section_correlation.py`

```bash
python -m src.cross_section.exp01_correlation.analyze_cross_section_correlation \
    --symbols BTCUSDT ETHUSDT SOLUSDT ADAUSDT XRPUSDT BNBUSDT \
    --start 2023-01 --end 2024-12 --timeframe 1h \
    --outdir reports/cross_section/exp01
```

## 方法

1. **全样本 & 分状态相关性矩阵**：按年 / 按 BTC 月度涨跌分段。
2. **滚动 30 天相关性**：观察是否稳定。
3. **Engle-Granger 协整检验**（所有 pair）：p<0.05 是 pairs trading 的前提。
4. **价差均值回归半衰期**：对最显著协整 pair OLS 估计 hedge ratio β，AR(1) 求半衰期。
5. **XS 动量 L/S 回测**：过去 7 天累计收益排序，top-K 做多 / bottom-K 做空，等权美元中性，
   每天 rebalance，同时输出 gross / net (扣 5bp 单边费) 两条净值曲线。

## 关键结论（BTC/ETH/SOL/ADA/XRP/BNB, 2023–2024, 1H）

| 指标 | 值 | 含义 |
|---|---|---|
| 平均非对角相关性 | **0.65** | 币圈高度共动 |
| 牛市 / 熊市相关性 | 0.63 / 0.71 | 熊市相关性更高（"down together"） |
| BTC-ETH 相关性 | **0.83** | 太高，做 pair 没价差可赚 |
| 协整 pair 数 | **0 / 15** | 简单 pairs trading 在主流币上不成立 |
| 价差半衰期 | 900–3000 小时 | 太长（>40 天），实际不可交易 |
| XS 动量 Gross Sharpe | 1.07 | 扣费前有 alpha |
| XS 动量 **Net Sharpe** | **0.75** | 年化净收益 20.5% |
| 最大回撤 | **-35.8%** | 离 "穿越牛熊稳定" 还有距离 |

## 判断

- **简单 pairs trading（BTC-ETH 等）在主流币上不成立**：协整不显著，半衰期过长。
- **Cross-section 动量方向可行但样本不够**：仅 6 个币种做 L/S，单币风险太集中，回撤仍大。
- **下一步（exp02）**：扩大到 20+ 币种、引入多因子（动量+反转+funding）、按板块中性化，
  目标把 Net Sharpe 从 0.75 推到 1.5+，回撤压到 15% 以内。
