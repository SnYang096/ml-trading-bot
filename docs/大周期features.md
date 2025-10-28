非常好的问题，这个是进入“**多时间尺度信号融合（Multi-Timescale Alpha Architecture）**”的关键点。
你目前的因子体系主要针对 **5 min – 1 h** 的中低频信号（微结构 + 短波趋势）。
一旦把时间尺度拉大到 **4 h、1 d、3 d、1 w**，
新的“慢变量（slow factors）”就会出现，它们能提供更强的趋势确认、资金流向和宏观结构信息。

---

## 🧭 Ⅰ. 时间尺度扩展原则

| 维度   | 小周期（5 min–1 h） | 大周期（4 h–1 w） | 说明                 |
| ---- | -------------- | ------------ | ------------------ |
| 主信号  | 动量、结构、压缩       | 趋势、位置、资金流    | 大周期主导方向            |
| 噪声特征 | 高频交易/盘口波动      | 平滑，低噪声       | 可容忍长滞后             |
| 因子作用 | 信号触发           | 过滤与确认        | 低频指导高频             |
| 特征形态 | 导数/加速度型        | 均值/累积型       | “快信号驱动 + 慢信号约束” 结构 |

---

## 🧩 Ⅱ. 大周期因子扩展目录（30 + 个）

| 类别              | 因子名                         | 定义/计算                       | 解释        | 特性     |
| --------------- | --------------------------- | --------------------------- | --------- | ------ |
| **趋势类**         | `supertrend_signal`         | 方向=close>supertrend         | 平滑趋势确认    | 稳定     |
|                 | `ema_ribbon_trend`          | EMA(20/50/100/200) 布局状态     | 长期趋势分层    | 方向滤波   |
|                 | `trend_regime_score`        | zscore(close/EMA200)        | 位置性趋势强度   | 长期结构判断 |
|                 | `adx_50`                    | 长周期ADX                      | 大趋势存在性    | 慢速确认   |
|                 | `trend_persistence`         | 连续趋势bar数                    | 趋势惯性度     | 趋势寿命   |
| **位置类（价值区/极端）** | `percentile_price_90d`      | 价格在90天分位                    | 相对高低估     | 定价区    |
|                 | `distance_to_year_high`     | (close-year_high)/year_high | 离历史阻力远近   | 中线压强   |
|                 | `distance_to_year_low`      | (close-year_low)/year_low   | 离支撑远近     | 下行空间   |
|                 | `price_vs_vwap_20d`         | (close-VWAP20d)/VWAP20d     | 中期资金平均价偏离 | 均值吸引力  |
| **资金流/成交量结构**   | `OBV_trend`                 | OBV slope                   | 主力流向      | 稳定确认   |
|                 | `volume_profile_balance`    | 高成交区 vs 低成交区                | 是否处于价值密集区 | 结构确认   |
|                 | `accumulation_days`         | 连续成交量放大 + 上涨                | 吸筹判定      | 中期布局信号 |
|                 | `distribution_days`         | 成交量放大 + 下跌                  | 出货信号      | 顶部特征   |
|                 | `smart_money_flow`          | 资金流领先价变动                    | 大户动向      | 提前预警   |
| **波动类**         | `volatility_regime_90d`     | ATR(20)/ATR(90)             | 波动周期阶段    | 风险状态   |
|                 | `vol_cluster_persistence`   | 波动高/低聚类持续度                  | 稳态/爆发切换   | 风险前兆   |
|                 | `hv_percentile`             | 历史波动率分位                     | 识别极端低波/高波 | 市场冷暖度  |
|                 | `realized_vol_30d_trend`    | 实际波动趋势                      | 长周期波动倾向   | 风险调节   |
| **价量结构与宏观技术形态** | `market_structure_high_low` | 连续高低点破位次数                   | 结构趋势稳定性   | 宏观态势   |
|                 | `multi_top_bottom_count`    | 多头/空头结构重复次数                 | 反复测试信号    | 顶底区强度  |
|                 | `broadening_pattern_flag`   | 扩散形态检测                      | 高波动反转信号   | 风险区    |
|                 | `weekly_candle_type`        | 周K阳线率                       | 市场偏向      | 市场重心   |
| **时间节奏与季节性**    | `day_of_week_sin/cos`       | 周期编码                        | 星期效应      | 模式性    |
|                 | `month_seasonality`         | 月份哑变量                       | 季节特征      | 长期资金节奏 |
|                 | `time_since_major_move`     | 距离上次>5%波动的bar数              | 冷却时间      | 波动复原周期 |
| **统计与资金强度**     | `rolling_sharpe_20d`        | mean(ret)/std(ret)          | 稳定收益期识别   | 信号质量   |
|                 | `rolling_max_drawdown`      | drawdown(20d)               | 局部风险度     | 止损调整   |
|                 | `price_autocorr_20d`        | 自相关系数                       | 趋势延续性     | 可预测性   |
|                 | `return_skewness_30d`       | 收益偏度                        | 极端倾向      | 风险溢价指标 |
| **跨资产/宏观类**     | `btc_dominance`             | BTC市占率变化                    | 风险偏好度     | 市场情绪   |
|                 | `eth_btc_ratio_trend`       | ETH/BTC趋势                   | 资金轮动方向    | 行业强弱   |
|                 | `macro_beta_index`          | 与宏观指数协方差                    | 风险联动      | Beta暴露 |
|                 | `funding_rate_rolling`      | 永续合约资金费率均值                  | 市场情绪偏热/偏冷 | 杠杆状态   |

---

## 🧮 Ⅲ. 大周期因子特征定位

| 因子类型 | 时间尺度  | 优势       | 在策略中的作用     |
| ---- | ----- | -------- | ----------- |
| 趋势型  | 4h–1w | 噪音低、方向稳定 | 主趋势确认、反转过滤  |
| 位置型  | 1d–1w | 对齐关键位    | 阻力/支撑区判定    |
| 资金流型 | 1h–1d | 验证突破真实性  | 过滤假信号       |
| 波动型  | 1d–1w | 调节仓位大小   | 动态风险管理      |
| 统计型  | 1w–1m | 描述状态稳定性  | 策略切换与风控     |
| 跨资产型 | 任意    | 宏观情绪指标   | 资产轮动或beta中和 |

---

## 🧠 Ⅳ. 多时间尺度信号融合（推荐框架）

```python
SignalComposite = (
    0.4 * short_term_confidence   # 5m–1h 动量/结构/压缩
  + 0.4 * mid_term_trend_strength # 4h–1d 趋势/资金流/波动
  + 0.2 * long_term_regime_score  # 1d–1w 状态/位置/宏观
)
```

* **短周期信号**：捕捉触发点
* **中周期信号**：确认方向与动量
* **长周期信号**：提供市场环境和风险过滤
  → 最终输出一个 **多时间尺度置信度 (Multi-Timescale Confidence Score)**，可用于进场过滤、加仓权重或风控参数调节。

---

## 📊 Ⅴ. 高层特征集总结（中低频 + 大周期）

| 分类     |  数量 | 示例                                     | 特征说明    |
| ------ | :-: | -------------------------------------- | ------- |
| 结构与动量  |  10 | breakout_quality, trend_duration       | 信号触发核心  |
| 均值与回归  |  6  | zscore_close, mean_reversion_score     | 区间震荡捕捉  |
| 波动与风险  |  8  | atr_14, volatility_regime_90d          | 波动状态识别  |
| 成交与资金流 |  8  | cvd, accumulation_days                 | 资金确认    |
| 趋势与位置  |  8  | ema_ribbon_trend, percentile_price_90d | 中期趋势判断  |
| 统计与状态  |  6  | rolling_sharpe_20d, price_autocorr_20d | 信号稳定性评估 |
| 跨资产与宏观 |  4  | btc_dominance, eth_btc_ratio_trend     | 市场大势信号  |
| 时间与季节性 |  4  | day_of_week_sin, month_seasonality     | 时间偏序规律  |

👉 **总计约 54 – 60 个高置信度可用因子**，
足以构建一套完整的中低频多时间尺度交易信号体系。

---

是否希望我帮你把这些 **中低频 + 大周期因子** 全部整理成一份
✅ `FeatureDefinitionTable`（表格 + 归一化建议 + 泛化算法类型 + 实现模板）？
我可以输出成 `Markdown + CSV + Python类模板` 三合一版本，
直接放进你的 Nautilus 实盘 pipeline 里使用。是否生成？
