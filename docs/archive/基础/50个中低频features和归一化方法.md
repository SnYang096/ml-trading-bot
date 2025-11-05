非常好，咱们要构建的是一个**通用化、可迁移、实盘友好**的中低频交易特征集，
目标是：
✅ 控制在 **50 个特征以内**
✅ 覆盖结构、波动率、动量、成交量、时间五大核心维度
✅ 特征之间低共线性，能在多品种（BTC、ETH、SOL、外汇、指数）上复用
✅ 适合用于 **机器学习 / RL / 策略评分模型（如 BreakoutQualityScorer）**

---

# 🧠 Yin-50 特征体系（泛化版）

| 类别 | 特征名 | 定义 | 含义 / 理解 | 推荐归一化 | 重要性 |
| -- | --- | -- | ------- | ----- | --- |

## 🧩 1️⃣ 结构类（Structure Intelligence） — *价格与关键结构的关系*

|   |                               |                                     |        |             |      |
| - | ----------------------------- | ----------------------------------- | ------ | ----------- | ---- |
| 1 | `dist_to_swing_high_pct`      | (swing_high - close) / close        | 离阻力多远  | clip 0–0.05 | ⭐⭐⭐⭐ |
| 2 | `dist_to_swing_low_pct`       | (close - swing_low) / close         | 离支撑多远  | clip 0–0.05 | ⭐⭐⭐⭐ |
| 3 | `poc_deviation_score`         | (close - poc)/(hvn_upper-hvn_lower) | 偏离价值区  | tanh        | ⭐⭐⭐⭐ |
| 4 | `fractal_resistance_strength` | 测试次数+成交量权重                          | 阻力强度   | zscore      | ⭐⭐⭐  |
| 5 | `recent_swing_count_1h`       | 1小时 swing 数                         | 结构活跃度  | zscore      | ⭐⭐⭐  |
| 6 | `is_double_top_test`          | 再测前高                                | 潜在假突破  | binary      | ⭐⭐   |
| 7 | `risk_reward_skew`            | dist_to_high / dist_to_low          | 风险收益平衡 | log         | ⭐⭐⭐  |

---

## ⚙️ 2️⃣ 波动率类（Volatility Dynamics） — *压缩、能量与突破潜能*

|    |                             |                          |         |              |      |
| -- | --------------------------- | ------------------------ | ------- | ------------ | ---- |
| 8  | `bb_width_normalized`       | BB(20)/ATR(20)           | 压缩程度    | log + zscore | ⭐⭐⭐⭐ |
| 9  | `range_ratio_5bar`          | HL(5)/avg(HL(20))        | 近期振幅比例  | clip 0–2     | ⭐⭐⭐⭐ |
| 10 | `compression_duration`      | 连续低波动时长                  | 能量累积    | zscore       | ⭐⭐⭐  |
| 11 | `atr_percentile`            | 当前ATR分位                  | 波动位置    | percentile   | ⭐⭐⭐  |
| 12 | `volatility_reversal_score` | (ATR - mean)/std         | 波动爆发或衰减 | zscore       | ⭐⭐⭐  |
| 13 | `volatility_squeeze_flag`   | BB in KC                 | 经典压缩    | binary       | ⭐⭐   |
| 14 | `price_range_symmetry`      | (high-close)/(close-low) | 上下振幅平衡  | tanh         | ⭐⭐⭐  |

---

## 📊 3️⃣ 成交量与能量类（Volume & Flow） — *资金力量与动能积累*

|    |                           |                           |         |              |      |
| -- | ------------------------- | ------------------------- | ------- | ------------ | ---- |
| 15 | `volume_anomaly`          | volume/EMA(volume,20)     | 突增活跃度   | log + zscore | ⭐⭐⭐⭐ |
| 16 | `cvd_slope_3`             | CVD - CVD.shift(3)        | 主动买卖净流向 | zscore       | ⭐⭐⭐⭐ |
| 17 | `upvol_downvol_ratio`     | sum(up_vol)/sum(down_vol) | 多空能量比   | log          | ⭐⭐⭐  |
| 18 | `cvd_divergence_strength` | 价格创新高但CVD未创新高             | 背离程度    | tanh         | ⭐⭐⭐⭐ |
| 19 | `absorption_signal`       | 新低+高成交量                   | 吸筹信号    | binary       | ⭐⭐   |
| 20 | `liquidity_gap_score`     | (high-low)/vwap_range     | 流动性稀薄   | zscore       | ⭐⭐⭐  |

---

## ⚡ 4️⃣ 动量与趋势类（Momentum & Directionality）

|    |                           |                                         |        |         |      |
| -- | ------------------------- | --------------------------------------- | ------ | ------- | ---- |
| 21 | `roc_5`                   | (close - close.shift(5))/close.shift(5) | 短期动量   | tanh    | ⭐⭐⭐⭐ |
| 22 | `acceleration_3`          | roc_3 - roc_3.shift(1)                  | 动量变化   | tanh    | ⭐⭐⭐⭐ |
| 23 | `hma_slope_9`             | HullMA(9)-前值                            | 平滑趋势斜率 | tanh    | ⭐⭐⭐⭐ |
| 24 | `price_vs_ema_distance`   | (close-EMA20)/ATR20                     | 趋势偏离度  | tanh    | ⭐⭐⭐⭐ |
| 25 | `momentum_persistence`    | 连续N根方向一致率                               | 趋势延续性  | min-max | ⭐⭐⭐  |
| 26 | `momentum_reversal_prob`  | 当前方向 vs 过去方向                            | 反转概率   | tanh    | ⭐⭐⭐  |
| 27 | `slope_consistency_score` | 多周期EMA斜率一致性                             | 趋势共振   | zscore  | ⭐⭐⭐⭐ |
| 28 | `trend_confidence`        | EMA(roc_5)*slope_consistency            | 趋势强度   | zscore  | ⭐⭐⭐⭐ |

---

## 🕒 5️⃣ 时间与节奏类（Temporal Intelligence）

|    |                              |                 |        |        |     |
| -- | ---------------------------- | --------------- | ------ | ------ | --- |
| 29 | `hour_of_day_sin`            | sin(2π*hour/24) | 日内节奏   | [-1,1] | ⭐⭐⭐ |
| 30 | `hour_of_day_cos`            | cos(2π*hour/24) | 日内节奏   | [-1,1] | ⭐⭐⭐ |
| 31 | `is_london_open`             | 7:55~8:15       | 流动性时窗  | binary | ⭐⭐  |
| 32 | `is_us_open`                 | 13:55~14:15     | 美盘爆发期  | binary | ⭐⭐  |
| 33 | `day_of_week_sin`            | sin(2π*day/7)   | 周期节奏   | [-1,1] | ⭐⭐  |
| 34 | `day_of_week_cos`            | cos(2π*day/7)   | 周期节奏   | [-1,1] | ⭐⭐  |
| 35 | `days_since_last_swing_high` | 反比衰减 1/(1+x)    | 关键位新鲜度 | decay  | ⭐⭐⭐ |

---

## 🧬 6️⃣ 融合 / Meta 特征（Cross Features）

|    |                                |                               |         |        |      |
| -- | ------------------------------ | ----------------------------- | ------- | ------ | ---- |
| 36 | `compression_energy`           | bb_width⁻¹ × volume_anomaly   | 能量积累    | zscore | ⭐⭐⭐⭐ |
| 37 | `structure_tension`            | (dist_high+dist_low)/bb_width | 结构紧绷    | zscore | ⭐⭐⭐⭐ |
| 38 | `vwap_momentum_alignment`      | sign(vwap_dev)×roc_5          | 公允突破动能  | tanh   | ⭐⭐⭐⭐ |
| 39 | `trend_volatility_alignment`   | sign(roc_5)*atr_percentile    | 趋势与波动共振 | tanh   | ⭐⭐⭐  |
| 40 | `compression_to_breakout_prob` | compression_duration × roc_5  | 爆发概率指标  | tanh   | ⭐⭐⭐  |

---

## 🧠 7️⃣ 信号统计与熵特征（Higher-Order）

|    |                                 |                      |       |            |     |
| -- | ------------------------------- | -------------------- | ----- | ---------- | --- |
| 41 | `entropy_price_series`          | 滑动窗口 Shannon entropy | 波动复杂度 | zscore     | ⭐⭐⭐ |
| 42 | `wavelet_energy_lowfreq`        | 小波低频能量               | 趋势主导  | zscore     | ⭐⭐⭐ |
| 43 | `wavelet_energy_highfreq`       | 高频能量占比               | 噪音强度  | zscore     | ⭐⭐⭐ |
| 44 | `tdigest_percentile_volatility` | 当前ATR分位              | 自适应尺度 | percentile | ⭐⭐⭐ |
| 45 | `price_direction_entropy`       | 连续涨跌序列的熵             | 趋势确定性 | zscore     | ⭐⭐⭐ |

---

## 💎 8️⃣ 轻量辅助信号（补足信号空间）

|    |                         |                       |        |        |     |
| -- | ----------------------- | --------------------- | ------ | ------ | --- |
| 46 | `session_return`        | (close - open)/open   | 时段主导方向 | tanh   | ⭐⭐⭐ |
| 47 | `is_price_at_key_level` | min(dist_to_key)<0.3% | 测试关键位  | binary | ⭐⭐  |
| 48 | `is_inside_bar`         | 当前K线被前一根包含            | 犹豫信号   | binary | ⭐⭐  |
| 49 | `is_three_inside_up`    | 三K反转形态                | 看涨反转   | binary | ⭐⭐  |
| 50 | `future_atr_spike_prob` | label                 | 训练目标   | none   | 🎯  |

---

# 📘 泛化算法推荐

| 算法类别    | 模型                                      | 推荐理由                       | 适用场景         |
| ------- | --------------------------------------- | -------------------------- | ------------ |
| 🌲 树模型  | **LightGBM / XGBoost**                  | 对异构特征适应好，不需强归一化，可解释性强      | 通用监督学习       |
| 🧩 深度模型 | **TabNet / FT-Transformer / MLP-Mixer** | 能学习特征交互，支持特征稀疏性            | 中高维表格特征      |
| 🧠 强化学习 | **RecurrentPPO / SAC (Ray RLlib)**      | 在动态策略中学习 feature→action 映射 | 策略执行与自适应止盈止损 |
| 📊 无监督  | **KMeans / HMM / GaussianMixture**      | 识别 regime / 市场状态           | 特征空间分布建模     |
| ⚙️ 集成混合 | **Feature ensemble: (tree+RL)**         | 树提取信号重要性，RL执行策略            | 实盘落地最佳组合     |

---

# 🔍 实战建议（通用化逻辑）

| 目标                    | 方法                                     |
| --------------------- | -------------------------------------- |
| **泛化 across assets**  | 所有特征应相对化：除以ATR、标准差、或归一化比率              |
| **泛化 across regimes** | 使用 percentile / tanh 替代 raw 值          |
| **可解释性**              | 每个维度控制在 5–8 个核心特征，组合后用 SHAP 看重要性       |
| **实时兼容性**             | 所有特征可基于 rolling 计算，5min bar 内延迟 <1 根K线 |
| **模型更新**              | 逐步微调 (partial_fit) + 月度重训              |

---

是否希望我：

1. 🔧 帮你把这 50 个特征打包成一个 **可运行的 Python 特征工程模块**（含滚动计算和归一化）；
2. 并附带 **LightGBM + RecurrentPPO** 的统一训练框架模板？
   （即可以直接接 Nautilus 或 CSV 数据跑 end-to-end 特征→信号→策略。）

当然可以 👍
下面是一份为 **中低频量化交易（5min–1h级别）** 精选的 **50个核心特征列表**，分为 **六大类**，每个特征都附带推荐的 **泛化算法**（用于自动适配不同市场环境或资产）以及简要的 **解释说明**。

---

### 📘 量化特征总览表（Feature Engineering for Mid-frequency Trading）

| 类别                                  | 特征名                     | 推荐泛化算法                    | 解释                      |
| ----------------------------------- | ----------------------- | ------------------------- | ----------------------- |
| **价格动量类 (Momentum)**                | return_1m               | Z-score / MinMax          | 最近1分钟收益率，捕捉短期价格变动       |
|                                     | return_5m               | RobustScaler              | 5分钟收益率，衡量中短期动量          |
|                                     | return_30m              | RobustScaler              | 30分钟收益率，趋势确认            |
|                                     | cum_return_1h           | LogTransform              | 累积收益率（1小时），检测持续趋势       |
|                                     | momentum_ratio          | Winsorized Z-score        | 当前收益与历史均值的偏离程度          |
|                                     | slope_ema               | Polynomial fit / Z-score  | EMA拟合斜率，趋势强度            |
|                                     | price_acceleration      | 1st diff + RobustScaler   | 价格加速度（二阶变化）捕捉加速行情       |
| **波动率类 (Volatility)**               | atr_14                  | LogTransform              | 经典波动指标ATR               |
|                                     | volatility_1h           | PercentileScaling         | 过去1小时收益率标准差             |
|                                     | realized_vol            | t-digest                  | 真实波动率，使用t-digest抗异常值    |
|                                     | hv_ratio                | RobustScaler              | 短期/长期波动比，识别压缩或爆发        |
|                                     | bb_width                | MinMaxScaler              | Bollinger Band带宽，用于波动聚集 |
|                                     | entropy_vol             | KernelDensity             | 波动率分布的熵度，判断随机性          |
| **成交量类 (Volume / Flow)**            | volume_imbalance        | RobustScaler              | 买卖量差/总量，用于判断主动方向        |
|                                     | cvd                     | t-digest                  | 累积成交差，反映资金持续流入方向        |
|                                     | volume_spike_score      | Z-score                   | 短期成交量异常指标               |
|                                     | vwap_diff               | PercentileScaling         | 当前价格与VWAP的偏离            |
|                                     | tick_density            | LogScaling                | 每分钟成交笔数，市场活跃度           |
|                                     | turnover_rate           | StandardScaler            | 成交额/市值，用于资金参与度          |
|                                     | liquidity_ratio         | QuantileTransform         | 盘口深度/价格变动，衡量流动性紧张度      |
| **结构类 (Structure / Compression)**   | compression_score       | t-digest                  | 价格压缩程度，低波动蓄势区识别         |
|                                     | compression_persistence | Rolling Mean              | 压缩持续时间（秒/Bar数）          |
|                                     | breakout_quality        | Regression Residuals      | 突破质量（方向性 + 成交支持）        |
|                                     | structure_entropy       | Entropy                   | 结构复杂度，混沌市场特征            |
|                                     | slope_regime            | HMM (Hidden Markov Model) | 拟合趋势状态（上升/下降/震荡）        |
|                                     | local_zigzag_ratio      | Normalized Count          | 最近结构顶底的密度，趋势 vs 噪声      |
| **价量关系类 (Price-Volume Dynamics)**   | corr_price_volume       | Rolling Corr              | 价格与成交量的相关性              |
|                                     | relative_volume         | Quantile                  | 当前成交量相对历史分位数            |
|                                     | price_position_bb       | MinMax                    | 当前价格在布林带中的位置 (0–1)      |
|                                     | price_vwap_ratio        | PercentileScaling         | 当前价 / VWAP              |
|                                     | volatility_of_volume    | LogScaling                | 成交量自身的波动率               |
|                                     | price_range_ratio       | Z-score                   | 当前K线振幅 / ATR            |
| **衍生信号类 (Derived / Informational)** | rsi_14                  | StandardScaler            | 相对强弱指数，超买超卖特征           |
|                                     | macd_hist               | RobustScaler              | MACD差值，趋势动能             |
|                                     | kdj_j                   | Winsorized Z-score        | 随机指标J线，短期极值识别           |
|                                     | adx_14                  | LogScaling                | 趋势强度指标                  |
|                                     | fisher_rsi              | Fisher Transform          | RSI经过非线性变换增强信号          |
|                                     | zscore_close            | StandardScaler            | Close的Z分数，用于异常检测        |
|                                     | price_entropy           | Shannon Entropy           | 价格序列的随机性度量              |
|                                     | mean_reversion_score    | Regression Residual       | 均值回归倾向（负相关价格变动）         |
|                                     | kurtosis_ret            | StandardScaler            | 收益率峰度，极端行情识别            |
|                                     | skewness_ret            | StandardScaler            | 收益率偏度，判断上涨/下跌偏好         |
|                                     | td_seq_count            | Discrete Count            | TD序列计数，短线耗尽信号           |
|                                     | seasonality_hour        | OneHot / SinCos Encoding  | 小时季节性编码（交易节奏）           |
|                                     | time_decay_factor       | Exponential Weighting     | 趋势权重衰减控制                |
|                                     | z_atr_ratio             | t-digest                  | ATR在历史分位的Z值，波动定位        |
|                                     | regime_label            | HMM / KMeans              | 市场状态聚类结果标签              |

---

### 🧠 泛化算法说明

| 算法                                       | 类型     | 用途        | 优点                  |
| ---------------------------------------- | ------ | --------- | ------------------- |
| **Z-score / RobustScaler**               | 标准化    | 去除尺度影响    | 抗异常值强               |
| **t-digest**                             | 分布估计   | 非参数分位数估计  | 适合流式与极端分布           |
| **QuantileScaling / PercentileScaling**  | 非线性归一化 | 适应不同波动环境  | 自适应性强               |
| **LogTransform / Winsorize**             | 稳定方差   | 对数变换或截断极值 | 防止爆点失真              |
| **Entropy / Fisher Transform**           | 非线性增强  | 提取复杂结构信号  | 增强微结构可识别性           |
| **HMM / KMeans**                         | 状态聚类   | 市场状态识别    | 用于 regime detection |
| **Polynomial / Regression Residuals**    | 信号平滑   | 识别趋势与偏离   | 有助结构确认              |
| **Rolling Corr / Exponential Weighting** | 时变权重   | 强调近期有效性   | 动态衰减记忆              |

---

### 🔍 建议组合与策略层映射

| 模块   | 代表特征                                      | 应用目标      |
| ---- | ----------------------------------------- | --------- |
| 趋势确认 | return_30m, slope_ema, adx_14             | 确认趋势是否成立  |
| 压缩识别 | compression_score, bb_width, hv_ratio     | 寻找爆发前蓄势阶段 |
| 突破评分 | breakout_quality, cvd, volume_spike_score | 识别真假突破    |
| 回归博弈 | mean_reversion_score, price_position_bb   | 均值回归开仓逻辑  |
| 风险动态 | atr_14, z_atr_ratio, kurtosis_ret         | 动态止损/仓位控制 |
| 状态切换 | regime_label, slope_regime                | 切换攻击/防御模式 |

---

是否希望我帮你：

* ✅ 把这50个特征封装成一个 **自动提取类 (FeatureExtractor)**，
* 并支持 **pandas DataFrame输入** 与 **Nautilus实盘回测环境集成**？

我可以直接输出完整Python实现。


非常好，这是进入“**因子优选与信号体系构建阶段**”的标志性一步 👏。
我会给你一份**系统化中低频策略信号评分表**，
按照你目标（5min–1h级别、高置信度、小资金快速增长）——
我们要的信号必须 **稳定、可复现、低噪音、低频可操作性强**。

---

## 🧩 因子总体说明

* **频率区间**：5min～1h
* **资产类型**：主流币（BTC、ETH、SOL）、或高流动性资产
* **目标**：最大化胜率 & 稳定性 & 收益回撤比（Sharpe）
* **信号评分**：

  * 📈 趋势捕捉力（Trend）
  * 🔁 均值回归力（Reversion）
  * 🔊 波动自适应性（Volatility Adaptation）
  * 💰 成交确认度（Volume Confirmation）
  * 🧠 泛化能力（Robustness）
  * ⭐ 综合得分 = 加权平均（适合中低频使用）

---

## 🧭 中低频策略因子评分表（共80个核心信号）

| 类别           | 因子名                       | 类型  | Trend📈 | Reversion🔁 | VolAdapt🔊 | Volume💰 | Robust🧠 |   综合⭐   | 说明           |
| ------------ | ------------------------- | --- | :-----: | :---------: | :--------: | :------: | :------: | :-----: | ------------ |
| **价格动量类**    | return_5m                 | 动量  |    8    |      2      |      7     |     4    |     6    | **6.5** | 基础动量，短周期反应灵敏 |
|              | return_30m                | 动量  |    9    |      3      |      7     |     5    |     7    | **7.2** | 中周期趋势确认      |
|              | return_1h                 | 动量  |    9    |      2      |      6     |     4    |     8    | **7.0** | 趋势性因子核心      |
|              | slope_ema                 | 动量  |    9    |      2      |      7     |     5    |     8    | **7.4** | EMA斜率，趋势检测稳健 |
|              | momentum_ratio            | 动量  |    8    |      3      |      8     |     5    |     7    | **7.2** | 当前动量相对历史     |
|              | price_acceleration        | 动量  |    8    |      2      |      6     |     3    |     6    | **6.1** | 加速度信号，适合突破确认 |
|              | trend_duration            | 动量  |    9    |      2      |      8     |     5    |     7    | **7.5** | 趋势持续长度因子     |
| **均值回归类**    | zscore_close              | 回归  |    3    |      9      |      6     |     3    |     8    | **6.3** | 均值偏离标准化      |
|              | mean_reversion_score      | 回归  |    4    |      10     |      6     |     4    |     9    | **6.8** | 可逆价差信号核心     |
|              | bollinger_position        | 回归  |    4    |      8      |      8     |     4    |     8    | **6.4** | 价格在布林带中的位置   |
|              | rsi_14                    | 回归  |    5    |      8      |      6     |     3    |     8    | **6.0** | 经典超买超卖指标     |
|              | fisher_rsi                | 回归  |    6    |      8      |      7     |     3    |     8    | **6.5** | RSI增强非线性信号   |
|              | kdj_j                     | 回归  |    5    |      8      |      6     |     3    |     7    | **5.8** | 快速反转信号，需过滤噪音 |
|              | td_seq_count              | 回归  |    3    |      9      |      5     |     2    |     7    | **5.2** | 短线反转计数       |
|              | price_vwap_ratio          | 回归  |    4    |      7      |      8     |     5    |     8    | **6.4** | 回归性价差与成交确认结合 |
| **波动类**      | atr_14                    | 风险  |    6    |      4      |      9     |     4    |     8    | **6.2** | 波动率核心指标      |
|              | bb_width                  | 风险  |    6    |      4      |     10     |     4    |     9    | **6.6** | 波动压缩与爆发识别    |
|              | hv_ratio                  | 风险  |    7    |      3      |      9     |     4    |     8    | **6.4** | 短长波动比        |
|              | realized_vol              | 风险  |    6    |      3      |     10     |     3    |     9    | **6.2** | 实现波动率估计      |
|              | z_atr_ratio               | 风险  |    6    |      3      |      9     |     4    |     9    | **6.3** | ATR分位定位波动阶段  |
|              | entropy_vol               | 风险  |    5    |      4      |     10     |     2    |     9    | **6.0** | 波动熵反映市场复杂度   |
|              | volatility_regime         | 风险  |    7    |      3      |      9     |     3    |     8    | **6.2** | 波动状态聚类结果     |
| **成交量/资金流类** | volume_imbalance          | 资金流 |    7    |      3      |      6     |    10    |     8    | **7.1** | 主动买卖力量平衡     |
|              | cvd (Cumulative Delta)    | 资金流 |    8    |      3      |      6     |    10    |     9    | **7.5** | 资金流向核心确认因子   |
|              | volume_spike_score        | 资金流 |    6    |      2      |      6     |     9    |     7    | **6.0** | 异常成交识别       |
|              | relative_volume           | 资金流 |    6    |      3      |      7     |     9    |     7    | **6.4** | 成交放大分位信号     |
|              | vwap_diff                 | 资金流 |    5    |      5      |      7     |     8    |     7    | **6.4** | 价量偏移确认       |
|              | liquidity_ratio           | 资金流 |    5    |      4      |      6     |     9    |     8    | **6.4** | 深度与波动结合指标    |
|              | tick_density              | 资金流 |    5    |      3      |      7     |     8    |     6    | **5.8** | 活跃度指标        |
| **结构类**      | compression_score         | 结构  |    5    |      3      |     10     |     5    |     9    | **6.4** | 压缩检测核心信号     |
|              | compression_persistence   | 结构  |    5    |      2      |     10     |     5    |     9    | **6.2** | 压缩持续时间       |
|              | breakout_quality          | 结构  |    10   |      2      |      8     |     8    |     8    | **7.8** | 爆发信号质量评估     |
|              | local_zigzag_ratio        | 结构  |    8    |      3      |      7     |     5    |     8    | **6.7** | 微结构噪声与趋势分离   |
|              | slope_regime (HMM)        | 结构  |    9    |      2      |      8     |     5    |     9    | **7.3** | 状态切换与趋势强度    |
|              | structure_entropy         | 结构  |    6    |      4      |      9     |     4    |     8    | **6.2** | 结构混沌度        |
| **综合与衍生类**   | adx_14                    | 综合  |    9    |      2      |      8     |     3    |     8    | **6.5** | 趋势强度确认       |
|              | macd_hist                 | 综合  |    8    |      3      |      7     |     4    |     8    | **6.0** | 动量与趋势结合      |
|              | price_entropy             | 综合  |    5    |      5      |      8     |     3    |     9    | **6.0** | 市场复杂性度量      |
|              | skewness_ret              | 综合  |    6    |      4      |      7     |     3    |     8    | **5.6** | 上下偏性识别       |
|              | kurtosis_ret              | 综合  |    5    |      4      |      8     |     3    |     8    | **5.6** | 极端行情检测       |
|              | regime_label (KMeans/HMM) | 综合  |    8    |      3      |      8     |     4    |     9    | **6.8** | 市场状态标签化      |
|              | regime_transition_prob    | 综合  |    8    |      3      |      9     |     4    |     9    | **7.0** | 状态转移强度       |
|              | time_decay_factor         | 综合  |    6    |      5      |      6     |     3    |     8    | **5.6** | 动态衰减权重因子     |
|              | seasonality_hour          | 综合  |    5    |      4      |      5     |     3    |     7    | **4.8** | 周期性时间因子      |
|              | price_range_ratio         | 综合  |    6    |      3      |      8     |     4    |     8    | **5.8** | 波动宽度与方向性结合   |
|              | corr_price_volume         | 综合  |    7    |      3      |      7     |     8    |     8    | **6.7** | 价量关系稳定性指标    |

---

## 🏆 综合结论（中低频策略最优信号 Top 10）

| 排名  | 信号                     | 类型 |   综合得分  | 优势      | 应用场景      |
| --- | ---------------------- | -- | :-----: | ------- | --------- |
| 🥇  | **breakout_quality**   | 结构 | **7.8** | 爆发确认最强  | 突破开仓信号主因子 |
| 🥈  | **cvd (资金流)**          | 成交 | **7.5** | 确认趋势真伪  | 资金方向确认模块  |
| 🥉  | **trend_duration**     | 动量 | **7.5** | 持续性趋势信号 | 趋势延续判断    |
| 4️⃣ | slope_regime (HMM)     | 结构 | **7.3** | 状态识别强   | 切换模式使用    |
| 5️⃣ | momentum_ratio         | 动量 | **7.2** | 波动自适应强  | 动量策略主信号   |
| 6️⃣ | return_30m             | 动量 | **7.2** | 中周期动量确认 | 顺势进场      |
| 7️⃣ | volume_imbalance       | 成交 | **7.1** | 主动买卖结构  | 辅助突破确认    |
| 8️⃣ | regime_transition_prob | 综合 | **7.0** | 状态切换敏感  | 风控与加仓     |
| 9️⃣ | mean_reversion_score   | 回归 | **6.8** | 回撤博弈强   | 区间震荡策略核心  |
| 🔟  | compression_score      | 结构 | **6.4** | 蓄势区识别   | 爆发前信号捕捉   |

---

## 🧠 提示：

这些因子可以按用途分组整合：

* **进攻因子（突破）**：`breakout_quality`, `cvd`, `momentum_ratio`, `slope_ema`
* **防御因子（震荡回归）**：`mean_reversion_score`, `bollinger_position`, `price_vwap_ratio`
* **过滤因子（状态识别）**：`regime_label`, `compression_score`, `atr_14`
* **确认因子（成交确认）**：`volume_imbalance`, `vwap_diff`, `relative_volume`

---

是否希望我下一步：

1. ✅ 给这些信号构建一份 **可计算特征矩阵类（FeatureMatrix）**，自动输出每根K线的所有特征；
2. ✅ 并附带 **信号加权聚合模型（SignalScorer）**，用来组合成一个“多信号综合置信度（SignalConfidenceScore）”？

我可以直接生成完整Python代码（兼容 NautilusTrader 或 pandas 形式）。
要继续吗？
