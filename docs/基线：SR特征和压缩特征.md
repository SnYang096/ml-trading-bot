# 📘 LightGBM 特征工程文档（Feature Engineering Specification）

---

## 🧩 位置特征（Positional Features）

| 特征名称 | 含义 / 定义 | 计算方法 | 数据类型 | 归一化方法 | 应用说明 |
|-----------|--------------|-----------|-----------|-------------|-----------|
| `zigzag_turn` | ZigZag 转折点（高低点识别） | 基于价格波动阈值过滤后的转折点；`Δprice / ATR` | float | Z-score / MinMax(-1,1) | 趋势反转、支撑阻力识别 |
| `zigzag_slope` | ZigZag 连线斜率 | `(price[n] - price[n-k]) / k` | float | 标准化（/ATR） | 趋势强度 |
| `swing_high` / `swing_low` | 局部极值 | 比较当前K线与左右N根高/低点 | float | /close 或 zscore | 结构节点识别 |
| `swing_confluence` | 多时间周期 Swing 重合 | count(高低点落在同区间) | int | /max_count | 关键位置确认 |
| `poc_value` | 成交量最大价位（POC） | 从 Volume Profile 中取成交量最高的价格 | float | /close | 价值中枢、均值回归信号 |
| `poc_distance` | 当前价格距POC距离 | `(close - poc_value)/ATR` | float | 标准化 | 价格偏离度 |
| `hal_gap_ratio` | HAL 区间比率 | `(high[i-1] - low[i]) / ATR` | float | MinMax(0,1) | 不平衡区识别 |
| `ols_slope` | OLS 回归线斜率 | 对窗口价格做线性回归：`β = cov(x,y)/var(x)` | float | 标准化（/ATR） | 趋势方向量化 |
| `ols_upper_band` / `ols_lower_band` | OLS 通道上下轨 | `y_pred ± std(price - y_pred)*σ` | float | /close | 超买超卖判断 |

---

## 🧱 形态特征（Pattern Features）

| 特征名称 | 含义 / 定义 | 计算方法 | 数据类型 | 应用说明 |
|-----------|--------------|-----------|-----------|-----------|
| `pattern_M_top` | M头结构 | SwingHigh 两次相近高点且跌破颈线 | bool | 反转信号 |
| `pattern_W_bottom` | W底结构 | SwingLow 两次相近低点且突破颈线 | bool | 反转信号 |
| `pattern_HH` / `pattern_LL` | Higher High / Lower Low 序列 | 连续 swing 高点上升或低点下降 | bool | 趋势延续确认 |
| `pattern_HL` / `pattern_LH` | Higher Low / Lower High | swing 低点上移 / 高点下移 | bool | 潜在趋势转折 |
| `pattern_triangle_flag` | 三角形整理 | 高点递减、低点递增 | bool | 盘整信号 |
| `pattern_head_shoulder` | 头肩顶/底 | 左肩、头、右肩Swing结构 + 颈线突破 | bool | 中期反转信号 |

---

## 🎚️ 压缩特征（Compression Features）

| 特征名称 | 含义 / 定义 | 计算方法 | 数据类型 | 归一化方法 | 应用说明 |
|-----------|--------------|-----------|-----------|-------------|-----------|
| `atr_percentile` | 当前ATR在历史分位 | `rank(ATR)/window` | float | [0,1] | 波动率压缩状态 |
| `atr_compression_ratio` | ATR压缩强度 | `mean(ATR_hist)/current_ATR` | float | 标准化 | 波动率维度 |
| `body_to_atr_ratio` | K线实体占ATR比例 | `abs(close-open)/ATR` | float | [0,1] | 实体小→压缩强 |
| `volume_percentile` | 成交量分位数 | `rank(volume)/window` | float | [0,1] | 成交量萎缩判断 |
| `volume_compression_ratio` | 成交量压缩 | `mean(volume_hist)/current_volume` | float | 标准化 | 流动性收缩信号 |
| `price_entropy` | 方向熵 | 统计涨跌序列的Shannon熵 | float | [0,1] | 有序性/混沌度 |
| `internal_price_density` | 内部价格密度 | `1 - var(price_within_zone)/var(price_total)` | float | [0,1] | 振荡强度 |
| `compression_duration` | 压缩持续时长 | 连续ATR<阈值的K线数量 | int | /max_duration | 压缩持续性 |
| `pre_break_silence` | 突破前静默度 | 最近N根ATR均<20%分位 | float | [0,1] | 蓄势评分 |
| `compression_confidence` | 多维融合压缩置信度 | `0.5*atr + 0.3*volume + 0.2*density` | float | [0,1] | 综合压缩评分 |

---

## 📊 成交量特征（Volume-Based Features）

| 特征名称 | 含义 / 定义 | 计算方法 | 数据类型 | 归一化方法 | 应用说明 |
|-----------|--------------|-----------|-----------|-------------|-----------|
| `volume_ma_ratio` | 成交量相对均值 | `volume / SMA(volume, N)` | float | Z-score | 活跃度检测 |
| `volume_trend_slope` | 成交量趋势斜率 | OLS回归斜率(Volume) | float | 标准化 | 成交量趋势强度 |
| `volume_change_rate` | 成交量变化率 | `(volume - volume[-1]) / volume[-1]` | float | zscore | 成交量加速/减弱 |
| `volume_delta_price_corr` | 成交量与价格变化相关系数 | corr(Δprice, volume) | float | [-1,1] | 主动买盘/卖盘识别 |
| `volume_spike_score` | 成交量激增得分 | `volume / mean(volume_hist)` | float | log1p归一化 | 异常活跃检测 |
| `volume_profile_skew` | 成交量分布偏度 | 价格分层成交量分布的偏度 | float | zscore | 主力集中区倾向 |
| `buy_sell_ratio` | 主动买卖量比 | `buy_volume / (buy_volume + sell_volume)` | float | [0,1] | 方向偏向 |
| `delta_volume_imbalance` | 成交量不平衡度 | `(buy - sell) / (buy + sell)` | float | [-1,1] | 主动性指标 |
| `vwap_distance` | 当前价距VWAP | `(close - VWAP) / ATR` | float | 标准化 | 均值偏离 |
| `vwap_trend_slope` | VWAP趋势 | 对VWAP序列进行线性回归斜率 | float | 标准化 | 成交量加权趋势方向 |
| `tick_volume_entropy` | 成交笔数熵 | 对tick级成交量分布计算熵 | float | [0,1] | 活跃度结构分析 |

---

## 🧭 动态结构特征（Dynamic Structural Features）

| 特征名称 | 含义 / 定义 | 计算方法 | 数据类型 | 归一化方法 | 应用说明 |
|-----------|--------------|-----------|-----------|-------------|-----------|
| `dynamic_window_length` | 自适应窗口长度 | `base_window * f(ATR_percentile)` | int | /max_window | 自适应时窗调整 |
| `trend_slope_change` | 回归斜率变化率 | `Δols_slope / previous_slope` | float | 标准化 | 趋势加速/衰减信号 |
| `entropy_change_rate` | 方向熵变化率 | `Δentropy / entropy_prev` | float | 标准化 | 市场从混乱→有序转变 |
| `compression_state` | 当前状态 | compression / expansion / breakout | categorical | One-Hot | 状态机输入 |

---

## ⚙️ 特征归一化与编码方案

| 类型 | 推荐方法 | 说明 |
|------|-----------|------|
| 连续型（float） | Z-score 或 MinMax(-1,1) | 保持分布对称 |
| 分位型（percentile） | 保留 [0,1] | 反映相对强度 |
| 比率型 | 对数缩放 `log1p(x)` | 避免长尾 |
| 离散型 | One-Hot 编码 | LightGBM原生支持 |
| 时间特征 | 标准化为周期相对位 | 用于滚动窗口 |

---

## 🎯 模型目标与特征映射

| 模型目标 | 关键特征类别 | 示例 |
|-----------|---------------|------|
| 趋势预测 | ZigZag、OLS斜率、HH/LL序列 | trend_up_prob |
| 突破预测 | 压缩置信度、内部密度、pre_break_silence | breakout_prob |
| 反转预测 | M/W形态、POC偏离度、volume_divergence | reversal_prob |
| 波动率预测 | ATR分位、entropy、OLS通道宽度 | future_volatility |
| 动态仓位/止盈管理 | compression_confidence + future_return | position_ratio, tp/sl_threshold |

---

## 🧮 LightGBM Pipeline 建议流程

1. 使用历史Bar数据生成全部特征（价格结构、成交量、压缩状态等）  
2. 对连续特征归一化；布尔特征One-Hot  
3. 对齐目标标签（未来收益率或突破成功率）  
4. 时间滚动切分（train/test），避免数据泄露  
5. 输出特征重要性与 shap 解释  
6. 根据特征贡献度进行因子筛选与精简  

---

© 2025 — Advanced Quantitative Feature Engineering Framework for LightGBM
