非常棒 👍 你这套表格已经进入“量化研究院级”的结构体系了。
下面是我帮你优化过的 **增强版表格版本**（保持你原有风格 + 加入我前面那套“ATR 距离比例 + 结构重叠强度 + 成交量强化逻辑”）。
它分为 6 个表格区块，并在最后增加了模型特征组合建议与实现重点。
这样你保存为 `.md` 后可以直接在 VSCode / Notion / Obsidian 中渲染使用。

---

# 📘 LightGBM 特征工程文档（Feature Engineering Specification v3）

---

## 🧩 一、位置与结构特征（Positional & Structural Features）

| 特征名称                                           | 含义 / 定义                | 计算方法                                                   | 数据类型  | 归一化方法   | 应用说明      |
| ---------------------------------------------- | ---------------------- | ------------------------------------------------------ | ----- | ------- | --------- |
| `zigzag_turn`                                  | ZigZag 转折点标记           | 依据波动阈值标识局部高/低点                                         | bool  | -       | 反转信号核心    |
| `zigzag_distance_high` / `zigzag_distance_low` | 当前价距最近ZigZag高/低点的标准化距离 | `(close - zigzag_high)/ATR`、`(close - zigzag_low)/ATR` | float | Z-score | 趋势结构位置识别  |
| `zigzag_range_ratio`                           | ZigZag波段相对ATR宽度        | `(zigzag_high - zigzag_low)/ATR`                       | float | 标准化     | 波动强度识别    |
| `swing_confluence`                             | 多周期Swing重叠强度           | 重叠区间数 / 总周期数                                           | float | [0,1]   | 趋势确认/结构共振 |
| `hal_distance`                                 | 当前价距HAL中线比例            | `(close - hal_mid)/ATR`                                | float | [-1,1]  | 区间偏移识别    |
| `hal_slope` / `hal_curvature`                  | HAL斜率与曲率               | ΔHAL/ATR, Δ²HAL/ATR                                    | float | 标准化     | 趋势方向与加速度  |
| `ols_upper_distance` / `ols_lower_distance`    | 与OLS通道上下轨距离            | `(OLS_Upper−Close)/ATR` 等                              | float | 标准化     | 超买/超卖识别   |
| `ols_bandwidth`                                | OLS通道宽度                | `(OLS_Upper−OLS_Lower)/ATR`                            | float | 标准化     | 波动压缩度量    |
| `ols_slope`                                    | OLS线斜率                 | `β = cov(x,y)/var(x)`                                  | float | 标准化     | 趋势强度      |
| `poc_value` / `poc_distance`                   | POC价格与当前价距             | `(close−poc_value)/ATR`                                | float | 标准化     | 价值中枢偏移    |
| `structure_overlap_score`                      | 多结构重叠强度                | 各结构距离差 < 0.5ATR 的重叠比例                                  | float | [0,1]   | 关键结构共振强度  |

---

## 🧱 二、形态特征（Pattern Features）

| 特征名称                                 | 含义 / 定义  | 计算方法          | 数据类型 | 应用说明   |
| ------------------------------------ | -------- | ------------- | ---- | ------ |
| `pattern_M_top` / `pattern_W_bottom` | M顶/W底结构  | 双顶/双底+颈线突破    | bool | 反转信号   |
| `pattern_HH` / `pattern_LL`          | 连续高高/低低  | swing高/低点单调变化 | bool | 趋势延续   |
| `pattern_HL` / `pattern_LH`          | 高低点抬高/降低 | 低点上移/高点下移     | bool | 趋势转折预警 |
| `pattern_triangle_flag`              | 三角形整理    | 高点递减、低点递增     | bool | 盘整确认   |
| `pattern_head_shoulder`              | 头肩结构     | 左肩-头-右肩+颈线验证  | bool | 中期反转信号 |

---

## 🎚️ 三、压缩特征（Compression Features）

| 特征名称                     | 含义 / 定义  | 计算方法                               | 数据类型  | 归一化方法 | 应用说明    |
| ------------------------ | -------- | ---------------------------------- | ----- | ----- | ------- |
| `atr_percentile`         | 当前ATR分位数 | `rank(ATR)/window`                 | float | [0,1] | 波动压缩状态  |
| `atr_compression_ratio`  | 波动压缩比    | `mean(ATR_hist)/ATR`               | float | 标准化   | 波动收缩检测  |
| `volume_percentile`      | 成交量分位数   | `rank(volume)/window`              | float | [0,1] | 成交量萎缩识别 |
| `price_entropy`          | 方向熵      | 涨跌方向 Shannon Entropy               | float | [0,1] | 市场混沌度   |
| `internal_price_density` | 内部价格密度   | `1 - var(price_zone)/var(total)`   | float | [0,1] | 价格集中度   |
| `compression_duration`   | 连续压缩时长   | 连续ATR<阈值的bar数                      | int   | /max  | 蓄势期检测   |
| `pre_break_silence`      | 突破前静默度   | 近N根ATR均<20%分位                      | float | [0,1] | 预爆发识别   |
| `compression_confidence` | 综合压缩置信度  | 0.5*ATR + 0.3*Volume + 0.2*Density | float | [0,1] | 总体压缩评分  |

---

## 📊 四、成交量强化特征（Volume Reinforcement Features）

| 特征名称                 | 含义 / 定义    | 计算方法                                      | 数据类型  | 归一化方法   | 应用说明   |
| -------------------- | ---------- | ----------------------------------------- | ----- | ------- | ------ |
| `volume_zscore`      | 成交量标准化强度   | `(vol−mean(vol_N))/std(vol_N)`            | float | Z-score | 活跃度识别  |
| `volume_trend`       | 成交量趋势斜率    | EMA(vol,10)−EMA(vol,20)                   | float | 标准化     | 方向确认   |
| `volume_volatility`  | 成交量波动率     | `std(vol_N)/mean(vol_N)`                  | float | 标准化     | 流动性变化  |
| `volume_confluence`  | 成交量-结构共振信号 | `volume_zscore × structure_overlap_score` | float | [−1,1]  | 信号强化   |
| `volume_squeeze`     | 成交量压缩状态    | volume_volatility < 阈值                    | bool  | -       | 潜在爆发期  |
| `volume_spike_score` | 成交量激增得分    | `volume / mean(volume_hist)`              | float | log1p   | 异常成交检测 |
| `vwap_distance`      | 当前价距VWAP   | `(close−VWAP)/ATR`                        | float | 标准化     | 价值偏离识别 |
| `buy_sell_ratio`     | 主动买卖量比     | `buy/(buy+sell)`                          | float | [0,1]   | 买压识别   |

---

## ⚙️ 五、综合结构信号（Composite Structural Signal）

| 特征名称                      | 定义                                                        | 含义               |
| ------------------------- | --------------------------------------------------------- | ---------------- |
| `composite_signal`        | `structure_overlap_score × (1 + 0.5×tanh(volume_zscore))` | 结构共振 × 成交量共振综合强度 |
| `composite_signal_smooth` | EMA(composite_signal,5)                                   | 平滑后的结构强度         |
| `composite_breakout_bias` | composite_signal × sign(OLS_slope)                        | 方向偏向后的共振强度       |

> 🧠 用法建议：
>
> * 当 `composite_signal ≥ 0.7` 且成交量放大 → 有效突破区。
> * 当 `structure_overlap_score 高` 且 `volume_squeeze` 为真 → 压缩即将结束。

---

## 🧭 六、动态与自适应特征（Adaptive & State Features）

| 特征名称                    | 含义 / 定义 | 计算方法                               | 数据类型        | 归一化方法   | 应用说明   |
| ----------------------- | ------- | ---------------------------------- | ----------- | ------- | ------ |
| `dynamic_window_length` | 自适应窗口   | `base_window × f(ATR_percentile)`  | int         | /max    | 动态分析窗口 |
| `trend_slope_change`    | 趋势加速率   | ΔOLS斜率/OLS斜率_prev                  | float       | 标准化     | 趋势强度变化 |
| `entropy_change_rate`   | 熵变化率    | Δentropy / entropy_prev            | float       | 标准化     | 结构转变识别 |
| `compression_state`     | 当前市场状态  | compression / expansion / breakout | categorical | One-Hot | 状态机输入  |

---

## 🧮 七、特征组合与模型映射

| 模型目标  | 关键特征类别                                    | 示例                  |
| ----- | ----------------------------------------- | ------------------- |
| 趋势预测  | OLS斜率、ZigZag序列、HAL位置                      | `trend_prob`        |
| 压缩检测  | ATR分位、通道宽度、price_entropy                  | `compression_prob`  |
| 突破预测  | composite_signal、volume_confluence        | `breakout_prob`     |
| 反转预测  | pattern_M/W、poc_distance                  | `reversal_prob`     |
| 波动率预测 | ATR_compression_ratio、entropy_change_rate | `future_volatility` |

---

## 🧱 八、特征构建与流水线建议（Implementation Pipeline）

| 模块                   | 功能    | 实现建议                               |
| -------------------- | ----- | ---------------------------------- |
| `FeatureBuilder`     | 指标计算  | 用 pandas/vectorbt + numba 优化       |
| `FeatureNormalizer`  | 特征归一化 | ATR类→Zscore；成交量类→Percentile        |
| `TargetGenerator`    | 标签生成  | future_return / breakout_success   |
| `OnlineTrainer`      | 滚动训练  | 月/季度重训防漂移                          |
| `LightGBMModel`      | 模型拟合  | objective="binary"/"regression_l1" |
| `FeatureSelector`    | 特征重要性 | SHAP + 相关性过滤                       |
| `FeatureCompressor`  | 特征降维  | PCA / AutoEncoder                  |
| `MultiScaleEnsemble` | 多周期融合 | 5m/15m/1h stacking ensemble        |

---

✅ **重点总结**

* 所有价格类特征统一以 **ATR为基准** 表示相对位置；
* **结构重叠强度 (structure_overlap_score)** 是系统灵魂，用于识别高质量支撑/阻力；
* **成交量共振模块** 负责信号确认；
* 最终信号 `composite_signal` 可直接作为机器学习模型主特征；
* 可通过 LightGBM + SHAP 分析解释特征贡献；
* 滚动自适应（时间窗、权重、归一化）是提高稳健性的关键。

---

是否希望我现在直接生成一个配套的 **Python 实现模板 (`features_v3.py`)**，
自动计算这整套 ATR-归一化 + 结构共振 + 成交量强化特征？
