# WPT + SR 多尺度对齐系统

## 核心原则

**所有 SR 均由 WPT 重构价格生成，实现天然多尺度对齐，无需 resample。**

## 一、系统架构

### 1.1 WPT 重构价格（基础层）

```
原始价格 → WPT 分解 → 多尺度子带 → 重构价格
```

- **低频子带（'aaaa'）** → 长期 SR（周线级）
- **中频子带（'aaad', 'aada'）** → 中期 SR（日线级）
- **高频子带（'dddd'）** → 短期 SR（分钟级）

### 1.2 SR 特征生成（应用层）

所有 SR 特征（POC、HAL、Swing、ZigZag）都基于 WPT 重构价格计算：

```python
# 伪代码
wpt_price_reconstructed = wpt_trend + wpt_fluctuation
sr_poc = compute_poc(wpt_price_reconstructed)
sr_hal = compute_hal(wpt_price_reconstructed)
sr_swing = compute_swing(wpt_price_reconstructed)
sr_zigzag = compute_zigzag(wpt_price_reconstructed)
```

## 二、特征依赖关系

### 2.1 依赖图

```
wpt_price_reconstructed (基础)
    ├── sqs_hal_high
    ├── sqs_hal_low
    ├── sr_strength_max
    ├── wpt_vpvr
    ├── liquidity_void
    └── wpt_volume_energy
```

### 2.2 特征分类

| 类别 | 特征 | 用途 |
|------|------|------|
| **多尺度 SR 结构** | WPT 低频 POC / OLS<br>WPT 中频 Swing High/Low<br>WPT 高频 ZigZag<br>SR 重叠密度 | 定义"在哪里交易" |
| **价量动态** | 滚动 CVD（1D, 5D）<br>Take Buy Ratio（TBR）<br>VPER（Volume-Price Energy Ratio）<br>CVD 包络斜率（Hilbert） | 判断"是否有主力参与" |
| **时频信号处理** | WPT 各子带能量 & 熵<br>Hurst（全序列 + WPT 子带）<br>Hilbert 相位差（CVD vs Price 残差）<br>Spectrum 主频 & 带宽 | 捕捉"节奏与领先关系" |
| **流动性验证** | WPT 降噪 VPVR<br>流动性真空区识别<br>WPT + Volume 能量协同 | 识别"真假突破" |

## 三、策略专属特征集

### 3.1 SR 反转策略

**盈利逻辑**：价格触及强共识区（POC/Swing）后，因流动性枯竭或订单堆积而反弹。

**核心特征**：
- **SR 强度**：WPT 低频 POC 与当前价距离、SR 重叠数、VPVR 高量节点支撑强度
- **流动性验证**：CVD 在 SR 区的净买入斜率、Hilbert 相位差、Take Buy Ratio
- **波动状态**：ATR/Close、Bollinger Band Width

**应避免特征**：
- 趋势指标（如 MA 斜率、ADX）
- 全局收益率百分位

### 3.2 SR 突破策略

**盈利逻辑**：价格突破关键位后，是否有持续流动性跟进推动趋势延续。

**核心特征**：
- **突破质量**：突破时成交量/20日均量、VPER 中频 spike、突破后 3 根 K 线收盘站稳比例
- **动能持续性**：WPT 中低频能量比、Hurst 指数、ROC(5)/ROC(20) 比值
- **真空区识别**：突破方向上的 VPVR 低量节点距离、Spectrum 主频迁移

**应避免特征**：
- SR 重叠密度（高重叠区反而难突破）
- RSI 极值（突破常发生在超买/超卖区，无区分度）

### 3.3 压缩区突破策略

**盈利逻辑**：市场在极低波动下积蓄能量，一旦打破平衡，波动率扩张带来大行情。

**核心特征**：
- **压缩强度**：Bollinger Band Width 分位数（<10%）、ATR/MA(ATR) 比值（<0.8）、Spectrum 频谱平坦度
- **突破触发**：最近 3 根 K 线 range 扩张率、Volume spike（>2σ）、CVD 净流入突变
- **方向确认**：突破首根 K 线 body/range 比、Take Buy Ratio 方向一致性、WPT 高频子带能量方向

**推荐模型**：CatBoost（对类别性特征和条件交互更高效，小样本更稳健）

**应避免特征**：
- 长期趋势指标（如 200MA）
- 全局 CVD 累积值

### 3.4 趋势跟踪策略

**盈利逻辑**：在强趋势市场中，跟随主力资金方向，吃主升浪。

**核心特征**：
- **趋势强度**：Hurst 指数（>0.6）、ADX(14) > 25、WPT 低频子带斜率
- **资金验证**：滚动 5D CVD 趋势（斜率）、MFE/MAE 比、Volume 与 price 同向率
- **节奏感知**：Hilbert 瞬时频率稳定性、Spectrum 主频周期长度、Mamba embedding

**应避免特征**：
- 局部 SR 位置（趋势中 SR 经常被无视）
- 短期波动率（趋势初期常伴随高波动）

## 四、流动性特征详解

### 4.1 WPT 降噪 VPVR

**核心思想**：使用 WPT 对价格进行降噪，剔除高频噪声，基于降噪后的价格构建 VPVR。

**输出特征**：
- `vpvr_pvp`: Point of Control（最高成交量对应的价格）
- `vpvr_hvn_count`: High Volume Node 数量
- `vpvr_lvn_count`: Low Volume Node 数量（流动性真空区）
- `vpvr_lvn_distance`: 当前价格到最近 LVN 的距离（归一化）
- `vpvr_volume_density`: 当前价格的成交量密度
- `vpvr_price_in_lvn`: 当前价格是否在 LVN 中（1.0/0.0）

### 4.2 流动性真空区识别

**核心认知**：流动性真空区 ≠ 历史低成交量区域，而是当前订单簿深度缺失。

**输出特征**：
- `liquidity_void_detected`: 是否检测到流动性真空区（1.0/0.0）
- `liquidity_void_speed`: 价格速度（归一化）
- `liquidity_void_volume_ratio`: 成交量比率（当前 vs. 参考）
- `liquidity_void_retracement`: 后续回撤（1~3 根 K 线内）
- `liquidity_void_false_breakout_risk`: 假突破风险评分（0-1）

### 4.3 WPT + Volume 能量协同分析

**核心指标**：
- **VPER (Volume-Price Energy Ratio)**：量价能量比
- **能量下移（Energy Cascade）**：高频能量向中低频转移
- **多尺度一致性**：至少两个中低频子带同时出现能量上升
- **真假突破评分**：基于多尺度能量和 VPER 的综合评分

**输出特征**：
- `wpt_vper_low`: 低频 VPER
- `wpt_vper_mid`: 中频 VPER
- `wpt_vper_high`: 高频 VPER
- `wpt_energy_cascade`: 能量下移指标
- `wpt_multi_scale_consistency`: 多尺度一致性评分（0-1）
- `wpt_breakout_confidence`: 突破置信度评分（0-1）
- `wpt_false_breakout_risk`: 假突破风险评分（0-1）

## 五、使用示例

### 5.1 加载策略特征

```python
from src.features.loader.strategy_feature_loader import StrategyFeatureLoader

loader = StrategyFeatureLoader()
df_features = loader.load_strategy_features(df_raw, strategy="sr_reversal")
```

### 5.2 手动提取流动性特征

```python
from src.features.time_series.utils_liquidity_features import extract_liquidity_features

# 提取所有流动性特征
df = extract_liquidity_features(df, feature_type="all")

# 仅提取 VPVR
df = extract_liquidity_features(df, feature_type="vpvr")

# 仅提取流动性真空区
df = extract_liquidity_features(df, feature_type="void")

# 仅提取 WPT + Volume 能量
df = extract_liquidity_features(df, feature_type="energy")
```

## 六、优势总结

| 优势 | 说明 |
|------|------|
| **多尺度对齐** | 所有 SR 基于 WPT 重构价格，天然多尺度对齐，无需 resample |
| **逻辑解耦** | 每个模型只学一种盈利模式，无内部冲突 |
| **风险隔离** | SR 反转失效 ≠ 趋势模型失效 |
| **特征聚焦** | SR 模型专注结构，趋势模型专注动量 |
| **可解释性高** | 每个信号都有明确来源（如"CVD 相位领先 + POC 支撑"） |

## 七、配置文件

- **特征依赖配置**：`config/feature_dependencies.yaml`
- **策略特征配置**：`config/strategy_features.yaml`
- **特征函数映射**：`src/features/loader/feature_function_mapping.py`

## 八、注意事项

1. **严格因果关系**：所有特征都使用 `shift(1)` 确保严格因果关系
2. **WPT 参数**：默认使用 `db4` 小波，`level=4`，可根据数据频率调整
3. **滚动窗口**：VPVR 使用 100 根 K 线窗口，流动性真空区使用 20 根 K 线窗口
4. **特征选择**：根据策略类型选择专属特征集，避免引入噪声

