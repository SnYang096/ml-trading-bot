# 标签设计如何区分策略 + 树模型的Regime Shift问题

本文档详细说明：
1. **不同标签设计如何导致模型学习到不同的模式，从而区分策略**
2. **树模型为什么容易受到regime shift影响**

---

## 一、标签设计如何区分策略（详细举例）

### 1.1 核心原理

**关键洞察**：不同的标签定义 → 不同的优化目标 → 模型学习到不同的模式

即使使用**相同的特征**，不同的标签会导致：
- 模型关注不同的样本
- 模型学习不同的特征组合
- 模型输出不同的预测

### 1.2 具体例子：同一个市场状态，不同的标签

假设在某个时间点`t`，市场状态如下：
- `cvd_long = -40000`（长期订单流负）
- `bb_width_normalized = 5.0`（波动率中等）
- `volume_ratio = 1.5`（成交量放大）
- `vp_poc_deviation = -0.03`（POC偏离负）

未来50根K线内的实际路径：
- MFE = 2.5 * ATR（最大有利偏移）
- MAE = 1.0 * ATR（最大不利偏移）
- 未来收益率 = +0.05（5%）

#### 场景A：sr_reversal_rr_reg_long（R/R标签）

**标签计算**：
```python
rr_label = min(max(MFE / MAE, 0), 3.0)
# = min(max(2.5 / 1.0, 0), 3.0)
# = 2.5
```

**模型学习目标**：
- 预测R/R比率（连续值，范围[0, 3.0]）
- 优化目标：最小化MSE（预测R/R vs 实际R/R）

**模型会学习什么？**
- 关注"什么情况下R/R会好"
- 学习特征组合：`cvd_long负 + vp_poc_deviation负` → 高R/R
- 原因：订单流负但结构集中 → 反转机会 → R/R好

**训练过程**（简化）：
```python
# 样本1: cvd_long=-40000, vp_poc_deviation=-0.03, rr_label=2.5
# 模型学习: 当cvd_long很负且vp_poc_deviation负时 → 预测高R/R

# 样本2: cvd_long=-40000, vp_poc_deviation=0.1, rr_label=1.0
# 模型学习: 当cvd_long负但vp_poc_deviation正时 → 预测低R/R

# 最终规则（简化）:
if cvd_long <= -35000 and vp_poc_deviation <= -0.02:
    predicted_rr = 2.3  # 高R/R
else:
    predicted_rr = 1.0  # 低R/R
```

**关键点**：
- 模型**不关心**绝对收益（+5%）
- 模型**只关心**R/R比率（2.5）
- 即使未来收益是负的，只要R/R好（MFE/MAE > 2），标签仍然是高的

#### 场景B：trend_following（百分位标签）

**标签计算**：
```python
future_return = +0.05  # 5%
# 在200个样本的滚动窗口中排名
rank = 150 / 200 = 0.75  # 排名第150，百分位75%
label = 0.75
```

**模型学习目标**：
- 预测趋势相对强度rank（连续值，范围[0, 1]）
- 优化目标：最小化MSE（预测rank vs 实际rank）

**模型会学习什么？**
- 关注"什么情况下趋势强度rank会高"
- 学习特征组合：`cvd_long负 + bb_width_normalized低` → 低rank
- 原因：订单流负且波动压缩 → 趋势弱 → rank低

**训练过程**（简化）：
```python
# 样本1: cvd_long=-40000, bb_width_normalized=5.0, label=0.75
# 模型学习: 当cvd_long负且bb_width中等时 → 预测中等rank

# 样本2: cvd_long=-40000, bb_width_normalized=3.0, label=0.3
# 模型学习: 当cvd_long负且bb_width低时 → 预测低rank

# 最终规则（简化）:
if cvd_long <= -35000 and bb_width_normalized <= 4.0:
    predicted_rank = 0.2  # 低rank（趋势弱）
else:
    predicted_rank = 0.7  # 高rank（趋势强）
```

**关键点**：
- 模型**不关心**绝对收益（+5%）
- 模型**只关心**相对强度rank（0.75）
- 即使未来收益是正的，如果在200个样本中排名低，标签仍然是低的

#### 场景C：compression_breakout（三元标签）

**标签计算**：
```python
if compression_score > threshold:
    if price_breaks_up and volume_confirms:
        label = +1  # 向上有效突破
    elif price_breaks_down and volume_confirms:
        label = -1  # 向下有效突破
    else:
        label = 0   # 假突破/回补
else:
    label = NaN     # 不在压缩区
```

假设在这个例子中：价格向上突破，成交量确认 → `label = +1`

**模型学习目标**：
- 预测方向+质量（离散值：-1/0/+1）
- 优化目标：最小化交叉熵（多分类）

**模型会学习什么？**
- 关注"什么情况下是真突破（+1）vs 假突破（0）"
- 学习特征组合：`volume_ratio高 + cvd_change_5正` → 真突破（+1）
- 原因：放量 + 订单流支持 → 真突破

**训练过程**（简化）：
```python
# 样本1: volume_ratio=1.5, cvd_change_5=5000, label=+1
# 模型学习: 当放量且订单流支持时 → 预测+1（真突破）

# 样本2: volume_ratio=1.5, cvd_change_5=-1000, label=0
# 模型学习: 当放量但订单流不支持时 → 预测0（假突破）

# 最终规则（简化）:
if volume_ratio > 1.3 and cvd_change_5 > 3000:
    predicted_label = +1  # 真突破
elif volume_ratio > 1.3 and cvd_change_5 < -500:
    predicted_label = 0   # 假突破
else:
    predicted_label = -1  # 向下突破
```

**关键点**：
- 模型**必须同时判断**方向和真假
- 0是"主动拒绝"，不是"不确定"
- 模型学习的是"突破质量"，不是"收益"

### 1.3 为什么不同标签能区分策略？

#### 原因1：优化目标不同

| 策略 | 标签类型 | 优化目标 | 模型关注点 |
|------|---------|---------|-----------|
| **sr_reversal** | R/R比率 | 最小化R/R预测误差 | "什么情况下R/R会好" |
| **trend_following** | 相对强度rank | 最小化rank预测误差 | "什么情况下趋势强度rank会高" |
| **compression_breakout** | 方向+质量 | 最小化分类误差 | "什么情况下是真突破" |

#### 原因2：损失函数不同

**sr_reversal（回归）**：
```python
loss = MSE(predicted_rr, actual_rr)
# 关注：预测R/R是否接近实际R/R
# 对"绝对收益"不敏感，只关心R/R比率
```

**trend_following（回归）**：
```python
loss = MSE(predicted_rank, actual_rank)
# 关注：预测rank是否接近实际rank
# 对"绝对收益"不敏感，只关心相对强度
```

**compression_breakout（多分类）**：
```python
loss = CrossEntropy(predicted_class, actual_class)
# 关注：预测类别是否正确
# 必须同时判断方向和真假
```

#### 原因3：样本权重不同

**sr_reversal**：
- 高R/R的样本权重更高
- 模型更关注"R/R好的情况"

**trend_following**：
- 极端rank的样本权重更高（趋势很强或很弱）
- 模型更关注"趋势极端的情况"

**compression_breakout**：
- 真突破（+1/-1）的样本权重更高
- 假突破（0）的样本权重较低
- 模型更关注"真突破的情况"

### 1.4 完整例子：同一个特征，不同策略学习到不同模式

**特征**：`cvd_long <= -35000`

#### 在sr_reversal中

**标签**：R/R比率
**学习到的模式**：
```
if cvd_long <= -35000 and vp_poc_deviation <= -0.02:
    predicted_rr = 2.5  # 高R/R（反转机会）
```

**语义**：订单流负 + 结构集中 → 反转机会 → R/R好

#### 在trend_following中

**标签**：趋势强度rank
**学习到的模式**：
```
if cvd_long <= -35000 and bb_width_normalized <= 4.0:
    predicted_rank = 0.2  # 低rank（趋势弱）
```

**语义**：订单流负 + 波动压缩 → 趋势弱 → rank低

#### 在compression_breakout中

**标签**：方向+质量（-1/0/+1）
**学习到的模式**：
```
if cvd_long <= -35000 and volume_ratio > 1.5:
    predicted_label = 0  # 假突破（订单流不支持）
```

**语义**：订单流负 + 放量 → 假突破 → label=0

### 1.5 训练目标的选择如何影响模型

#### 回归任务（sr_reversal, trend_following）

**特点**：
- 输出连续值
- 损失函数：MSE
- 模型学习"平滑的映射"

**影响**：
- 模型可以学习"程度"（R/R是2.0还是2.5）
- 模型可以学习"相对关系"（rank是0.7还是0.8）

#### 多分类任务（compression_breakout）

**特点**：
- 输出离散值（-1/0/+1）
- 损失函数：交叉熵
- 模型学习"硬分类边界"

**影响**：
- 模型必须学习"明确的分类边界"
- 模型不能学习"程度"（要么是真突破，要么是假突破）

---

## 二、树模型的Regime Shift问题

### 2.1 什么是Regime Shift？

**Regime Shift**：市场状态的根本性变化，导致历史训练的模式失效。

**例子**：
- 训练期：低波动、趋势明显
- 测试期：高波动、震荡为主
- 结果：模型在训练期表现好，但在测试期表现差

### 2.2 为什么树模型容易受到Regime Shift影响？

#### 原因1：硬阈值依赖

**树模型的特点**：
- 使用**硬阈值**分割特征空间
- 阈值是**历史拟合的**，对数据分布敏感

**具体例子**：

**训练期（2023年，低波动）**：
```python
# 树模型学习到的规则
if cvd_long <= -34895 and bb_width_normalized <= 4.86:
    veto_tc = True  # 趋势不健康
```

**为什么这个阈值有效？**
- 在2023年的数据分布中，`cvd_long <= -34895`确实对应"趋势不健康"
- 阈值`-34895`是历史数据的某个分位数

**测试期（2024年，高波动）**：
```python
# 同样的规则
if cvd_long <= -34895 and bb_width_normalized <= 4.86:
    veto_tc = True
```

**问题**：
- 2024年市场波动更大，`cvd_long`的分布发生了变化
- 在2024年，`cvd_long <= -34895`可能对应"正常波动"，不再是"趋势不健康"
- 但模型仍然使用**硬阈值**`-34895`，导致误判

**数据分布变化**：

| 时期 | cvd_long分布 | 阈值-34895的含义 |
|------|-------------|-----------------|
| **2023年（训练）** | 均值=-10000, 标准差=15000 | 2.5%分位数（极端负值） |
| **2024年（测试）** | 均值=-30000, 标准差=25000 | 20%分位数（正常波动） |

**结果**：
- 训练期：阈值`-34895`确实对应"极端情况"（2.5%分位数）
- 测试期：阈值`-34895`对应"正常情况"（20%分位数）
- 模型**过度否决**，导致错过很多好机会

#### 原因2：局部拟合

**树模型的特点**：
- 通过**局部划分**学习模式
- 每个叶子节点只覆盖特征空间的**局部区域**

**具体例子**：

**训练期**：
```python
# 树模型学习到的规则（简化）
if cvd_long <= -34895 and bb_width_normalized <= 4.86:
    # 这个区域：100个样本，90个是"趋势不健康"
    veto_tc = True
elif cvd_long > -34895 and bb_width_normalized > 4.86:
    # 这个区域：200个样本，180个是"趋势健康"
    veto_tc = False
```

**问题**：
- 规则是**局部拟合**的，只适用于训练期的数据分布
- 当数据分布变化时，这些局部区域的含义也变化了

**测试期**：
```python
# 同样的规则
if cvd_long <= -34895 and bb_width_normalized <= 4.86:
    # 这个区域：100个样本，但只有30个是"趋势不健康"（分布变了）
    veto_tc = True  # 但实际应该veto的只有30个，不是90个
```

**结果**：
- 模型在60%的样本上**误判**（应该veto但没veto，或不应该veto但veto了）

#### 原因3：特征交互的脆弱性

**树模型的特点**：
- 学习**特征组合**（如`cvd_long AND bb_width_normalized`）
- 这些组合是**历史拟合的**，对数据分布敏感

**具体例子**：

**训练期**：
```python
# 树模型学习到的规则
if cvd_long <= -34895 AND bb_width_normalized <= 4.86:
    # 这个组合：在训练期，90%的情况下是"趋势不健康"
    veto_tc = True
```

**为什么这个组合有效？**
- 在训练期，`cvd_long`和`bb_width_normalized`的**联合分布**使得这个组合确实对应"趋势不健康"

**测试期**：
```python
# 同样的规则
if cvd_long <= -34895 AND bb_width_normalized <= 4.86:
    # 这个组合：在测试期，只有30%的情况下是"趋势不健康"（联合分布变了）
    veto_tc = True  # 但实际应该veto的只有30%
```

**问题**：
- `cvd_long`和`bb_width_normalized`的**相关性**在测试期发生了变化
- 在训练期，两者高度相关（订单流负时波动也压缩）
- 在测试期，两者相关性降低（订单流负时波动可能不压缩）

**结果**：
- 模型在70%的样本上**误判**

### 2.3 完整例子：Regime Shift导致模型失效

#### 场景：从低波动趋势市场到高波动震荡市场

**训练期（2023年1-6月）**：
- 市场特征：低波动、趋势明显、订单流稳定
- 数据分布：
  - `cvd_long`：均值=-10000, 标准差=15000
  - `bb_width_normalized`：均值=6.0, 标准差=2.0
  - 两者相关性：0.7（高度相关）

**树模型学习到的规则**：
```python
# Rule 1: 趋势不健康（TC Gate）
if cvd_long <= -34895 and bb_width_normalized <= 4.86:
    veto_tc = True
    # 训练期准确率：90%

# Rule 2: 趋势健康（TC允许）
if cvd_long > -10000 and bb_width_normalized > 5.0:
    allow_tc = True
    # 训练期准确率：85%
```

**测试期（2024年1-6月）**：
- 市场特征：高波动、震荡为主、订单流不稳定
- 数据分布：
  - `cvd_long`：均值=-30000, 标准差=25000（分布右移且变宽）
  - `bb_width_normalized`：均值=8.0, 标准差=3.0（分布右移且变宽）
  - 两者相关性：0.3（相关性降低）

**同样的规则在测试期的表现**：
```python
# Rule 1: 趋势不健康（TC Gate）
if cvd_long <= -34895 and bb_width_normalized <= 4.86:
    veto_tc = True
    # 测试期准确率：30%（大幅下降！）
    # 原因：阈值-34895在测试期不再是"极端值"，而是"正常值"

# Rule 2: 趋势健康（TC允许）
if cvd_long > -10000 and bb_width_normalized > 5.0:
    allow_tc = True
    # 测试期准确率：40%（大幅下降！）
    # 原因：阈值-10000在测试期不再是"正常值"，而是"极端值"
```

**结果**：
- 模型在训练期表现好（Sharpe=1.5）
- 模型在测试期表现差（Sharpe=-0.3）
- **Regime Shift导致模型失效**

### 2.4 为什么分层架构能缓解Regime Shift？

#### 方案1：使用分位数阈值（相对位置）

**树模型（硬阈值）**：
```python
if cvd_long <= -34895:  # 硬阈值
    veto_tc = True
```

**分层架构（分位数阈值）**：
```python
# 计算动态分位数
cvd_long_q20 = quantile(cvd_long, window=200, q=0.2)

if cvd_long <= cvd_long_q20:  # 相对位置（20%分位数）
    veto_tc = True
```

**优势**：
- 阈值**自适应**数据分布
- 即使数据分布变化，20%分位数仍然对应"极端值"
- 更稳健，不容易受到Regime Shift影响

#### 方案2：使用路径原语（策略无关）

**树模型（策略特定）**：
```python
# 直接输出策略信号
reversal_signal = reversal_model.predict(features)
# 信号依赖于训练期的数据分布
```

**分层架构（路径原语）**：
```python
# 输出路径原语（策略无关）
primitives = nn_model.predict(features)
# primitives = {'dir': 0.7, 'mfe_atr': 2.5, 'mae_atr': 1.0}

# Router根据primitives选择regime（可以自适应）
if primitives['mfe_atr'] / primitives['mae_atr'] > 2.0:
    regime = 'TREND'
```

**优势**：
- 路径原语是**策略无关**的，不依赖于特定策略的数据分布
- Router可以根据**当前市场状态**（primitives）选择regime
- 更灵活，更容易适应Regime Shift

#### 方案3：使用Gate规则（硬约束 + 软学习）

**树模型（全部硬规则）**：
```python
# 所有规则都是硬阈值
if cvd_long <= -34895:
    veto_tc = True
```

**分层架构（Gate + Safety Head）**：
```python
# Gate：硬规则（但使用分位数）
if cvd_long <= quantile(cvd_long, 0.2):
    veto_tc = True

# Safety Head：软学习（可以适应）
safety_prob = safety_head(primitives, regime_emb)
if safety_prob > 0.7:
    veto_tc = True
```

**优势**：
- Gate提供**硬约束**（但使用相对位置）
- Safety Head提供**软学习**（可以适应新regime）
- 两者结合，既稳健又灵活

### 2.5 Regime Shift的典型场景

#### 场景1：波动率regime变化

**训练期**：低波动（ATR percentile = 0.3）
**测试期**：高波动（ATR percentile = 0.8）

**树模型问题**：
- 硬阈值`bb_width_normalized <= 4.86`在训练期有效
- 在测试期，波动率整体提高，阈值失效

**分层架构优势**：
- 使用分位数阈值：`bb_width_normalized <= quantile(bb_width, 0.2)`
- 即使波动率整体提高，20%分位数仍然对应"低波动"

#### 场景2：订单流regime变化

**训练期**：订单流稳定（cvd_long分布集中）
**测试期**：订单流不稳定（cvd_long分布分散）

**树模型问题**：
- 硬阈值`cvd_long <= -34895`在训练期有效
- 在测试期，订单流分布变化，阈值失效

**分层架构优势**：
- 使用路径原语：`dir`, `mfe_atr`, `mae_atr`
- 这些原语是**相对值**（相对于ATR），不依赖于绝对订单流值

#### 场景3：相关性regime变化

**训练期**：`cvd_long`和`bb_width_normalized`高度相关（0.7）
**测试期**：两者相关性降低（0.3）

**树模型问题**：
- 特征组合`cvd_long <= -34895 AND bb_width_normalized <= 4.86`在训练期有效
- 在测试期，相关性降低，组合失效

**分层架构优势**：
- 使用路径原语，不依赖于特征之间的相关性
- Router根据primitives选择regime，不依赖于特征组合

---

## 三、总结

### 3.1 标签设计如何区分策略

**核心机制**：
1. **不同标签 → 不同优化目标**
   - R/R标签 → 优化R/R比率
   - 百分位标签 → 优化相对强度
   - 三元标签 → 优化方向+质量

2. **不同优化目标 → 模型学习不同模式**
   - 即使使用相同特征，模型关注不同的样本
   - 模型学习不同的特征组合
   - 模型输出不同的预测

3. **不同训练目标 → 不同的学习方式**
   - 回归任务 → 学习平滑映射
   - 多分类任务 → 学习硬分类边界

### 3.2 树模型的Regime Shift问题

**核心原因**：
1. **硬阈值依赖**：阈值是历史拟合的，对数据分布敏感
2. **局部拟合**：规则是局部拟合的，只适用于训练期的数据分布
3. **特征交互脆弱**：特征组合依赖于历史的相关性，容易失效

**典型场景**：
1. 波动率regime变化
2. 订单流regime变化
3. 相关性regime变化

**解决方案**：
1. 使用分位数阈值（相对位置）
2. 使用路径原语（策略无关）
3. 使用Gate规则（硬约束 + 软学习）

### 3.3 为什么分层架构更稳健

**关键优势**：
1. **策略解耦**：NN学习通用原语，不绑定具体策略
2. **相对位置**：使用分位数阈值，自适应数据分布
3. **软硬结合**：Gate提供硬约束，Safety Head提供软学习

**结果**：
- 更不容易受到Regime Shift影响
- 更容易适应新的市场状态
- 更容易诊断和修复问题
