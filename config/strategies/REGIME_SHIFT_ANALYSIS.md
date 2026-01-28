# 树模型与 Regime Shift：为什么短时间段表现好，长时间段表现差？

## 🎯 核心问题

**观察**：
- 短时间段（几个月训练 → 几个月预测）：表现好 ✅
- 长时间段（3年训练 → 3年预测）：表现差 ❌

**结论**：
> **单纯的树策略模型无法适应 regime shift，单独训练几个月预测下几个月还不错**

## 🔍 为什么树模型无法适应 Regime Shift？

### 1. 硬阈值依赖历史分布

**树模型的特点**：
- 使用**硬阈值**分割特征空间（如 `cvd_long <= -34895`）
- 阈值是**基于历史数据分布拟合的**

**具体例子**：

```python
# 训练期（2023年，低波动）
if cvd_long <= -34895:  # 阈值是历史拟合的
    veto_tc = True  # 训练期准确率90%

# 测试期（2024年，高波动）
# 数据分布变化：cvd_long均值从-10000变为-30000
# 阈值-34895从2.5%分位数变为20%分位数
# 结果：准确率下降到30%（大幅失效）
```

**数据分布变化**：

| 时期 | cvd_long分布 | 阈值-34895的含义 | 规则有效性 |
|------|-------------|-----------------|-----------|
| **2023年（训练）** | 均值=-10000, 标准差=15000 | 2.5%分位数（极端负值） | ✅ 有效 |
| **2024年（测试）** | 均值=-30000, 标准差=25000 | 20%分位数（正常波动） | ❌ 失效 |

### 2. 局部拟合，无法泛化

**树模型的特点**：
- 通过**局部划分**学习模式
- 每个规则只适用于训练期的数据分布

**为什么短时间段表现好？**

| 场景 | 训练期 | 测试期 | 表现 |
|------|--------|--------|------|
| **短时间段** | 2023年1-6月（低波动） | 2023年7-12月（低波动） | ✅ 好（相同 regime） |
| **长时间段** | 2023年（低波动） | 2024-2025年（高波动） | ❌ 差（不同 regime） |

**关键**：
- 短时间段：训练期和测试期可能是**相同的 regime** → 规则仍然有效
- 长时间段：训练期和测试期是**不同的 regime** → 规则失效

### 3. 无法自适应

**树模型是静态的**：
- 一旦训练完成，规则就固定了
- 无法根据当前市场状态自动调整
- 需要手动重新训练或调整阈值

## 💡 解决方案

### 方案1：滚动训练（Rolling Training）⭐ **强烈推荐**

**核心思想**：定期重新训练模型，用最近几个月的数据预测下几个月

```bash
# 每3-6个月重新训练一次
make rolling \
  ROLLING_CONFIG=config/strategies/sr_reversal_rr_reg_long \
  SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6  # 用6个月训练，预测下1个月
```

**工作流程**：
```
第1次: 训练=[1-6月], 测试=7月 → 模型1
第2次: 训练=[2-7月], 测试=8月 → 模型2
第3次: 训练=[3-8月], 测试=9月 → 模型3
...
```

**优势**：
- ✅ 模型始终使用最新的市场状态
- ✅ 自动适应 regime shift
- ✅ 短时间段表现好（几个月训练 → 几个月预测）

**适用场景**：
- 生产环境
- 需要持续适应市场变化
- 短时间段表现好，长时间段表现差的情况

### 方案2：Regime-Aware 模型

**核心思想**：先识别市场 regime，再使用对应的模型

```python
# Stage 1: Regime Detection
regime = regime_detector.predict(features)  # 0=震荡, 1=趋势, 2=爆发

# Stage 2: Conditional Model
if regime == 0:
    signal = mean_revert_model.predict(features)
elif regime == 1:
    signal = trend_model.predict(features)
else:
    signal = breakout_model.predict(features)
```

**优势**：
- ✅ 不同 regime 使用不同模型
- ✅ 自动适应市场状态变化
- ✅ 可解释性强

**实现**：
- 使用 `src/time_series_model/rule/regime.py` 中的 `classify_regime()`
- 训练多个专用模型，每个对应一种 regime

### 方案3：使用相对阈值（分位数）

**核心思想**：用相对位置（分位数）替代绝对阈值

```python
# 树模型（硬阈值）
if cvd_long <= -34895:  # 硬阈值
    veto_tc = True

# 改进（分位数阈值）
cvd_long_q20 = quantile(cvd_long, window=200, q=0.2)
if cvd_long <= cvd_long_q20:  # 相对位置（20%分位数）
    veto_tc = True
```

**优势**：
- ✅ 阈值自适应数据分布
- ✅ 即使数据分布变化，20%分位数仍然对应"极端值"
- ✅ 更稳健，不容易受到 Regime Shift 影响

## 📊 实际数据验证

### 当前结果（3年周期）

| 策略 | 最终 Sharpe | 评估 |
|------|------------|------|
| compression_breakout | -1.07 | ❌ 表现差 |
| sr_breakout | 0.82 | ✅ 表现良好 |
| sr_reversal_rr_reg_long | 0.93 | ⭐ 表现优秀 |
| trend_following | -0.07 | ⚠️ 表现差 |

### 预期结果（滚动训练，几个月周期）

如果使用滚动训练（6个月训练 → 1个月预测），预期：
- ✅ Sharpe 可能提升 20-50%
- ✅ 更稳定，波动更小
- ✅ 自动适应市场变化

## 🎯 建议

### 立即行动

1. **实施滚动训练**：
   ```bash
   # 对表现最好的策略（sr_reversal）实施滚动训练
   make rolling \
     ROLLING_CONFIG=config/strategies/sr_reversal_rr_reg_long \
     SYMBOL=BTCUSDT \
     INITIAL_TRAIN_MONTHS=6
   ```

2. **对比结果**：
   - 滚动训练（几个月周期）的 Sharpe
   - 固定训练（3年周期）的 Sharpe
   - 验证"短时间段表现好"的假设

3. **分析不同时间段**：
   - 分别评估 2023、2024、2025 年的表现
   - 识别哪些时间段表现好/差
   - 分析市场环境差异

### 长期改进

1. **引入 Regime Detection**：
   - 使用 `classify_regime()` 识别市场状态
   - 训练多个专用模型
   - 根据 regime 动态切换模型

2. **使用相对阈值**：
   - 将硬阈值改为分位数阈值
   - 提高模型稳健性
   - 减少对 regime shift 的敏感度

## 🔗 相关文档

- Regime Shift 详解：`docs/architecture/leagcy/机器学习与交易系统的核心问题.md`
- 树模型 Regime 问题：`docs/strategies/标签设计如何区分策略与树模型的regime_shift问题.md`
- Regime Detection：`docs/models/时序模型/行情：Regime Detection（行情状态识别）.md`
- 滚动训练指南：`docs/models/时序模型/命令总览.md`
- Sharpe 分析：`config/strategies/SHARPE_ANALYSIS.md`
