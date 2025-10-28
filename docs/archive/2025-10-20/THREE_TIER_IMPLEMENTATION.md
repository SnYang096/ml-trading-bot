# 三层分层架构实现总结

## 🎯 设计理念

基于文档 `/docs/底层原理/分层注意力.md` 的核心思想：

```
战略层(4H) → 战术层(15m) → 执行层(1m)
Direction   →  Location   →   Timing
能否交易？  →  在哪儿交易？→  何时交易？
```

## ✅ 已完成的实现

### 1. 配置层 (`config.yaml`)

```yaml
# 三层结构定义
bar_types:
  "1m": "BTCUSDT.BINANCE-1-MINUTE-LAST-INTERNAL"   # 执行层
  "15m": "BTCUSDT.BINANCE-15-MINUTE-LAST-INTERNAL"  # 战术层
  "4h": "BTCUSDT.BINANCE-4-HOUR-LAST-INTERNAL"      # 战略层

# 三层职责配置
three_tier:
  strategic_layer: "4h"   # 定方向
  tactical_layer: "15m"   # 定结构
  execution_layer: "1m"   # 定入场
  
  layer_roles:
    "4h":  # 战略层
      role: "direction"
      features: ["market_state", "trend_bias", "cvd_direction"]
      min_confidence: 0.5
      
    "15m":  # 战术层
      role: "structure"
      features: ["dynamic_sr", "volume_profile", "liquidity_zones"]
      min_confidence: 0.4
      
    "1m":  # 执行层
      role: "timing"
      features: ["candlestick_pattern", "volume_spike", "momentum"]
      min_confidence: 0.3
```

### 2. 核心模块 (`three_tier_layer.py`)

**类结构**:
- `LayerDecision`: 单层决策结果
- `ThreeTierDecision`: 三层综合决策
- `ThreeTierLayer`: 三层决策系统

**关键方法**:

#### 战略层 (`make_strategic_decision`)
```python
# 输入：4H的市场状态、趋势偏向、CVD流向
# 输出：方向(long/short)、置信度
# 注意力：哪个宏观信号最可信？
```

#### 战术层 (`make_tactical_decision`)  
```python
# 输入：SR区间、Volume Profile、15m数据
# 输出：最佳交易区间、置信度
# 注意力：哪条SR线最可能起作用？
```

#### 执行层 (`make_execution_decision`)
```python
# 输入：1m K线、量能、动量
# 输出：入场触发信号、置信度
# 注意力：哪个K线信号最可靠？
```

#### 融合决策 (`fuse_three_tiers`)
```python
# 决策逻辑：
# 1. 战略层confidence >= 0.5 ✓
# 2. 战术层confidence >= 0.4 ✓  
# 3. 执行层confidence >= 0.3 ✓
# → 三层都通过才能开仓
```

### 3. 策略集成 (`strategy.py`)

```python
# 时间周期定义
self.timeframes = ['1m', '15m', '4h']
self.execution_tf = '1m'   # 执行层：定入场
self.tactical_tf = '15m'    # 战术层：定结构  
self.strategic_tf = '4h'    # 战略层：定方向

# 三层模型初始化
self.models = {
    '1m': DynamicSRModel('1m', self.cfg),   # 执行层模型
    '15m': DynamicSRModel('15m', self.cfg),  # 战术层模型
    '4h': DynamicSRModel('4h', self.cfg),    # 战略层模型
}

# 三层决策系统
self.three_tier = ThreeTierLayer(config=self.cfg)
```

## 🔄 决策流程

```
1. 【4H Bar 到达】
   ├─ 更新4H数据
   ├─ 战略层决策：判断趋势方向
   └─ 输出：Direction (long/short/neutral)

2. 【15m Bar 到达】
   ├─ 更新15m数据
   ├─ 战术层决策：识别SR和价值区
   └─ 输出：Location (price_zone)

3. 【1m Bar 到达】
   ├─ 更新1m数据
   ├─ 执行层决策：捕捉入场时机
   ├─ 融合三层决策
   └─ 如果三层都通过 → 开仓

4. 【开仓】
   ├─ 只在1m上执行
   ├─ 使用战略层的方向
   ├─ 使用战术层的价格区间
   └─ 使用执行层的精确时机
```

## 📊 分层注意力机制

### 战略层注意力 (4H)
```python
# 问题：哪个宏观信号最可信？
weights = {
    "cvd_direction": 0.5,   # CVD最诚实（资金流向）
    "trend_bias": 0.3,       # 趋势偏向
    "market_state": 0.2      # 市场状态
}
```

### 战术层注意力 (15m)
```python
# 问题：哪条SR线最可能起作用？
for each SR:
    distance_weight = 1.0 / (1.0 + distance)  # 距离越近权重越高
    strength_weight = sr.strength              # 强度越高权重越高
    attention = distance_weight * strength_weight
```

### 执行层注意力 (1m)
```python
# 问题：哪个K线信号最可靠？
weights = {
    "volume_spike": 0.4,      # 量能最重要
    "candle_pattern": 0.35,   # K线形态次之
    "momentum": 0.25          # 动量确认
}
```

## 🎯 核心优势

### 1. 逻辑清晰
- 每层职责明确，不混淆
- Direction → Location → Timing 完美闭环

### 2. 抗过拟合
- 每层只做一个决策
- 参数空间小
- 易于回测验证

### 3. 可解释性强
```
✅ 战略层(0.75): 多头趋势, CVD流入
✅ 战术层(0.68): support@60,200, 强度0.82
✅ 执行层(0.54): engulfing, 量增2.3x
→ 开仓做多 @ 60,250
```

### 4. 实战可操作
- 盯盘时能清楚看到每层的状态
- 每层输出明确
- 决策链可追溯

## 🔧 待完成的工作

### 1. 数据加载
- 需要1m级别的历史数据
- 4H数据可以从1m聚合

### 2. 信号处理逻辑调整
- 修改 `_process_signals()` 使用三层决策
- 确保只在1m上执行开仓
- 其他层只做决策判断

### 3. 测试验证
```bash
# 单层测试
pytest tests/test_strategic_layer.py
pytest tests/test_tactical_layer.py  
pytest tests/test_execution_layer.py

# 集成测试
make backtest-dynamic-sr-btc
```

## 📝 配置示例

```yaml
# 保守配置：三层都要求高置信度
three_tier:
  layer_roles:
    "4h":
      min_confidence: 0.7   # 战略层要求70%+
    "15m":
      min_confidence: 0.6   # 战术层要求60%+
    "1m":
      min_confidence: 0.5   # 执行层要求50%+

# 激进配置：降低要求
three_tier:
  layer_roles:
    "4h":
      min_confidence: 0.4
    "15m":
      min_confidence: 0.3
    "1m":
      min_confidence: 0.2
```

## 🚀 下一步

1. ✅ 配置文件已更新
2. ✅ 核心模块已实现
3. ✅ 策略集成已完成
4. ⏳ 需要测试验证
5. ⏳ 需要调整信号处理逻辑

---

**版本**: v3.0 - Three-Tier Hierarchical Architecture  
**日期**: 2025-10-19  
**状态**: 核心实现完成，待测试验证

