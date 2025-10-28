# 三层架构快速开始指南

## 🎯 核心概念

```
4H(战略层) → 定方向  → "能否交易？"
15m(战术层) → 定结构  → "在哪儿交易？"
1m(执行层)  → 定入场  → "何时交易？"
```

## ✅ 已实现的功能

### 1. 配置文件 (`config.yaml`)
- ✅ 三层时间周期：1m/15m/4h
- ✅ 各层职责定义
- ✅ 分层注意力权重
- ✅ 最小置信度要求

### 2. 核心模块 (`three_tier_layer.py`)
- ✅ 战略层决策：MarketState + TrendBias + CVD
- ✅ 战术层决策：Dynamic SR + Volume Profile
- ✅ 执行层决策：K线模式 + 量能 + 动量
- ✅ 三层融合决策：requires_all_layers

### 3. 策略集成 (`strategy.py`)
- ✅ 时间周期更新：['1m', '15m', '4h']
- ✅ 三层模型初始化
- ✅ ThreeTierLayer导入和实例化

## ⚠️ 当前状态

### 已完成
1. ✅ 配置层：三层结构定义
2. ✅ 核心层：ThreeTierLayer实现
3. ✅ 集成层：strategy.py基础集成

### 待完成
1. ⏳ **信号处理逻辑调整** - 关键！
   - 当前：`_process_signals()` 使用旧的多周期融合
   - 需要：改用 `three_tier.fuse_three_tiers()`
   
2. ⏳ **开仓执行限制** - 关键！
   - 当前：可能在任意周期开仓
   - 需要：强制只在1m执行开仓
   
3. ⏳ **数据准备**
   - 需要：1m级别数据（tick可以聚合）
   - 需要：4h级别数据（从1m聚合）

## 🔧 快速修复计划

### Step 1: 检查Linter错误
```bash
cd /home/yin/trading/rlbot
python -m pylint nautilus_project/src/yin_bot/dynamic_sr/three_tier_layer.py
```

### Step 2: 测试三层模块
```python
# test_three_tier.py
from three_tier_layer import ThreeTierLayer

config = {"three_tier": {...}}
tt = ThreeTierLayer(config)

# 测试战略层
strategic = tt.make_strategic_decision(
    market_state="accumulation",
    trend_bias=0.6,
    cvd_direction=0.5,
    bars_4h=df_4h
)
print(f"战略层: {strategic.confidence:.2f}, {strategic.direction}")

# 测试战术层
tactical = tt.make_tactical_decision(
    strategic_direction="long",
    sr_zones=[...],
    volume_profile={...},
    bars_15m=df_15m,
    current_price=60000
)
print(f"战术层: {tactical.confidence:.2f}, {tactical.price_zone}")

# 测试执行层
execution = tt.make_execution_decision(
    strategic_direction="long",
    tactical_zone=(60200, 60400),
    bars_1m=df_1m,
    current_price=60300
)
print(f"执行层: {execution.confidence:.2f}, {execution.trigger_signal}")

# 融合决策
decision = tt.fuse_three_tiers(strategic, tactical, execution)
print(f"最终决策: {decision.should_trade}, {decision.reason}")
```

### Step 3: 简单回测
```bash
# 使用小数据集测试
make backtest-dynamic-sr-btc 2>&1 | grep -E "(战略层|战术层|执行层|三层)"
```

## 📋 关键代码位置

### 需要修改的地方

1. **`strategy.py` Line ~470-500**: 信号处理逻辑
```python
# 当前（需要改）
decision = self.confluence_layer.fuse(signals, ...)

# 改为（三层）
strategic = self.three_tier.make_strategic_decision(...)
tactical = self.three_tier.make_tactical_decision(...)
execution = self.three_tier.make_execution_decision(...)
decision = self.three_tier.fuse_three_tiers(strategic, tactical, execution)
```

2. **`strategy.py` Line ~630-700**: 开仓执行
```python
# 添加检查
if tf != self.execution_tf:  # 不是1m
    self.log.info(f"⏭️ {tf}层决策已更新，等待1m执行")
    return  # 不执行开仓

# 只有1m才执行
if tf == self.execution_tf:
    self._execute_signal(signal, decision, tf)
```

## 🚦 测试检查清单

### 初始化测试
- [ ] 配置加载正确
- [ ] 三层模型初始化成功
- [ ] timeframes = ['1m', '15m', '4h']

### 单层测试
- [ ] 战略层(4h): 能输出方向和置信度
- [ ] 战术层(15m): 能识别SR和价值区
- [ ] 执行层(1m): 能检测K线模式和量能

### 融合测试
- [ ] 三层都通过 → should_trade = True
- [ ] 任一层不通过 → should_trade = False
- [ ] 综合置信度计算正确

### 执行测试
- [ ] 只在1m上开仓
- [ ] 使用战略层的方向
- [ ] 参考战术层的价格区间

## 🎯 预期效果

### 修复前（旧架构）
```
5m/15m/1h 多周期融合 → 信号混乱
每个周期都能开仓 → 过度交易
加仓失控 → 巨额亏损
```

### 修复后（三层架构）
```
4h定方向 → 趋势明确
15m定结构 → SR清晰
1m定入场 → 时机精确
只在1m开仓 → 执行统一
```

## 📊 日志示例

```
✅ 三层架构初始化: 4h(方向) → 15m(结构) → 1m(入场)

[14:00:00] 4H Bar → 战略层决策
  ✅ 战略层(0.75): 多头趋势(0.60), CVD流入(0.50)
  
[14:15:00] 15m Bar → 战术层决策
  ✅ 战术层(0.68): support@60,200, 强度0.82
  
[14:15:30] 1m Bar → 执行层决策
  ✅ 执行层(0.54): engulfing, 量增2.3x
  
[14:15:30] 融合决策
  ✅ 战略层(0.75): ... | ✅ 战术层(0.68): ... | ✅ 执行层(0.54): ...
  → 应该交易: True, 方向: long, 入场: 60,250
  
[14:15:30] 1m执行层 → 开仓做多 @ 60,250
```

## 🔥 立即行动

1. **检查语法**: 运行linter确保无错误
2. **单元测试**: 测试三层独立功能
3. **小规模回测**: 用1小时数据测试
4. **完整回测**: 用1天数据验证效果

---

**准备就绪！现在可以开始测试了。** 🚀

