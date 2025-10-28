# 三层架构诊断报告

## 🔍 问题现象

一周回测（6.1M ticks）：
- ✅ 成功完成（5分9秒）
- ❌ 0个交易
- ❌ 战略层一直"4H数据不足"
- ❌ 战术层一直"无SR区间"

## 🐛 根本原因

### 问题1: 战略层数据一直为0

**日志分析**:
```
❌ 战略层不通过(0.00<0.4): 4H数据不足
```

**可能原因**:
1. `make_strategic_decision()`中检查`if bars_4h.empty or len(bars_4h) < 20`
2. 传入的是`self.bars_data.get(self.strategic_tf, pd.DataFrame())`
3. 但战略层bars可能没有正确累积

**检查点**:
```python
# strategy.py line ~494
strategic_decision = self.three_tier.make_strategic_decision(
    bars_4h=self.bars_data.get(self.strategic_tf, pd.DataFrame())
)
```

### 问题2: 战术层SR区间为空

**日志分析**:
```
❌ 战术层不通过(0.00<0.3): 无SR区间
```

**原因**:
```python
# line ~500
sr_zones = []
for tf, conf, state, trigger, sr, feats in local_scores:
    if tf == self.tactical_tf:  # 'tactical'
        sr_zones.append({...})
```

但`local_scores`中的`tf`是什么？检查`_process_signals`：
```python
for tf in self.timeframes:  # ['execution', 'tactical', 'strategic']
    srs = self.models[tf].detect_sr_levels(bars)
    for sr in srs:
        local_scores.append((tf, conf, state, trigger, sr, feats))
```

**应该没问题**！除非... `self.models[tf]` 初始化有问题？

## 🔬 深度诊断

### 检查1: DynamicSRModel初始化

```python
# strategy.py line ~85
self.models = {
    'execution': DynamicSRModel('execution', self.cfg),
    'tactical': DynamicSRModel('tactical', self.cfg),
    'strategic': DynamicSRModel('strategic', self.cfg'),
}
```

**问题**: `DynamicSRModel('execution', ...)` 
- DynamicSRModel的第一个参数是`tf: str`
- 它内部可能用这个字符串做某些判断或日志
- **但是'execution'不是有效的timeframe！**

应该是：
```python
self.models = {
    'execution': DynamicSRModel('5m', self.cfg),    # 传实际周期
    'tactical': DynamicSRModel('30m', self.cfg),
    'strategic': DynamicSRModel('4h', self.cfg'),
}
```

## ✅ 解决方案

### 方案A: 分离层级名和实际周期（推荐）

```python
# config.yaml
bar_types:
  "execution": "BTCUSDT.BINANCE-5-MINUTE-LAST-INTERNAL"

timeframe_mapping:  # 新增！
  "execution": "5m"
  "tactical": "30m"
  "strategic": "4h"

# strategy.py
self.tf_to_period = {
    'execution': '5m',
    'tactical': '30m',
    'strategic': '4h'
}

self.models = {
    'execution': DynamicSRModel(self.tf_to_period['execution'], self.cfg),
    'tactical': DynamicSRModel(self.tf_to_period['tactical'], self.cfg),
    'strategic': DynamicSRModel(self.tf_to_period['strategic'], self.cfg'),
}
```

### 方案B: 修改DynamicSRModel接受任意字符串

### 方案C: 直接用周期字符串（回退方案）

```python
# 简单粗暴：不用层级名了
self.timeframes = ['5m', '30m', '4h']
self.execution_tf = '5m'
```

## 🎯 立即行动

实施方案A：
1. 添加timeframe_mapping配置
2. 在strategy.py中建立映射
3. 初始化models时使用实际周期
4. 其他地方继续用层级名

这样既保持了层级名的优雅，又确保models正确初始化。

---

**诊断完成，等待修复！**

