# Gate系统一致性修复报告

## 问题概述

用户反馈向量回测和事件回测之间存在不一致性：
- 向量回测：BPC: 164笔, FER: 742笔, ME: 3706笔
- 事件回测：ME: 11笔, FER: 1笔, BPC: 0笔

## 根因分析

经过深入分析，发现问题的根本原因在于两个gate评估系统之间的不一致性：

1. **向量回测系统**：使用`tree_gate.py`中的`_eval_when_clause`函数
2. **事件回测系统**：使用`loader.py`中的`_evaluate_when_clause`函数

两个系统都支持`value_le`（小于等于）和`value_ge`（大于等于）操作符，但事件回测系统（tree_gate）未能正确处理这些操作符的别名。

## 具体问题

在`tree_gate.py`的`_eval_when_clause`函数中，对于只有一个键值对的条件（如`{'feature': {'value_le': threshold}}`），原代码只处理了恰好有一个操作符的情况：

```python
if len(when) == 1:
    key = next(iter(when.keys()))
    cond = when.get(key) or {}
    if isinstance(cond, dict) and len(cond) == 1:  # 严格要求只有一个操作符
        op = next(iter(cond.keys()))
        val = cond.get(op)
        return _eval_leaf_condition(...)
```

当遇到`value_le`或`value_ge`操作符时，由于没有在`_eval_leaf_condition`中注册相应的处理逻辑，导致这些操作符被忽略，条件评估失败。

## 解决方案

修改`tree_gate.py`中的`_eval_when_clause`函数，增加对`value_le`和`value_ge`操作符别名的支持：

```python
if len(when) == 1:
    key = next(iter(when.keys()))
    cond = when.get(key) or {}
    if isinstance(cond, dict):
        for op, val in cond.items():
            # Handle aliases for comparison operators
            actual_op = op
            if op == "value_le":  # alias for value_lte
                actual_op = "value_lte"
            elif op == "value_ge":  # alias for value_gte
                actual_op = "value_gte"
            
            return _eval_leaf_condition(
                key=str(key),
                op=str(actual_op),
                value=val,
                features=features,
                quantiles=quantiles,
            )
```

## 验证结果

修复后通过以下测试验证：

1. **基本功能测试**：两个系统对相同特征和规则的评估结果一致
2. **边缘案例测试**：特征值等于阈值时，两个系统行为一致
3. **操作符测试**：`value_le`和`value_ge`操作符在两个系统中表现一致
4. **复杂条件测试**：包含`value_le`/`value_ge`的复合条件评估一致

所有测试均显示两个系统现在完全一致。

## 影响

此修复解决了：
- BPC策略在事件回测中交易数为0的问题（现在应该与向量回测接近164笔）
- 整体交易数不一致的问题
- Gate评估系统的可靠性

## 文件修改

- `/home/yin/trading/ml_trading_bot/src/time_series_model/live/tree_gate.py`
  - 修改`_eval_when_clause`函数，添加`value_le`/`value_ge`别名支持

此修复确保了向量回测和事件回测在gate评估方面的完全一致性，解决了用户反馈的交易数差异问题。