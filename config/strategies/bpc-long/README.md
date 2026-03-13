# BPC (BreakoutPullbackContinuation) 策略

## 策略概述

BreakoutPullbackContinuation archetype：趋势中回踩后延续原方向。

**核心理念**：
- Breakout: 突破压缩区/前高，放量+CVD同向
- Pullback: 回踩但不破结构，缩量+CVD吸收
- Continuation: 重新启动，再次放量+CVD恢复

## 训练配置

- **数据集**: highcap6 (BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT)
- **时间窗口**: 2023-01-01 至 2024-12-31
- **标签类型**: forward_rr (outcome audit标签，全量数据)
- **模型类型**: LightGBM 回归
- **样本数**: 2132 (训练集)
- **特征数**: 143

## 模型性能

- **交叉验证**: 5-fold时序交叉验证
- **平均CV指标**: -0.0720
- **训练完成时间**: 2026-01-30

## 特征分层

### Atomic 层（原子信号）
- **breakout**: 价格突破强度、成交量确认、CVD确认、VPIN确认
- **pullback**: 回踩深度、回踩质量、缩量确认、CVD吸收
- **continuation**: 恢复强度、动量确认、放量确认、CVD动量、VPIN上升
- **neutral**: 波动率压缩、成交量压缩

### Composite 层（组合信号）
- bpc_score_breakout: 突破综合分
- bpc_score_pullback: 回踩综合分
- bpc_score_continuation: 续行综合分
- bpc_score_neutral: 中性综合分

### Contextual 层（状态信号）
- bpc_breakout_direction: 突破方向
- bpc_direction_confidence: 方向置信度
- bpc_is_after_breakout: 是否在突破后
- bpc_was_in_pullback: 是否经历过回踩

---

---

## 📜 树模型规则导出（固定训练 LightGBM）
