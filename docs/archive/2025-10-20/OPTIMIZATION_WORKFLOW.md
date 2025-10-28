# 优化工作流程

## 📋 已完成任务

### ✅ 1. VectorBT Makefile集成
```bash
make vectorbt-test          # 单日快速测试
make vectorbt-test-week     # 一周测试
make vectorbt-test-month    # 一个月测试
```

### ✅ 2. 信号诊断工具
```bash
make diagnose-signals       # 分析为何交易信号少
```

功能：
- 三层决策逐层诊断
- SR级别检测统计
- 置信度分布分析
- 瓶颈识别与参数建议

### ✅ 3. 参数优化工具
```bash
make optimize-params        # 网格搜索最佳参数
make analyze-param PARAM=xxx VALUES=a,b,c  # 单参数敏感性分析
```

功能：
- VectorBT加速网格搜索
- Top 10参数组合输出
- 结果保存到CSV

## 🔄 进行中

### ⏳ Nautilus月度回测
**状态**: 运行中（已11分钟）
**进程**: PID 50620, CPU 100%, MEM 32.3%
**预计**: 还需5-10分钟

**目标**: 验证VectorBT的+10.33%收益结果

## 📊 VectorBT测试结果

### 一个月数据 (2025-05)
```
数据量: 12,528,655 ticks
聚合时间: 49.6秒
回测时间: 3.3秒
总耗时: 53秒

结果:
- Total Return: +10.33%
- Sharpe Ratio: 1.12
- Max Drawdown: -5.67%
- Win Rate: 66.67%
- Total Trades: 1 (仅1笔交易!)
```

### 🔴 关键问题
**只有1笔交易！**

可能原因：
1. 三层决策条件过于严格
2. 置信度阈值过高
3. SR过滤过于严格
4. 数据不足（4H只有186 bars）

## 📈 下一步行动

### 1. 等待Nautilus回测完成
- 对比VectorBT和Nautilus结果
- 验证交易逻辑一致性

### 2. 运行信号诊断
```bash
make diagnose-signals
```

识别瓶颈：
- 战略层通过率
- 战术层SR数量
- 执行层入场频率

### 3. 参数优化
基于诊断结果，调优：
```bash
# 示例：降低置信度阈值
make analyze-param \
  PARAM=three_tier.execution_min_confidence \
  VALUES=0.1,0.2,0.3,0.4,0.5
```

关键参数：
- `three_tier.strategic_min_confidence`: 当前0.4 → 尝试0.2-0.5
- `three_tier.tactical_min_confidence`: 当前0.3 → 尝试0.1-0.4
- `three_tier.execution_min_confidence`: 当前0.3 → 尝试0.1-0.4
- `sr_model.min_strength`: 当前0.5 → 尝试0.3-0.7
- `risk_management.stop_loss_pct`: 当前0.02 → 尝试0.015-0.03

### 4. 快速迭代
使用VectorBT快速验证调整效果：
1. 修改config.yaml
2. `make vectorbt-test-month` (53秒)
3. 观察交易数量和收益
4. 重复直至满意
5. 最后用Nautilus详细验证

## 💡 优化策略

### 放宽条件 vs 保持严格
**当前问题**: 信号过少（只有1笔交易）

**方案A - 渐进放宽**:
```yaml
# 第一步：降低10%
three_tier:
  layer_roles:
    strategic:
      min_confidence: 0.36  # 从0.4
    tactical:
      min_confidence: 0.27  # 从0.3
    execution:
      min_confidence: 0.27  # 从0.3
```

**方案B - 大幅放宽**:
```yaml
# 更激进，目标：产生10-20笔交易
three_tier:
  layer_roles:
    strategic:
      min_confidence: 0.25
    tactical:
      min_confidence: 0.20
    execution:
      min_confidence: 0.20

sr_model:
  min_strength: 0.35  # 从0.5
```

**方案C - 调整权重**:
保持置信度阈值，调整注意力权重：
```yaml
three_tier:
  attention_weights:
    strategic: 0.5  # 从0.4 (提高战略层影响)
    tactical: 0.3   # 从0.35
    execution: 0.2  # 从0.25
```

## 🎯 目标

1. **短期目标**: 产生5-10笔交易/月
   - 验证三层决策逻辑正确性
   - 观察胜率和收益分布

2. **中期目标**: 优化至15-20笔交易/月
   - Sharpe > 1.5
   - Win Rate > 55%
   - Max DD < -10%

3. **长期目标**: 稳定盈利
   - 月收益 > 5%
   - 年化Sharpe > 2.0
   - 最大回撤 < -15%

## 📝 实验记录

### Experiment 1: Baseline (当前配置)
- **数据**: 2025-05 (1个月)
- **交易数**: 1
- **收益**: +10.33%
- **结论**: 信号过少，需要放宽条件

### Experiment 2: TBD
(待运行诊断工具后决定)

---

**更新时间**: 2025-10-19 18:23
**Nautilus回测状态**: 运行中 (11分钟)

