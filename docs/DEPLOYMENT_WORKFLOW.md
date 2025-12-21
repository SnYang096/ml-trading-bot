# 上线流程指南

本文档描述从模型训练完成到生产环境部署的完整上线流程。

## 目录

1. [上线前检查清单](#上线前检查清单)
2. [模型验证](#模型验证)
3. [回测验证](#回测验证)
4. [性能测试](#性能测试)
5. [生产部署](#生产部署)
6. [监控和维护](#监控和维护)

---

## 上线前检查清单

### 1. 代码质量检查

```bash
# 运行所有测试
make test

# 运行代码检查
make lint

# 检查代码格式
make format
```

**检查项**:
- ✅ 所有测试通过
- ✅ 无 lint 错误
- ✅ 代码格式符合规范

### 2. 模型质量检查

**检查项**:
- ✅ 训练集和测试集性能差异合理（无过拟合）
- ✅ 滚动训练各月份性能稳定
- ✅ 特征重要性合理
- ✅ 模型预测分布正常

### 3. 数据质量检查

**检查项**:
- ✅ 特征计算无 NaN/Inf
- ✅ 特征分布正常
- ✅ 数据时间范围完整
- ✅ 数据缓存可用

---

## 模型验证

### 1. 查看训练结果

```bash
# 查看滚动训练汇总结果
cat results/rolling/{strategy_name}/monthly_results.json
```

**关注指标**:
- 各月份测试集性能
- 性能稳定性（方差）
- 特征重要性一致性

### 2. 模型诊断

```bash
# 运行模型诊断
python -m src.time_series_model.diagnostics.model_diagnostics \
  --model results/rolling/{strategy_name}/latest/model.pkl \
  --data-path data/parquet_data \
  --symbol BTCUSDT \
  --timeframe 240T
```

**诊断内容**:
- 特征重要性分析
- 预测分布分析
- 残差分析

---

## 回测验证

### 1. 单模型回测

```bash
make ts-vectorbot-backtest \
  BACKTEST_MODEL=results/rolling/{strategy_name}/latest \
  BACKTEST_SYMBOL=BTCUSDT \
  BACKTEST_START=2024-01-01 \
  BACKTEST_END=2024-10-31 \
  BACKTEST_TIMEFRAME=240T
```

**关注指标**:
- 总收益率
- 夏普比率
- 最大回撤
- 胜率
- 平均持仓时间

### 2. 滚动模型回测

对每个滚动窗口的模型进行回测，验证模型序列的整体表现：

```bash
# 需要编写脚本遍历 monthly_results.json 中的模型
python scripts/backtest_rolling_models.py \
  --strategy {strategy_name} \
  --symbol BTCUSDT \
  --start-date 2024-01-01 \
  --end-date 2024-10-31
```

### 3. 风险分析

**检查项**:
- ✅ 最大回撤在可接受范围
- ✅ 单笔亏损不超过阈值
- ✅ 连续亏损次数合理
- ✅ 仓位暴露合理

---

## 性能测试

### 1. 特征计算性能

```bash
# 测试特征计算速度
python -m src.features.loader.performance_test \
  --strategy-config config/strategies/{strategy_name} \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-10-31
```

**性能要求**:
- 单次特征计算 < 10s（月度数据）
- 增量特征计算 < 1s（单条数据）

### 2. 模型预测性能

```bash
# 测试模型预测速度
python -m src.time_series_model.diagnostics.prediction_performance_test \
  --model results/rolling/{strategy_name}/latest/model.pkl \
  --n-samples 1000
```

**性能要求**:
- 单次预测 < 10ms
- 批量预测（1000条）< 100ms

### 3. 内存使用

监控特征计算和模型预测的内存使用情况：

```bash
# 使用 memory_profiler
python -m memory_profiler scripts/test_memory_usage.py
```

---

## 生产部署

### 1. 模型打包

```bash
# 创建模型包
python scripts/package_model.py \
  --model results/rolling/{strategy_name}/latest \
  --output models/{strategy_name}_{version}.tar.gz
```

**模型包内容**:
- 模型文件 (model.pkl)
- 特征配置 (feature_dependencies.yaml)
- 策略配置 (strategy config)
- 元数据 (metadata.json)

### 2. 环境配置

**生产环境要求**:
- Python 3.12+
- 所需依赖包
- 数据访问权限
- 模型文件存储

### 3. 部署脚本

```bash
# 部署模型到生产环境
python scripts/deploy_model.py \
  --model-package models/{strategy_name}_{version}.tar.gz \
  --environment production
```

### 4. 配置更新

更新生产环境配置文件：
- 模型路径
- 数据路径
- API 密钥
- 监控配置

---

## 监控和维护

### 1. 性能监控

**监控指标**:
- 模型预测准确性
- 特征计算延迟
- 系统资源使用
- 交易执行成功率

### 2. 模型重训练

**重训练触发条件**:
- 模型性能下降（IC < 阈值）
- 定期重训练（每月/每季度）
- 市场环境变化

**重训练流程**:
```bash
# 1. 收集最新数据
# 2. 运行滚动训练
make rolling \
  ROLLING_CONFIG=config/strategies/{strategy_name} \
  SYMBOL=BTCUSDT \
  ...

# 3. 验证新模型
# 4. 部署新模型
```

### 3. 回滚机制

如果新模型性能下降，需要快速回滚：

```bash
# 回滚到上一个版本
python scripts/rollback_model.py \
  --strategy {strategy_name} \
  --previous-version {previous_version}
```

### 4. 日志和报警

**日志记录**:
- 模型预测日志
- 特征计算日志
- 交易执行日志
- 错误日志

**报警设置**:
- 模型性能异常
- 系统错误
- 数据异常
- 交易失败

---

## 上线检查表

### 代码层面
- [ ] 所有测试通过
- [ ] 代码审查完成
- [ ] 文档更新完成

### 模型层面
- [ ] 模型训练完成且性能满足要求
- [ ] 模型验证通过
- [ ] 特征重要性合理

### 数据层面
- [ ] 数据质量检查通过
- [ ] 特征计算无异常
- [ ] 数据缓存可用

### 回测层面
- [ ] 单模型回测通过
- [ ] 滚动模型回测通过
- [ ] 风险指标在可接受范围

### 性能层面
- [ ] 特征计算性能满足要求
- [ ] 模型预测性能满足要求
- [ ] 内存使用合理

### 部署层面
- [ ] 模型打包完成
- [ ] 环境配置正确
- [ ] 部署脚本测试通过

### 监控层面
- [ ] 监控系统配置完成
- [ ] 报警规则设置完成
- [ ] 日志系统配置完成

---

## 常见问题

### Q: 如何判断模型是否过拟合？

A: 
- 训练集和测试集性能差异大
- 滚动训练各月份性能不稳定
- 特征重要性变化大

### Q: 模型性能下降怎么办？

A:
1. 检查数据质量
2. 检查特征计算是否正确
3. 考虑重训练模型
4. 评估是否需要更新特征配置

### Q: 如何选择部署版本？

A:
- 选择滚动训练中性能稳定的月份
- 避免选择性能异常的月份
- 考虑模型的新旧程度（不要太旧）

---

## 相关文档

- [系统架构文档](ARCHITECTURE.md)
- [研发流程指南](DEVELOPMENT_WORKFLOW.md)
- [策略配置指南](strategies/)

