# 研发流程指南

本文档描述从特征开发到模型训练的完整研发流程。

## 目录

1. [开发环境准备](#开发环境准备)
2. [特征开发流程](#特征开发流程)
3. [特征测试流程](#特征测试流程)
4. [因子评估流程](#因子评估流程)
5. [特征选择流程](#特征选择流程)
6. [模型训练流程](#模型训练流程)
7. [模型评估流程](#模型评估流程)

---

## 开发环境准备

### 1. 环境设置

```bash
# 创建虚拟环境
conda create -n ml_trading python=3.12
conda activate ml_trading

# 安装依赖
pip install -e .[dev]

# 安装 Git hooks（可选）
make install-hooks
```

### 2. Docker 环境（可选）

```bash
# 构建 Docker 镜像
make docker-build

# 运行 Docker 容器
make docker-run
```

---

## 特征开发流程

### 1. 实现特征函数

在 `src/features/time_series/` 中实现特征函数：

```python
from src.features.registry import register_feature

@register_feature("compute_my_feature_from_series", category="my_category")
def compute_my_feature_from_series(
    *,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    """Narrow-IO 特征函数
    
    Args:
        close: 收盘价序列
        volume: 成交量序列
        period: 计算周期
    
    Returns:
        DataFrame with output columns
    """
    # 实现特征逻辑
    result = ...
    return pd.DataFrame({"my_feature": result})
```

### 2. 注册特征配置

在 `config/feature_dependencies.yaml` 中添加配置：

```yaml
my_feature_f:
  module: time_series
  compute_func: compute_my_feature_from_series
  dependencies: []  # 依赖的其他特征
  required_columns:
    - close
    - volume
  output_columns:
    - my_feature
  category: my_category
  description: 我的特征描述
  compute_params:
    period: 20
  pass_full_df: false
  column_mappings:
    close: close
    volume: volume
```

### 3. 编写测试

在 `tests/features/` 中编写测试：

```python
def test_my_feature():
    # 创建测试数据
    df = create_test_data()
    
    # 测试特征计算
    result = compute_my_feature_from_series(...)
    
    # 验证结果
    assert "my_feature" in result.columns
    assert len(result) == len(df)
    ...
```

### 4. 运行测试

```bash
# 运行单个测试
python -m pytest tests/features/test_my_feature.py -v

# 运行所有特征测试
make test-features
```

---

## 特征测试流程

### 测试类型

1. **单元测试**: 测试特征函数的正确性
2. **集成测试**: 测试特征与其他特征的交互
3. **性能测试**: 测试特征计算性能
4. **无未来泄漏测试**: 确保特征不包含未来信息

### 关键测试

```bash
# 运行所有特征测试
python -m pytest tests/features/ -v

# 运行特定类型的测试
python -m pytest tests/features/ -k "liquidity" -v
python -m pytest tests/features/ -k "future_leak" -v
```

### 测试覆盖率

```bash
# 生成覆盖率报告
python -m pytest tests/features/ --cov=src/features --cov-report=html
```

---

## 特征筛选与优化流程（完整）

特征筛选是一个系统性的多阶段过程，需要结合人工评估和自动化工具：

### 阶段 1: 单因子评估 (Factor Evaluation)

**目标**: 评估所有特征的 IC/IR，筛选出明显无效的特征

```bash
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_SYMBOL=BTCUSDT \
  TS_FACTOR_TIMEFRAME=240T \
  TS_FACTOR_START=2024-01-01 \
  TS_FACTOR_END=2025-10-31
```

**评估指标**:
- **IC Mean**: 信息系数均值（> 0.01 较好）
- **IC IR**: 信息比率（> 0.5 较好）
- **Win Rate**: 胜率（> 0.5 较好）
- **IC Decay**: IC 衰减分析（稳定性）

**筛选标准**:
- 移除 IC IR < 0.3 的特征（明显无效）
- 移除 IC Mean 接近 0 的特征
- 移除与其他特征高度相关的冗余特征

**输出**:
- `results/factor_ts_eval/ts_eval_*.html` - 详细的因子评估报告
- 包含每个因子的 IC 曲线、胜率统计、相关性矩阵

---

### 阶段 2: 特征配置对比 (Feature Ablation Study)

**目标**: 对比不同特征配置的回测表现，评估特征组的重要性

```bash
make ts-strategy-feature-compare \
  STRAT_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  STRAT_COMPARE_OVERRIDES="baseline=config/features/baseline.yaml full=config/features/full.yaml advanced=config/features/advanced.yaml"
```

**对比内容**:
- 不同特征组合的训练集/测试集性能
- 特征组对最终性能的贡献度
- 性能 vs 复杂度权衡

**分析重点**:
- 哪些特征组带来显著提升？
- 哪些特征组可以移除而不影响性能？
- 最优的特征组合是什么？

**输出**:
- `results/strategy_compare/strategy_feature_compare_summary.csv` - 对比结果汇总
- 包含各配置的准确率、夏普比率、最大回撤等指标

---

### 阶段 3: 规则 vs ML 验证 (Model Comparison)

**目标**: 验证机器学习模型确实优于规则策略，而不是简单的规则包装

```bash
make ts-sr-reversal-model-comparison \
  SR_COMP_SYMBOL=BTCUSDT \
  SR_COMP_TIMEFRAME=240T \
  SR_COMP_START=2024-01-01 \
  SR_COMP_END=2025-10-31
```

**对比模型**:
1. **Rule-based**: 纯规则策略（baseline）
2. **ML**: 机器学习模型（XGBoost/LightGBM）
3. **ML+Volatility**: ML 模型 + 波动率模型

**验证要点**:
- ML 模型是否显著优于规则？
- ML 模型的收益是否稳定？
- ML 模型的交易次数是否合理？

**输出**:
- `results/model_comparison/{timeframe}/comparison_report.html` - 详细对比报告
- 包含各模型的回测曲线、性能指标、交易统计

---

### 阶段 4: 参数优化 (Parameter Optimization)

#### 4.1 规则参数优化

**目标**: 找到规则策略的最佳参数组合

```bash
make ts-sr-reversal-rule-optimization \
  SR_COMP_SYMBOL=BTCUSDT \
  SR_COMP_TIMEFRAME=240T
```

**优化方法**:
- 网格搜索或随机搜索
- 参数 plateau 识别（参数敏感性分析）
- 最优参数组合选择

**输出**:
- `results/rule_optimization/optimization_results.csv` - 参数优化结果
- `results/rule_optimization/optimization_report.html` - 包含 plateau charts

#### 4.2 ML 参数优化

**目标**: 可视化 ML 模型超参数的影响，找到最优参数

```bash
# 首先运行参数扫描
make ts-sr-reversal-model-comparison SR_COMP_TIMEFRAME=240T

# 然后生成参数热力图
make ts-ml-plateau-charts SR_COMP_TIMEFRAME=240T
```

**分析内容**:
- 超参数热力图（显示参数对性能的影响）
- Parameter plateau 区域识别
- 最优参数组合推荐

**输出**:
- Parameter plateau charts 添加到 `comparison_report.html`
- 帮助识别参数的最优范围和敏感区域

---

### 阶段 5: 自动化降维 (Dimensionality Reduction) - 可选

**目标**: 如果特征数量仍然过多，使用自动化方法进一步精简

```bash
make ts-dim-compare \
  DIM_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  SYMBOL=BTCUSDT \
  START_DATE=2024-01-01 \
  END_DATE=2024-12-31 \
  ENCODING_DIM=32
```

**三阶段流程**:

1. **Stage 1: Missing/Stability Filter**
   - 移除缺失率 > 20% 的特征
   - 移除低方差特征

2. **Stage 2: IC Ranking**
   - 按信息系数排序
   - 选择 top K 特征

3. **Stage 3: Correlation-based Selection**
   - 去除高相关性冗余特征
   - 保留最具代表性的特征

**注意**: 这一步通常在手动筛选之后使用，作为最终的特征精简。

**输出**:
- `results/dim_compare/{strategy}_{symbol}_{timestamp}/top_factors.json` - 选中的特征列表
- `results/dim_compare/{strategy}_{symbol}_{timestamp}/results.json` - 详细结果

---

## 完整特征筛选工作流示例

以 SR Reversal 策略为例：

```bash
# ============================================
# 阶段 1: 单因子评估
# ============================================
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_SYMBOL=BTCUSDT \
  TS_FACTOR_TIMEFRAME=240T \
  TS_FACTOR_START=2024-01-01 \
  TS_FACTOR_END=2025-10-31

# 查看报告，筛选出 IC IR < 0.3 的特征，从 features_all.yaml 中移除

# ============================================
# 阶段 2: 特征配置对比（消融实验）
# ============================================
make ts-strategy-feature-compare \
  STRAT_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  STRAT_COMPARE_OVERRIDES="baseline=config/features/baseline.yaml full=config/strategies/sr_reversal_long/features_all.yaml"

# 分析对比结果，确定最优特征组合

# ============================================
# 阶段 3: 规则 vs ML 验证
# ============================================
make ts-sr-reversal-model-comparison \
  SR_COMP_SYMBOL=BTCUSDT \
  SR_COMP_TIMEFRAME=240T \
  SR_COMP_START=2024-01-01 \
  SR_COMP_END=2025-10-31

# 验证 ML 模型确实优于规则策略

# ============================================
# 阶段 4: 参数优化
# ============================================
# 4.1 规则参数优化
make ts-sr-reversal-rule-optimization \
  SR_COMP_SYMBOL=BTCUSDT \
  SR_COMP_TIMEFRAME=240T

# 4.2 ML 参数优化
make ts-ml-plateau-charts SR_COMP_TIMEFRAME=240T

# 根据 plateau charts 调整模型超参数

# ============================================
# 阶段 5: (可选) 自动化降维
# ============================================
# 如果特征数量仍然过多，使用自动化降维
make ts-dim-compare \
  DIM_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  SYMBOL=BTCUSDT \
  START_DATE=2024-01-01 \
  END_DATE=2024-12-31

# 使用输出的 top_factors.json 作为最终特征列表
```

---

## 特征筛选最佳实践

### 1. 迭代式筛选

- ✅ 不要一次性筛选所有特征
- ✅ 分阶段逐步筛选和验证
- ✅ 每次筛选后重新评估整体性能

### 2. 多角度验证

- ✅ IC/IR 评估（单因子层面）
- ✅ 回测性能对比（组合层面）
- ✅ 规则 vs ML 对比（方法层面）

### 3. 参数优化

- ✅ 先优化规则参数（如果使用规则 baseline）
- ✅ 再优化 ML 超参数
- ✅ 使用 plateau charts 识别参数敏感区域

### 4. 避免过拟合

- ✅ 使用独立的验证集
- ✅ 关注测试集性能，而非训练集
- ✅ 考虑特征数量的复杂度惩罚

---

## 常见问题

### Q: 如何判断特征是否有效？

A: 综合多个指标：
- IC IR > 0.3（单因子层面）
- 包含该特征的配置回测性能更好（组合层面）
- 特征重要性在模型中较高

### Q: 何时使用自动化降维？

A: 
- 特征数量 > 100 时考虑
- 手动筛选后仍需要进一步精简
- 作为最终的特征数量控制手段

### Q: 如何平衡特征数量和性能？

A:
- 使用特征配置对比找出最优平衡点
- 关注性能提升 vs 复杂度增加的比例
- 考虑实盘计算成本

---

## 模型训练流程

### 1. 快速验证（单次训练）

使用固定时间范围训练一个模型，用于快速验证：

```bash
make train-strategy \
  STRATEGY_CONFIG=config/strategies/sr_reversal_long \
  SYMBOL=BTCUSDT \
  TIMEFRAME=240T \
  START_DATE=2024-01-01 \
  END_DATE=2024-10-31 \
  TEST_SIZE=0.15
```

**输出**:
- `results/strategies/{strategy_name}/model.pkl` - 训练好的模型
- `results/strategies/{strategy_name}/results.json` - 训练结果

### 2. 生产训练（滚动训练）

使用扩展窗口按月滚动训练，模拟真实生产环境：

```bash
make rolling \
  ROLLING_CONFIG=config/strategies/sr_reversal_long \
  SYMBOL=BTCUSDT \
  TIMEFRAME=240T \
  INITIAL_TRAIN_MONTHS=6 \
  MIN_TRAIN_MONTHS=3 \
  ROLLING_START=2024-01 \
  ROLLING_END=2024-10
```

**工作流程**:
```
第1次: 训练=[2024-01 到 2024-06], 测试=2024-07 → 模型1
第2次: 训练=[2024-01 到 2024-07], 测试=2024-08 → 模型2
第3次: 训练=[2024-01 到 2024-08], 测试=2024-09 → 模型3
...
```

**输出**:
- `results/rolling/{strategy_name}/{month}/model.pkl` - 每月模型
- `results/rolling/{strategy_name}/monthly_results.json` - 汇总结果

### 3. 使用特征选择结果

如果进行了特征选择，可以在训练时使用选中的特征：

```bash
make rolling \
  ROLLING_CONFIG=config/strategies/sr_reversal_long \
  SYMBOL=BTCUSDT \
  USE_TOP_FACTORS=results/dim_compare/.../top_factors.json \
  ...
```

---

## 模型评估流程

### 1. 查看训练结果

```bash
# 查看单次训练结果
cat results/strategies/{strategy_name}/results.json

# 查看滚动训练汇总结果
cat results/rolling/{strategy_name}/monthly_results.json
```

### 2. 回测验证

```bash
make ts-vectorbot-backtest \
  BACKTEST_MODEL=results/rolling/{strategy_name}/latest \
  BACKTEST_START=2024-01-01 \
  BACKTEST_END=2024-10-31
```

### 3. 消融实验（可选）

对比不同特征配置的性能：

```bash
make ts-strategy-feature-compare \
  STRAT_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  STRAT_COMPARE_SYMBOL=BTCUSDT \
  STRAT_COMPARE_OVERRIDES="baseline=config/features/baseline.yaml full=config/features/full.yaml"
```

---

## 最佳实践

### 1. 特征开发

- ✅ 使用 Narrow-IO 设计模式
- ✅ 编写完整的单元测试
- ✅ 确保无未来泄漏
- ✅ 文档化特征含义和参数

### 2. 因子评估

- ✅ 评估多个时间框架
- ✅ 评估多个资产
- ✅ 关注 IC 和胜率的稳定性
- ✅ 检查与其他因子的相关性

### 3. 模型训练

- ✅ 使用滚动训练模拟生产环境
- ✅ 使用特征选择避免过拟合
- ✅ 监控训练集和测试集性能
- ✅ 保存完整的训练日志

### 4. 代码质量

- ✅ 运行测试确保代码正确性
- ✅ 运行 lint 检查代码风格
- ✅ 提交前运行完整测试套件
- ✅ 编写清晰的提交信息

---

## 常见问题

### Q: 如何添加新特征？

A: 参见 [特征开发流程](#特征开发流程)

### Q: 特征计算太慢怎么办？

A: 
1. 检查是否使用了缓存
2. 考虑并行计算
3. 优化特征算法

### Q: 模型过拟合怎么办？

A:
1. 使用特征选择减少特征数量
2. 增加训练数据量
3. 调整模型正则化参数

### Q: 如何选择特征？

A: 使用 `ts-dim-compare` 进行自动特征选择

---

## 相关文档

- [系统架构文档](ARCHITECTURE.md)
- [上线流程指南](DEPLOYMENT_WORKFLOW.md)
- [特征使用指南](features/)

