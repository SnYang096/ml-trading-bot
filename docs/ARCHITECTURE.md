# 系统架构文档

## 目录

1. [系统概述](#系统概述)
2. [架构层次](#架构层次)
3. [核心模块](#核心模块)
4. [数据流](#数据流)
5. [特征工程](#特征工程)
6. [模型训练](#模型训练)
7. [策略执行](#策略执行)

---

## 系统概述

ML Trading Bot 是一个基于机器学习的量化交易系统，采用**事件驱动架构**和**配置驱动设计**，支持多策略、多资产、多时间框架的量化研究和实盘交易。

### 核心设计原则

1. **配置驱动**：策略配置、特征配置、模型配置均通过 YAML 文件管理
2. **事件驱动**：使用事件驱动的架构处理市场数据和交易执行
3. **模块化**：特征工程、模型训练、策略执行相互解耦
4. **可扩展**：易于添加新特征、新策略、新模型

---

## 架构层次

```
┌─────────────────────────────────────────────────────────┐
│                   策略层 (Strategy Layer)                │
│  - sr_reversal, sr_breakout, compression_breakout, etc  │
│  - 策略特定配置 (YAML)                                   │
│  - 策略特定回测逻辑                                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                   模型层 (Model Layer)                   │
│  - XGBoost, LightGBM, CatBoost                          │
│  - 滚动训练 (Rolling Training)                          │
│  - 模型评估和选择                                         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                  特征层 (Feature Layer)                  │
│  - 特征计算引擎 (Feature Computer)                      │
│  - 特征注册表 (Feature Registry)                        │
│  - 特征依赖管理 (Feature Dependencies)                  │
│  - 特征缓存 (Feature Cache)                             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                   数据层 (Data Layer)                    │
│  - 市场数据加载 (Market Data Loader)                    │
│  - Tick 数据处理 (Tick Data Processing)                 │
│  - 数据缓存 (Data Cache)                                │
└─────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. 特征工程模块

**位置**: `src/features/`

**核心组件**:

- **Feature Registry** (`registry.py`): 特征注册和发现
- **Feature Computer** (`loader/parallel_computer.py`): 并行特征计算
- **Feature Dependencies** (`config/feature_dependencies.yaml`): 特征依赖配置
- **Feature Modules** (`time_series/`): 各种特征实现

**特征分类**:

1. **Baseline Features** (`baseline_features.py`): 基础技术指标 (ATR, RSI, MACD, etc.)
2. **Advanced Features**:
   - WPT Features (`utils_wpt_features.py`): 小波包变换特征
   - VPIN Features (`utils_order_flow_features.py`): 订单流特征
   - DTW Features (`utils_dtw_features.py`): 动态时间规整特征
   - Spectrum Features (`utils_spectrum_features.py`): 频谱特征
   - GARCH/EVT Features (`utils_garch_features.py`, `utils_evt_features.py`): 波动率模型特征
   - Hilbert Features (`utils_hilbert_features.py`): Hilbert 变换特征
   - Hurst Features (`utils_hurst_features.py`): Hurst 指数特征
   - Liquidity Features (`utils_liquidity_features.py`): 流动性特征
   - Footprint Features (`utils_footprint.py`): Footprint 特征

**特征计算流程**:

```
原始数据 → 特征注册表 → 依赖解析 → 并行计算 → 特征缓存 → 输出
```

### 2. 数据加载模块

**位置**: `src/data_tools/`

**核心组件**:

- **MarketDataLoader** (`data_loader.py`): 市场数据加载
- **TickLoader** (`tick_loader.py`): Tick 数据处理和缓存
- **DataCache** (`cache/`): 数据缓存管理

**支持的数据源**:

- OHLCV K线数据 (Parquet 格式)
- Tick 数据 (Parquet 格式)
- 多时间框架支持 (1m, 5m, 15m, 1h, 4h, 1d)

### 3. 模型训练模块

**位置**: `src/time_series_model/`

**核心组件**:

- **Strategy Trainer** (`training/strategy_trainer.py`): 策略训练器
- **Rolling Trainer** (`training/rolling_trainer.py`): 滚动训练
- **Model Evaluator** (`diagnostics/`): 模型评估工具

**支持的模型**:

- XGBoost (回归/分类)
- LightGBM (回归/分类)
- CatBoost (回归/多分类)

### 4. 策略执行模块

**位置**: `src/time_series_model/backtesting/`, `src/time_series_model/live/`

**核心组件**:

- **EventDrivenStrategy** (`backtesting/event_driven_strategy.py`): 事件驱动策略基类
- **Strategy Backtests** (`backtesting/strategy_backtests.py`): 策略特定回测逻辑
- **Live Trading** (`live/`): 实盘交易模块

**策略类型**:

1. **SR Reversal**: 支撑阻力反转策略
2. **SR Breakout**: 支撑阻力突破策略
3. **Compression Breakout**: 压缩突破策略
4. **Trend Following**: 趋势跟随策略

---

## 数据流

### 训练数据流

```
1. 加载市场数据 (MarketDataLoader)
   ↓
2. 计算特征 (Feature Computer)
   ↓
3. 生成标签 (Label Generator)
   ↓
4. 数据预处理 (Normalization, Feature Selection)
   ↓
5. 模型训练 (Strategy Trainer)
   ↓
6. 模型评估 (Model Evaluator)
   ↓
7. 模型保存 (Model Storage)
```

### 预测数据流

```
1. 加载最新市场数据
   ↓
2. 增量特征计算 (Incremental Feature Computer)
   ↓
3. 特征预处理
   ↓
4. 模型预测 (Model Predictor)
   ↓
5. 信号生成 (Signal Generator)
   ↓
6. 交易执行 (Trade Executor)
```

### 回测数据流

```
1. 加载历史数据
   ↓
2. 计算特征 (并行/缓存)
   ↓
3. 模型预测
   ↓
4. 策略逻辑执行 (止损/止盈/仓位管理)
   ↓
5. 绩效统计 (Performance Metrics)
   ↓
6. 报告生成 (Report Generator)
```

---

## 特征工程

### 特征计算架构

**Narrow-IO 设计模式**:

特征函数只接收必要的 Series/DataFrame，而不是整个宽表，提高性能和可测试性。

```python
# Narrow-IO 示例
@register_feature("compute_atr_from_series", category="baseline")
def compute_atr_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """只接收需要的列，返回 DataFrame"""
    ...
```

### 特征依赖管理

特征依赖通过 `feature_dependencies.yaml` 管理：

```yaml
feature_name:
  dependencies: [dep1, dep2]  # 依赖的其他特征
  required_columns: [col1, col2]  # 需要的原始数据列
  output_columns: [out1, out2]  # 输出的列
  compute_params: {...}  # 计算参数
```

### 特征缓存

- **月度缓存**: 特征按月缓存，提高重复计算效率
- **并行计算**: 多进程并行计算不同月份的特征
- **缓存版本控制**: 支持特征代码变更后的缓存失效

---

## 模型训练

### 滚动训练 (Rolling Training)

使用扩展窗口 (Expanding Window) 按月滚动训练：

```
第1次: 训练=[1-6月], 测试=7月 → 模型1
第2次: 训练=[1-7月], 测试=8月 → 模型2
第3次: 训练=[1-8月], 测试=9月 → 模型3
...
```

### 特征筛选与优化流程

特征筛选是一个多阶段的系统性过程，不仅包括自动化的降维，还包括人工评估和对比验证：

#### 阶段 1: 单因子评估 (Factor Evaluation)

使用 `ts-factor-eval` 评估所有特征的 IC/IR：

```bash
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_SYMBOL=BTCUSDT \
  TS_FACTOR_TIMEFRAME=240T \
  TS_FACTOR_START=2024-01-01 \
  TS_FACTOR_END=2025-10-31
```

**评估指标**:
- IC Mean: 信息系数均值
- IC IR: 信息比率
- Win Rate: 胜率
- IC Decay: IC 衰减分析

**筛选标准**: 移除 IC/IR 明显不行的特征（如 IC IR < 0.3）

**输出**: `results/factor_ts_eval/ts_eval_*.html` - 详细的因子评估报告

#### 阶段 2: 特征配置对比 (Feature Ablation Study)

使用 `ts-strategy-feature-compare` 对比不同特征配置的回测表现：

```bash
make ts-strategy-feature-compare \
  STRAT_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  STRAT_COMPARE_OVERRIDES="baseline=config/features/baseline.yaml full=config/features/full.yaml"
```

**对比内容**:
- 不同特征组合的回测性能
- 特征组的重要性评估
- 性能-复杂度权衡

**输出**: `results/strategy_compare/strategy_feature_compare_summary.csv` - 对比结果

#### 阶段 3: 规则 vs ML 验证 (Model Comparison)

使用 `ts-sr-reversal-model-comparison` 验证机器学习模型是否优于规则策略：

```bash
make ts-sr-reversal-model-comparison \
  SR_COMP_SYMBOL=BTCUSDT \
  SR_COMP_TIMEFRAME=240T
```

**对比模型**:
- Rule-based: 纯规则策略（baseline）
- ML: 机器学习模型
- ML+Volatility: 机器学习 + 波动率模型

**验证目标**: 确认 ML 模型确实带来改进，而不是简单的规则包装

**输出**: `results/model_comparison/{timeframe}/comparison_report.html` - 对比报告

#### 阶段 4: 参数优化 (Parameter Optimization)

##### 4.1 规则参数优化

使用 `ts-sr-reversal-rule-optimization` 找到规则策略的最佳参数：

```bash
make ts-sr-reversal-rule-optimization \
  SR_COMP_SYMBOL=BTCUSDT \
  SR_COMP_TIMEFRAME=240T
```

**优化内容**:
- 规则参数网格搜索
- 参数 plateau 识别
- 参数敏感性分析

**输出**: `results/rule_optimization/optimization_results.csv` + plateau charts

##### 4.2 ML 参数优化

使用 `ts-ml-plateau-charts` 可视化 ML 模型参数的影响：

```bash
make ts-ml-plateau-charts \
  SR_COMP_TIMEFRAME=240T
```

**分析内容**:
- 超参数热力图
- 参数 plateau 区域
- 最优参数组合

**输出**: Parameter plateau charts 添加到对比报告中

#### 阶段 5: 自动化降维 (Dimensionality Reduction) - 可选

如果特征数量仍然过多，可以使用 `ts-dim-compare` 进行自动化降维：

**三阶段流程**:

1. **Stage 1**: Missing/stability filter
   - 移除缺失率 > 20% 的特征
   - 移除低方差特征

2. **Stage 2**: IC ranking
   - 按信息系数 (IC) 排序
   - 选择 top K 特征

3. **Stage 3**: Correlation-based selection
   - 去除高相关性冗余特征
   - 保留最具代表性的特征

**注意**: 自动化降维通常在手动筛选和对比之后使用，作为最终的特征精简步骤。

---

### 三个特征筛选命令的详细对比

为了更好地理解特征筛选流程，这里详细说明三个核心命令的区别和用途：

#### 1. `ts-factor-eval` - 单因子评估

**评估粒度**: **单个特征（因子）级别**

**评估方法**: 直接计算每个特征与未来收益的相关性

**评估指标**:
- IC (Information Coefficient): 信息系数，衡量因子与未来收益的线性相关性
- IC IR (Information Ratio): IC 的均值除以标准差，衡量 IC 的稳定性
- Win Rate: 因子预测正确的比例
- IC Decay: IC 在不同时间滞后期的衰减情况

**评估范围**: 
- 评估所有特征（从策略配置的 `features.yaml` 读取）
- 或指定特定特征列表

**输出**:
- 每个特征的 IC/IR 统计
- IC 时间序列曲线
- 因子相关性矩阵
- HTML 可视化报告

**用途**:
- ✅ **快速筛选**: 识别明显无效的特征（IC IR < 0.3）
- ✅ **特征发现**: 找出表现最好的单个特征
- ✅ **特征诊断**: 检查特征的稳定性和衰减情况

**局限性**:
- ❌ 不考虑特征之间的交互作用
- ❌ 不考虑特征组合后的效果
- ❌ 不涉及模型训练

**示例**:
```bash
# 评估所有特征
make ts-factor-eval TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long

# 评估特定特征
make ts-factor-eval TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long \
  TS_FACTOR_FACTORS="rsi,atr,macd"
```

---

#### 2. `ts-strategy-feature-compare` - 特征配置对比（消融实验）

**评估粒度**: **特征组合（配置）级别**

**评估方法**: 训练完整的模型，对比不同特征配置的整体性能

**评估指标**:
- 模型准确率（训练集/测试集）
- 夏普比率
- 最大回撤
- 交易次数
- 其他回测指标

**评估范围**:
- 对比不同的特征配置（如 baseline vs full）
- 每个配置包含多个特征的组合

**工作流程**:
1. 加载不同的特征配置 YAML
2. 为每个配置训练模型
3. 在测试集上评估性能
4. 对比各配置的表现

**输出**:
- 各配置的性能对比表（CSV）
- 各配置的详细训练结果
- 特征组重要性评估

**用途**:
- ✅ **消融实验**: 评估特征组的贡献（加入/移除某组特征的影响）
- ✅ **特征组合优化**: 找出最优的特征组合
- ✅ **性能验证**: 验证特征选择对模型性能的实际影响

**优势**:
- ✅ 考虑了特征之间的交互作用
- ✅ 反映了真实模型训练场景
- ✅ 可以评估特征组合的整体效果

**局限性**:
- ❌ 计算成本高（需要训练多个模型）
- ❌ 不能直接识别单个特征的好坏

**示例**:
```bash
# 对比 baseline 和 full 特征配置
make ts-strategy-feature-compare \
  STRAT_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  STRAT_COMPARE_OVERRIDES="baseline=config/features/baseline.yaml full=config/features/full.yaml"
```

---

#### 3. `ts-dim-compare` - 自动化降维

**评估粒度**: **特征集合的整体优化**

**评估方法**: 使用统计方法自动筛选特征子集

**筛选流程**:

1. **Stage 1: Missing/Stability Filter**
   - 移除缺失率 > 20% 的特征
   - 移除低方差特征（稳定性差）

2. **Stage 2: IC Ranking**
   - 计算每个特征的 IC
   - 按 IC 排序
   - 选择 top K 特征

3. **Stage 3: Correlation-based Selection**
   - 计算特征之间的相关性矩阵
   - 去除高相关性（> 阈值）的冗余特征
   - 保留最具代表性的特征

**评估指标**:
- 特征的统计属性（缺失率、方差）
- IC 值
- 特征相关性

**输出**:
- `top_factors.json`: 选中的特征列表
- `results.json`: 详细的筛选结果和统计信息

**用途**:
- ✅ **特征数量控制**: 将特征数量从 100+ 减少到 30-50
- ✅ **自动化筛选**: 无需人工干预的批量筛选
- ✅ **特征精简**: 在手动筛选后进一步精简特征

**优势**:
- ✅ 速度快（无需训练模型）
- ✅ 可自动化执行
- ✅ 基于统计方法，客观可靠

**局限性**:
- ❌ 只考虑统计属性，不考虑实际模型性能
- ❌ 可能移除有用的特征（如果 IC 低但与其他特征组合效果好）
- ❌ 不能处理特征交互

**示例**:
```bash
make ts-dim-compare \
  DIM_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  SYMBOL=BTCUSDT \
  START_DATE=2024-01-01 \
  END_DATE=2024-12-31
```

---

### 三个命令的关系和使用场景

#### 它们不是重复的，而是互补的：

1. **`ts-factor-eval`**: 单个特征的快速筛选
   - 适用于：特征数量很多（>50）的初期筛选
   - 目标：快速识别明显无效的特征

2. **`ts-strategy-feature-compare`**: 特征组合的实际效果验证
   - 适用于：已经初步筛选后的特征组合优化
   - 目标：验证特征组合对模型性能的实际影响

3. **`ts-dim-compare`**: 自动化特征精简
   - 适用于：特征数量仍然过多的最终精简
   - 目标：将特征数量控制到合理范围（30-50）

#### 典型使用流程：

```
阶段 1: ts-factor-eval
  ↓ (筛选出 IC IR < 0.3 的特征)
  
阶段 2: ts-strategy-feature-compare
  ↓ (对比不同特征配置，找出最优组合)
  
阶段 3: ts-dim-compare (可选)
  ↓ (如果特征仍然过多，自动化精简)
  
最终特征列表
```

#### 决策树：如何选择使用哪个命令？

```
特征数量 > 50?
├─ 是 → 使用 ts-factor-eval 快速筛选
└─ 否 → 
    需要验证特征组合效果?
    ├─ 是 → 使用 ts-strategy-feature-compare
    └─ 否 → 
         特征数量仍然 > 50?
         ├─ 是 → 使用 ts-dim-compare 自动化精简
         └─ 否 → 直接使用当前特征列表
```

---

### 总结对比表

| 特性             | ts-factor-eval | ts-strategy-feature-compare | ts-dim-compare   |
| ---------------- | -------------- | --------------------------- | ---------------- |
| **评估粒度**     | 单个特征       | 特征组合（配置）            | 特征集合         |
| **评估方法**     | IC/IR 统计     | 模型训练+回测               | 统计筛选         |
| **计算成本**     | 低             | 高（需训练模型）            | 中               |
| **考虑交互**     | ❌              | ✅                           | ❌                |
| **考虑模型性能** | ❌              | ✅                           | ❌                |
| **自动化程度**   | 中             | 低（需配置）                | 高               |
| **主要用途**     | 快速筛选       | 消融实验                    | 特征精简         |
| **典型输入**     | 特征列表       | 特征配置 YAML               | 特征列表         |
| **典型输出**     | IC/IR 报告     | 性能对比表                  | top_factors.json |
| **使用阶段**     | 初期筛选       | 中期优化                    | 后期精简         |

---

### 完整特征筛选工作流示例

以 SR Reversal 策略为例的完整流程：

```bash
# 1. 单因子评估：评估所有特征的 IC/IR
make ts-factor-eval TS_FACTOR_STRATEGY=config/strategies/sr_reversal_long

# 2. 特征配置对比：对比 baseline vs full 特征配置
make ts-strategy-feature-compare \
  STRAT_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  STRAT_COMPARE_OVERRIDES="baseline=config/features/baseline.yaml full=config/features/full.yaml"

# 3. 规则 vs ML 验证：确认 ML 优于规则
make ts-sr-reversal-model-comparison \
  SR_COMP_SYMBOL=BTCUSDT \
  SR_COMP_TIMEFRAME=240T

# 4. 参数优化：找到最佳参数
make ts-sr-reversal-rule-optimization SR_COMP_TIMEFRAME=240T
make ts-ml-plateau-charts SR_COMP_TIMEFRAME=240T

# 5. (可选) 自动化降维：最终特征精简
make ts-dim-compare \
  DIM_COMPARE_CONFIG=config/strategies/sr_reversal_long \
  SYMBOL=BTCUSDT
```

---

## 策略执行

### 事件驱动架构

策略基于事件驱动执行：

```python
# 事件类型
- BarEvent: K线更新事件
- SignalEvent: 信号生成事件
- OrderEvent: 订单事件
- FillEvent: 成交事件
```

### 策略生命周期

```
初始化 → 数据订阅 → 事件处理循环 → 信号生成 → 订单执行 → 持仓管理 → 清算
```

### 风险控制

- **止损**: ATR-based 止损
- **止盈**: 目标价位止盈
- **仓位管理**: 基于置信度和风险的动态仓位
- **最大持仓周期**: 防止长时间持仓

---

## 配置文件结构

### 策略配置

```
config/strategies/{strategy_name}/
  ├── features.yaml          # 特征列表
  ├── model.yaml            # 模型配置
  ├── labels.yaml           # 标签配置
  ├── backtest.yaml         # 回测配置
  └── meta.yaml             # 策略元数据
```

### 特征配置

```
config/feature_dependencies.yaml  # 所有特征的依赖和配置
```

---

## 技术栈

- **语言**: Python 3.12+
- **数据处理**: Pandas, NumPy
- **机器学习**: XGBoost, LightGBM, CatBoost
- **信号处理**: PyWavelets (WPT), SciPy (信号处理)
- **回测引擎**: VectorBot, Nautilus Trader
- **容器化**: Docker
- **配置管理**: YAML

---

## 扩展指南

### 添加新特征

1. 在 `src/features/time_series/` 中实现特征函数
2. 使用 `@register_feature` 装饰器注册
3. 在 `feature_dependencies.yaml` 中配置依赖和参数
4. 编写测试用例

### 添加新策略

1. 创建策略配置目录 `config/strategies/{strategy_name}/`
2. 定义特征列表、模型配置、标签配置
3. 实现策略特定的回测逻辑（如需要）
4. 编写策略文档

### 添加新模型

1. 实现模型接口
2. 集成到 `StrategyTrainer`
3. 添加模型配置选项

---

## 相关文档

- [研发流程指南](DEVELOPMENT_WORKFLOW.md)
- [上线流程指南](DEPLOYMENT_WORKFLOW.md)
- [特征使用指南](features/)
- [策略配置指南](strategies/)

