# 命令对比：factor-test vs factor-compute vs timeframe-forward-report vs dim-compare

## 快速对比表

| 命令 | 主要用途 | 输出 | 使用场景 | 复杂度 |
|------|---------|------|---------|--------|
| `factor-test` | 测试因子有效性 | IC/IR 报告（HTML/TXT/JSON） | 研究阶段：评估因子质量 | ⭐ 低 |
| `factor-compute` | 计算因子值 | CSV/Parquet 文件 | 实盘：计算指定因子 | ⭐ 低 |
| `timeframe-forward-report` | 分析时间框架和预测周期 | 相关性报告 | 策略设计：选择最优时间框架 | ⭐⭐ 中 |
| `dim-compare` | 特征选择和降维对比 | 模型性能报告、top_factors.json | 模型优化：特征选择 | ⭐⭐⭐ 高 |

---

## 1. `factor-test` - 因子有效性测试

### 功能
测试单个或多个因子的预测能力，计算 IC（Information Coefficient）、IR（Information Ratio）等指标。

### 使用场景
- ✅ **研究阶段**：评估新因子是否有效
- ✅ **因子筛选**：比较多个因子的表现
- ✅ **快速验证**：快速检查因子的 IC/IR 值

### 示例
```bash
make factor-test \
  FACTOR_TEST_FACTORS="price_to_zz_high_pct,price_to_poc_pct" \
  FACTOR_TEST_SYMBOL=BTCUSDT,ETHUSDT \
  FACTOR_TEST_FEATURE_TYPE=baseline \
  FACTOR_TEST_START_DATE=2024-01-01 \
  FACTOR_TEST_END_DATE=2024-12-31
```

### 输出
- `factor_test_report.html` - 可视化 HTML 报告（包含图表）
- `factor_test_report.txt` - 文本报告
- `factor_test_results.json` - JSON 数据

### 特点
- ⚡ **快速**：只计算需要的因子，不做模型训练
- 📊 **可视化**：生成包含图表的 HTML 报告
- 🎯 **专注**：只关注因子的预测能力（IC/IR）

---

## 2. `factor-compute` - 因子值计算

### 功能
计算指定因子的数值，输出到文件，用于实盘或后续分析。

### 使用场景
- ✅ **实盘部署**：只计算需要的因子，节省计算资源
- ✅ **批量计算**：为多个时间点计算因子值
- ✅ **数据导出**：将因子值导出为 CSV/Parquet 格式

### 示例
```bash
make factor-compute \
  FACTOR_COMPUTE_FACTORS="rsi_7 macd" \
  FACTOR_COMPUTE_SYMBOL=BTCUSDT \
  FACTOR_COMPUTE_START_DATE=2024-01-01 \
  FACTOR_COMPUTE_END_DATE=2024-12-31 \
  FACTOR_COMPUTE_OUTPUT=results/factors/computed_factors.csv
```

### 输出
- CSV 或 Parquet 文件，包含因子值

### 特点
- 🚀 **高效**：只计算指定的因子，不计算其他
- 💾 **灵活**：支持多种输出格式
- 🔧 **实用**：专为实盘场景设计

---

## 3. `timeframe-forward-report` - 时间框架和预测周期分析

### 功能
分析不同时间框架（5T, 15T, 60T等）和预测周期（forward bars）的相关性，找出最优组合。

### 使用场景
- ✅ **策略设计**：选择最适合的时间框架
- ✅ **参数优化**：确定最优的预测周期
- ✅ **多时间框架分析**：比较不同时间框架的表现

### 示例
```bash
make timeframe-forward-report \
  SYMBOLS=BTCUSDT,ETHUSDT \
  TF_ANALYSIS_TIMEFRAMES="5T,15T,60T,240T" \
  TF_ANALYSIS_FORWARD_BARS="1,3,6,12,24" \
  TF_ANALYSIS_FEATURE_TYPE=baseline
```

### 输出
- `timeframe_forward_details.csv` - 详细的相关性数据
- `timeframe_forward_summary.csv` - 汇总报告
- 策略配置文件（如果指定了 `TF_ANALYSIS_RUN_TAG`）

### 特点
- 🔍 **全面**：测试多个时间框架和预测周期的组合
- 📈 **相关性分析**：使用 Pearson 和 Spearman 相关系数
- 🎯 **策略导向**：帮助选择最优的时间框架配置

---

## 4. `dim-compare` - 特征选择和降维对比

### 功能
对比不同特征集（不同因子数量）的模型性能，进行特征选择和降维。

### 使用场景
- ✅ **特征选择**：找出最重要的因子
- ✅ **降维优化**：减少特征数量同时保持性能
- ✅ **模型优化**：提升模型性能和稳定性

### 示例
```bash
make dim-compare \
  SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT \
  HORIZONS=1,5,10,15 \
  DIM_COMPARE_FEATURE_TYPE=baseline \
  TIMEFRAME=5T \
  FACTOR_COUNTS=20,40,60,80,100
```

### 输出
- `top_factors.json` - 选出的最优因子列表
- HTML 报告 - 详细的性能对比报告
- 模型文件 - 训练好的模型
- SHAP 分析结果（如果启用）

### 特点
- 🎓 **完整流程**：包含数据加载、特征工程、模型训练、评估
- 📊 **性能对比**：对比不同特征集的模型性能
- 🔬 **深度分析**：包含 SHAP 分析、回测等
- ⏱️ **耗时较长**：需要训练多个模型

---

## 使用流程建议

### 阶段 1：因子研究
```bash
# 1. 测试新因子的有效性
make factor-test FACTOR_TEST_FACTORS="new_factor_1,new_factor_2"

# 2. 如果因子有效，继续下一步
```

### 阶段 2：策略设计
```bash
# 1. 分析最优时间框架和预测周期
make timeframe-forward-report \
  SYMBOLS=BTCUSDT \
  TF_ANALYSIS_TIMEFRAMES="5T,15T,60T,240T"

# 2. 根据报告选择最优配置
```

### 阶段 3：特征选择
```bash
# 1. 对比不同特征集，找出最优因子组合
make dim-compare \
  SYMBOLS=BTCUSDT,ETHUSDT \
  DIM_COMPARE_FEATURE_TYPE=comprehensive \
  FACTOR_COUNTS=20,40,60,80,100

# 2. 使用 top_factors.json 中的因子进行训练
```

### 阶段 4：实盘部署
```bash
# 1. 只计算需要的因子（从 top_factors.json）
make factor-compute \
  FACTOR_COMPUTE_FACTORS="factor1 factor2 factor3" \
  FACTOR_COMPUTE_OUTPUT=realtime_factors.csv
```

---

## 关键区别总结

### `factor-test` vs `factor-compute`
- **factor-test**：评估因子质量（IC/IR），不输出因子值
- **factor-compute**：计算因子值，不评估质量

### `timeframe-forward-report` vs `dim-compare`
- **timeframe-forward-report**：分析时间框架和预测周期，不训练模型
- **dim-compare**：训练模型并对比性能，包含完整的机器学习流程

### 选择建议
- 🧪 **快速测试因子** → `factor-test`
- 💻 **实盘计算因子** → `factor-compute`
- ⏰ **选择时间框架** → `timeframe-forward-report`
- 🎯 **优化特征集** → `dim-compare`

---

## 参数对比

### factor-test
- `FACTOR_TEST_FACTORS` - 要测试的因子（必需）
- `FACTOR_TEST_SYMBOL` - 交易对
- `FACTOR_TEST_START_DATE` - 开始日期
- `FACTOR_TEST_END_DATE` - 结束日期
- `FACTOR_TEST_FEATURE_TYPE` - 特征类型（baseline/comprehensive等）

### factor-compute
- `FACTOR_COMPUTE_FACTORS` - 要计算的因子（必需）
- `FACTOR_COMPUTE_INPUT` - 输入文件（可选）
- `FACTOR_COMPUTE_SYMBOL` - 交易对（如果从数据目录加载）
- `FACTOR_COMPUTE_OUTPUT` - 输出文件（必需）

### timeframe-forward-report
- `SYMBOLS` - 交易对列表
- `TF_ANALYSIS_TIMEFRAMES` - 时间框架列表（如 "5T,15T,60T"）
- `TF_ANALYSIS_FORWARD_BARS` - 预测周期列表（如 "1,3,6,12"）
- `TF_ANALYSIS_FEATURE_TYPE` - 特征类型

### dim-compare
- `SYMBOLS` - 交易对列表
- `DIM_COMPARE_FEATURE_TYPE` - 特征类型
- `TIMEFRAME` - 时间框架
- `FACTOR_COUNTS` - 因子数量列表（如 "20,40,60,80"）
- `HORIZONS` - 预测周期列表
- `TIME_WINDOWS` - 时间窗口（训练时间段）

