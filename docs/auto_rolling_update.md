# 自动滚动更新文档 (Auto Rolling Update)

## 概述

`make auto-rolling-update` 是一个**自动化滚动训练和更新**命令，它会：

1. **自动检测所有可用的数据文件**（跨多个年份，从最早到最新）
2. **训练初始模型**（使用前 N 个月的数据）
3. **滚动更新到最新数据**（逐月扩展训练窗口，测试下一个月）
4. **生成详细报告**（包含所有月份的指标对比）

## 核心特性

✅ **自动数据发现**: 自动查找所有可用的月度数据文件，无需手动指定年份范围  
✅ **跨年度支持**: 可以跨多个年份进行滚动训练（如 2023→2024→2025）  
✅ **增量更新**: 支持从上次训练的位置继续更新（`--update-only`）  
✅ **详细报告**: 自动生成 HTML 报告，包含所有训练结果  

## 使用方法

### 基本用法

```bash
# 自动检测所有数据，从最早的6个月开始训练，滚动到最新月份
make auto-rolling-update SYMBOL=BTCUSDT

# 指定初始训练月数
make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=6

# 指定最小训练月数要求（至少需要3个月才能训练）
make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=6 MIN_TRAIN_MONTHS=3
```

### 高级用法

```bash
# 启用订单流特征
make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=6 ADD_ORDER_FLOW=1

# 指定输出目录名称
make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=6 OUTPUT=my_rolling_update

# 仅更新（从上次训练位置继续）
make auto-rolling-update-only SYMBOL=BTCUSDT OUTPUT=results/auto_rolling_btcusdt_20241102_123456
```

## 工作流程

### 自动检测数据文件

脚本会自动查找所有匹配的数据文件：
- `{SYMBOL}-aggTrades-YYYY-MM.parquet`
- `{SYMBOL}-aggTrades-YYYY-MM.zip`
- `{SYMBOL}_YYYY-MM.parquet`
- 等等...

并按时间顺序排序（从最早到最新）。

### 滚动训练流程

假设找到 2024-01 到 2024-12 的数据，初始训练月数为 6：

1. **第1次迭代**：
   - 训练：2024-01 到 2024-06（前6个月）
   - 测试：2024-07（第7个月）

2. **第2次迭代**：
   - 训练：2024-01 到 2024-07（前7个月，扩展窗口）
   - 测试：2024-08（第8个月）

3. **第3次迭代**：
   - 训练：2024-01 到 2024-08（前8个月）
   - 测试：2024-09（第9个月）

4. **... 以此类推**

5. **最后一次迭代**：
   - 训练：2024-01 到 2024-11（前11个月）
   - 测试：2024-12（第12个月）

### 跨年度示例

假设数据从 2023-06 到 2024-12：

1. **第1次迭代**：训练 2023-06~2023-11，测试 2023-12
2. **第2次迭代**：训练 2023-06~2023-12，测试 2024-01
3. **第3次迭代**：训练 2023-06~2024-01，测试 2024-02
4. **... 以此类推**
5. **最后一次**：训练 2023-06~2024-11，测试 2024-12

## 输出文件

运行后会生成以下文件：

```
results/auto_rolling_{symbol}_{timestamp}/
├── monthly_results.csv              # 所有月份的详细结果（CSV）
├── summary.json                      # 汇总信息（JSON）
├── monthly_rolling_report.html       # HTML 可视化报告
└── model_YYYY-MM.txt                 # 每个月的模型文件（LightGBM格式）
```

### CSV 文件结构

`monthly_results.csv` 包含以下列：
- `test_month`: 测试月份（如 "2024-07"）
- `total_trades`: 总交易数
- `total_return`: 总回报率（%）
- `win_rate`: 胜率（%）
- `profit_factor`: 盈亏比
- `max_drawdown`: 最大回撤（%）
- `train_months`: 训练使用的月数
- `train_samples`: 训练样本数
- `test_samples`: 测试样本数
- `num_features`: 特征数量
- `train_start`: 训练开始月份
- `train_end`: 训练结束月份

### JSON 文件结构

`summary.json` 包含：
- `symbol`: 交易符号
- `total_months_tested`: 测试的总月数
- `earliest_month`: 最早的数据月份
- `latest_month`: 最新的数据月份
- `last_trained_month`: 最后训练的月份
- `avg_return`, `avg_win_rate`, `avg_profit_factor`, `avg_max_drawdown`: 平均指标
- `total_trades`: 总交易数
- `configuration`: 运行配置
- `created_at`: 创建时间

## 增量更新（Update Only）

如果你已经运行过完整的滚动训练，可以使用 `--update-only` 模式只更新新的月份：

```bash
# 第一次运行（完整训练）
make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=6 OUTPUT=my_rolling_update

# 后续更新（只训练新月份）
make auto-rolling-update-only SYMBOL=BTCUSDT OUTPUT=results/my_rolling_update
```

### 工作原理

1. 从 `summary.json` 读取 `last_trained_month`
2. 只训练从 `last_trained_month + 1` 到最新月份的月份
3. 将新结果追加到现有的 `monthly_results.csv`
4. 更新 `summary.json`

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SYMBOL` | BTCUSDT | 交易符号 |
| `INITIAL_TRAIN_MONTHS` | 6 | 初始训练月数（前N个月用于第一次训练） |
| `MIN_TRAIN_MONTHS` | 3 | 最小训练月数（至少需要N个月才能训练） |
| `OUTPUT` | auto | 输出目录名称（可选，默认自动生成） |
| `ADD_ORDER_FLOW` | 0 | 是否启用订单流特征（1=启用，0=禁用） |
| `UPDATE_ONLY` | 0 | 是否仅更新模式（1=仅更新，0=完整训练） |

## 与现有命令的区别

### `make train` vs `make auto-rolling-update`

| 特性 | `make train` | `make auto-rolling-update` |
|------|-------------|---------------------------|
| 数据范围 | 手动指定 `START_DATE` 和 `END_DATE` | 自动检测所有可用数据 |
| 训练方式 | 单次训练 | 滚动训练（多次迭代） |
| 输出 | 单个模型 | 多个模型（每月一个） |
| 用途 | 训练单个时间段的模型 | 评估模型在多个时间段的表现 |

### `make rolling-monthly` vs `make auto-rolling-update`

| 特性 | `make rolling-monthly` | `make auto-rolling-update` |
|------|----------------------|---------------------------|
| 数据范围 | 手动指定 `YEAR`（单一年份） | 自动检测所有年份 |
| 跨年度 | ❌ 不支持 | ✅ 支持 |
| 数据发现 | 需指定年份 | 自动发现 |
| 使用场景 | 评估单一年份 | 评估从历史到现在的表现 |

## 典型使用场景

### 场景 1: 评估模型表现（历史到最新）

```bash
# 使用所有可用数据，从2023年开始训练，滚动到2024年最新月份
make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=6
```

**输出**: 展示模型从历史到现在每个月的表现，评估模型的稳定性和适应性。

### 场景 2: 定期更新模型

```bash
# 第一次：完整训练
make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=6 OUTPUT=btc_rolling_production

# 每周/每月更新一次（只训练新月份）
make auto-rolling-update-only SYMBOL=BTCUSDT OUTPUT=results/btc_rolling_production
```

**输出**: 保持模型始终基于最新数据训练，同时保留历史评估记录。

### 场景 3: 对比不同训练窗口大小

```bash
# 测试初始训练6个月的效果
make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=6 OUTPUT=btc_6months

# 测试初始训练12个月的效果
make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=12 OUTPUT=btc_12months
```

**输出**: 对比不同初始训练窗口大小对模型性能的影响。

## 报告解读

### HTML 报告包含

1. **训练摘要**: 总月数、总交易数、特征工程方法
2. **性能摘要**: 平均回报、胜率、盈亏比、最大回撤的统计（均值、标准差、最小值、最大值）
3. **按月详细结果**: 每个月份的详细指标
4. **配置信息**: 运行参数

### 关键指标

- **Total Return (%)**: 月度总回报率，越高越好
- **Win Rate (%)**: 胜率，>50% 为良好
- **Profit Factor**: 盈亏比，>1 为盈利
- **Max Drawdown (%)**: 最大回撤，越小越好
- **Train Samples**: 训练样本数，越多越好（但要注意过拟合）

## 注意事项

⚠️ **数据要求**:
- 确保 `data/parquet_data/` 目录中有足够的数据文件
- 文件命名需要匹配模式：`{SYMBOL}-aggTrades-YYYY-MM.parquet` 或类似格式

⚠️ **训练时间**:
- 滚动训练会训练多个模型，耗时较长
- 建议在服务器上运行，或在后台运行

⚠️ **增量更新**:
- 使用 `--update-only` 时，需要指定正确的 `OUTPUT` 目录
- 确保 `summary.json` 和 `monthly_results.csv` 存在且完整

⚠️ **最小训练月数**:
- 默认至少需要 3 个月的数据才能开始训练
- 如果某个月份的训练数据不足，会跳过该月份的测试

## 示例输出

```bash
$ make auto-rolling-update SYMBOL=BTCUSDT INITIAL_TRAIN_MONTHS=6

🔍 Finding all available data files...
   Found 24 months of data:
   Earliest: 2023-01
   Latest: 2024-12

🔄 Starting Auto Rolling Update

[1/18] 2023-07
Train: 2023-01 to 2023-06 (6 months)
Test:  2023-07
...

📊 SUMMARY
Month       Trades   Return     Win%     PF      MaxDD    
--------------------------------------------------------------------------------
2023-07     145      2.45%      52.3%    1.15    3.2%
2023-08     132      -1.23%     48.1%    0.95    4.5%
...

💾 Results saved to: results/auto_rolling_btcusdt_20241102_123456/
   - monthly_results.csv
   - summary.json
   - model_*.txt (one per month)
   - monthly_rolling_report.html
```

## 故障排除

### 问题 1: 找不到数据文件

**错误**: `❌ No data files found for BTCUSDT in data/parquet_data!`

**解决**:
- 检查数据文件是否存在
- 检查文件命名格式是否正确
- 运行 `make data-pipeline` 下载和转换数据

### 问题 2: 训练数据不足

**错误**: `⚠️  Skipping 2024-07: insufficient training data (2 < 3)`

**解决**:
- 下载更多历史数据
- 降低 `MIN_TRAIN_MONTHS` 参数（不推荐）
- 从有足够数据的月份开始

### 问题 3: 增量更新找不到上次训练位置

**错误**: `⚠️  Failed to resume from last trained month`

**解决**:
- 检查 `summary.json` 是否存在
- 确保 `OUTPUT` 参数指向正确的目录
- 或者重新运行完整训练

## 最佳实践

1. **首次运行**: 使用 `INITIAL_TRAIN_MONTHS=6` 或 `12`，确保有足够的初始训练数据
2. **定期更新**: 使用 `--update-only` 模式，每周或每月更新一次
3. **保存输出目录**: 记录 `OUTPUT` 目录名称，用于后续更新
4. **监控报告**: 定期查看 HTML 报告，评估模型性能趋势

## 相关命令

- `make train`: 训练单个时间段的模型
- `make rolling-monthly`: 单一年份内的滚动训练
- `make rolling-quarterly`: 季度滚动训练
- `make data-pipeline`: 下载和转换数据

