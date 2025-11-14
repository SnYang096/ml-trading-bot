# `make train` vs `make rolling` 区别详解

## 核心区别总结

| 特性 | `make train` | `make rolling` |
|------|-------------|----------------|
| **训练方式** | 单次训练，一个模型 | 滚动训练，多个模型 |
| **时间窗口** | 固定时间范围（START_DATE → END_DATE） | 扩展窗口（Expanding Window） |
| **模型数量** | **1个模型** | **N个模型**（每个测试月份一个） |
| **用途** | 一次性评估、快速验证 | 生产环境、稳定性评估、回测 |
| **输出** | `models/trained_model_*.pkl` | `results/rolling_*/monthly_results.csv` |
| **推荐场景** | 开发测试、参数调优 | **生产训练、回测** |

## 详细对比

### 1. `make train` - 单次训练

**特点**：
- 使用**固定时间范围**训练**一个模型**
- 适合快速验证、参数调优、一次性评估

**工作流程**：
```
输入: START_DATE=2024-01-01, END_DATE=2024-12-31
      ↓
训练数据: 2024-01-01 到 2024-12-31 的所有数据
      ↓
输出: 1个模型 (trained_model_*.pkl)
```

**示例**：
```bash
make train SYMBOLS=BTCUSDT,ETHUSDT \
  START_DATE=2024-01-01 END_DATE=2024-12-31 \
  TRAIN_FEATURE_TYPE=comprehensive \
  TRAIN_USE_TOP_FACTORS=results/dim_compare/.../top_factors.json
```

**输出**：
- `models/trained_model_*.pkl` - 单个模型文件
- `models/trained_model_*_scalers.pkl` - 特征缩放器
- `models/trained_model_*_info.json` - 模型元数据
- `models/trained_model_*_info_report.html` - HTML 报告

### 2. `make rolling` - 滚动训练

**特点**：
- 使用**扩展窗口**（Expanding Window）训练**多个模型**
- 每个测试月份使用之前所有月份的数据训练一个新模型
- 适合生产环境、评估模型稳定性、真实回测场景

**工作流程**：
```
输入: ROLLING_START=2024-01, ROLLING_END=2024-12
      INITIAL_TRAIN_MONTHS=6
      ↓
第1次: 训练数据=[2024-01, 2024-06], 测试=2024-07 → 模型1
第2次: 训练数据=[2024-01, 2024-07], 测试=2024-08 → 模型2
第3次: 训练数据=[2024-01, 2024-08], 测试=2024-09 → 模型3
...
第6次: 训练数据=[2024-01, 2024-11], 测试=2024-12 → 模型6
      ↓
输出: 6个模型 + 月度性能报告
```

**示例**：
```bash
make rolling SYMBOLS=BTCUSDT,ETHUSDT \
  ROLLING_START=2024-01 ROLLING_END=2024-12 \
  INITIAL_TRAIN_MONTHS=6 \
  MIN_TRAIN_MONTHS=3 \
  ROLLING_FEATURE_TYPE=comprehensive \
  ROLLING_FREQ=60T \
  ROLLING_FBS=5 \
  ROLLING_USE_TOP_FACTORS=results/dim_compare/.../top_factors.json
```

**输出**：
- `results/rolling_*/monthly_results.csv` - 所有月份的详细结果
- `results/rolling_*/summary.json` - 汇总信息
- `results/rolling_*/monthly_rolling_report.html` - HTML 报告
- `results/rolling_*/model_YYYY-MM.txt` - 每个月份的模型信息

## 关键区别详解

### 区别 1: 训练数据的使用方式

**`make train`**：
- 使用**所有数据**训练一个模型
- 数据范围：START_DATE 到 END_DATE
- 没有时间序列交叉验证（除非指定 CV_FOLDS）

**`make rolling`**：
- 使用**扩展窗口**：每个测试月份使用之前所有月份的数据
- 模拟真实生产环境：用历史数据训练，预测未来
- 自动进行时间序列交叉验证

### 区别 2: 模型数量

**`make train`**：
```
输入: 2024-01-01 到 2024-12-31
输出: 1个模型
```

**`make rolling`**：
```
输入: 2024-01 到 2024-12, INITIAL_TRAIN_MONTHS=6
输出: 6个模型（2024-07, 2024-08, ..., 2024-12）
```

### 区别 3: 评估方式

**`make train`**：
- 单次评估
- 使用 OOS（Out-of-Sample）数据评估（如果指定）
- 适合快速验证模型性能

**`make rolling`**：
- **多次评估**，每个测试月份一个评估结果
- 可以观察模型性能随时间的变化
- 可以检测模型漂移（drift）
- 更接近真实交易场景

### 区别 4: 适用场景

**`make train` 适合**：
- ✅ 快速验证模型性能
- ✅ 参数调优和特征选择
- ✅ 开发阶段测试
- ✅ 一次性评估

**`make rolling` 适合**：
- ✅ **生产环境训练**（推荐）
- ✅ **回测评估**
- ✅ 评估模型稳定性
- ✅ 检测模型漂移
- ✅ 真实交易场景模拟

## 应该用哪个训练生产模型？

### 答案：**`make rolling`** 或 **`make auto-rolling-update`**

**原因**：
1. **更接近真实场景**：滚动训练模拟真实交易环境（用历史数据训练，预测未来）
2. **评估稳定性**：可以观察模型在不同时间段的性能变化
3. **检测漂移**：如果模型性能突然下降，可以及时发现
4. **多个模型**：每个时间段有独立的模型，可以更好地适应市场变化

### 推荐工作流程

#### 方案 A: 使用 `make rolling`（手动指定时间范围）

```bash
# 1. 先运行 dim-compare 获取 top factors
make dim-compare SYMBOLS=BTCUSDT,ETHUSDT \
  START_DATE=2024-01-01 END_DATE=2024-12-31 \
  DIM_COMPARE_FEATURE_TYPE=comprehensive \
  TIMEFRAME=60T

# 2. 设置 DIM_DIR
DIM_DIR=results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_20240101_20241231_tf60T_h5

# 3. 使用 rolling 训练生产模型
make rolling SYMBOLS=BTCUSDT,ETHUSDT \
  ROLLING_START=2024-01 ROLLING_END=2024-12 \
  INITIAL_TRAIN_MONTHS=6 \
  MIN_TRAIN_MONTHS=3 \
  ROLLING_FEATURE_TYPE=comprehensive \
  ROLLING_FREQ=60T \
  ROLLING_FBS=5 \
  ROLLING_USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json
```

#### 方案 B: 使用 `make auto-rolling-update`（自动检测最新数据，推荐）

```bash
# 1. 先运行 dim-compare
make dim-compare SYMBOLS=BTCUSDT,ETHUSDT \
  START_DATE=2024-01-01 END_DATE=2024-12-31 \
  DIM_COMPARE_FEATURE_TYPE=comprehensive \
  TIMEFRAME=60T

# 2. 设置 DIM_DIR
DIM_DIR=results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_20240101_20241231_tf60T_h5

# 3. 自动滚动更新到最新数据（推荐）
make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  MIN_TRAIN_MONTHS=3 \
  --use-top-factors $(DIM_DIR)/top_factors.json \
  --feature-type comprehensive \
  --freq 60T \
  --forward-bars 5
```

**`auto-rolling-update` 的优势**：
- ✅ 自动检测所有可用数据
- ✅ 自动训练到最新月份
- ✅ 支持增量更新（`--update-only`）
- ✅ 更适合生产环境

## 实际使用建议

### 开发阶段
```bash
# 快速验证：使用 make train
make train SYMBOLS=BTCUSDT \
  START_DATE=2024-11-01 END_DATE=2024-12-31 \
  TRAIN_FEATURE_TYPE=comprehensive
```

### 生产阶段
```bash
# 生产训练：使用 make rolling 或 make auto-rolling-update
make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  --use-top-factors results/dim_compare/.../top_factors.json \
  --feature-type comprehensive \
  --freq 60T \
  --forward-bars 5
```

### 回测评估
```bash
# 回测：使用 make rolling（指定历史时间范围）
make rolling SYMBOLS=BTCUSDT \
  ROLLING_START=2024-01 ROLLING_END=2024-12 \
  INITIAL_TRAIN_MONTHS=6 \
  ROLLING_USE_TOP_FACTORS=results/dim_compare/.../top_factors.json
```

## 总结

| 场景 | 推荐命令 | 原因 |
|------|---------|------|
| **生产模型训练** | `make rolling` 或 `make auto-rolling-update` | 更接近真实场景，评估稳定性 |
| **回测评估** | `make rolling` | 可以观察模型在不同时间段的性能 |
| **快速验证** | `make train` | 速度快，适合开发测试 |
| **参数调优** | `make train` | 快速迭代，验证参数效果 |

**关键要点**：
- ✅ **生产环境**：使用 `make rolling` 或 `make auto-rolling-update`
- ✅ **回测**：使用 `make rolling`
- ✅ **开发测试**：使用 `make train`
- ✅ 两者都支持 `--use-top-factors` 加载 dim-compare 的结果

