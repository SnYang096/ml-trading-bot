# 完整工作流程：从研究到生产

## 概述

正确的 ML 交易模型开发工作流程应该是：

1. **研究阶段** (`make dim-compare`): 评估降维效果，找到最优特征和压缩维度
2. **生产训练** (`make train`): 使用降维后的特征训练生产模型
3. **滚动更新** (`make auto-rolling-update`): 定期用最新数据更新模型

**核心原则**: **所有生产训练都应该使用降维后的特征**（Top-K + Autoencoder），而不是原始482个特征。

## 完整工作流程（推荐）

### 步骤 1: 研究降维效果 (`make dim-compare`)

**目的**: 找到最优的特征集和压缩维度

```bash
# 使用一个季度的数据研究降维效果
make dim-compare SYMBOL=BTCUSDT \
  START_DATE=2025-05-01 END_DATE=2025-07-31 \
  ENCODING_DIM=32
```

**输出文件** (在 `results/production_dimensionality_20250501_20250731/`):
- `production_results.json` - 包含所有4个阶段的性能对比
- `dimensionality_report.html` - HTML 可视化报告
- `top_factors.json` - 代表性特征列表（Stage 3: 60-100个特征）✅ **新增**
- `representative_factors.json` - 代表性特征列表（另一种格式）✅ **新增**
- `production_autoencoder.pth` - 最佳 Autoencoder 模型（Stage 4）✅ **新增**

**关键信息**:
- **代表性特征**: `top_factors.json` 包含60-100个特征名称
- **最佳压缩维度**: `production_results.json` 中的 `data_info.stage4_compressed_dim`（如32）
- **Autoencoder 模型**: `production_autoencoder.pth`
- **性能对比**: HTML 报告展示4个阶段的对比

### 步骤 2: 训练生产模型 (`make train`) - 可选

**目的**: 使用降维后的特征训练单个模型（用于一次性评估或部署）

```bash
# 使用降维后的特征训练模型
DIM_RESULTS_DIR=results/production_dimensionality_20250501_20250731

make train SYMBOL=BTCUSDT \
  START_DATE=2025-01-01 END_DATE=2025-07-31 \
  USE_TOP_FACTORS=$(DIM_RESULTS_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_RESULTS_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**注意**: 
- 从 `production_results.json` 读取 `data_info.stage4_compressed_dim` 作为 `ENCODING_DIM`
- **这一步是可选**的，可以跳过。`make auto-rolling-update` **已经包含训练过程**
- `make train`: 训练**一个**模型（单个时间段）
- `make auto-rolling-update`: **已经包含训练** - 训练**多个**模型（每月一个）

**输出**:
- `models/trained_model_*.pkl` - 生产模型（使用降维特征训练）
- `models/trained_model_*_scalers.pkl` - 特征标准化器
- `models/trained_model_*_info.json` - 模型元信息
- `models/trained_model_*_info_report.html` - HTML 报告

### 步骤 3: 滚动更新 (`make auto-rolling-update`)

**目的**: 滚动训练多个模型（每月一个），评估模型稳定性，使用降维后的特征

**重要**: `make auto-rolling-update` **已经包含训练过程**，会在循环中为每个月训练一个模型。如果只需要滚动更新功能，可以跳过步骤 2。

```bash
# 使用降维特征进行滚动更新
DIM_RESULTS_DIR=results/production_dimensionality_20250501_20250731

make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  USE_TOP_FACTORS=$(DIM_RESULTS_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_RESULTS_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**输出**:
- `results/auto_rolling_*/monthly_results.csv` - 所有月份的详细结果
- `results/auto_rolling_*/summary.json` - 汇总信息
- `results/auto_rolling_*/monthly_rolling_report.html` - HTML 报告
- `results/auto_rolling_*/model_YYYY-MM.txt` - 每个月的模型文件

**增量更新**（每周/每月运行一次）:

```bash
# 只更新新月份（从上次位置继续）
make auto-rolling-update-only SYMBOL=BTCUSDT \
  OUTPUT=results/auto_rolling_btcusdt_XXX \
  USE_TOP_FACTORS=$(DIM_RESULTS_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_RESULTS_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

## 为什么所有训练都应该使用降维特征？

### 1. **特征质量提升**
- Stage 2 (IC筛选): 去除与目标无关的特征
- Stage 3 (相关性筛选): 去除冗余特征（相关性 > 0.9）
- Stage 4 (Autoencoder压缩): 学习特征的低维表示

### 2. **模型性能提升**
- 降维后的特征通常性能更好（从研究结果可以看到）
- 减少过拟合风险
- 提高训练速度

### 3. **生产一致性**
- 研究阶段找到的最优配置应该应用到所有生产训练
- 确保研究和生产使用相同的特征集

## 修复内容总结

### ✅ 已修复的问题

1. **滚动训练降维特征支持**
   - ✅ `auto-rolling-update` 支持 `--use-top-factors` 和 `--use-autoencoder`
   - ✅ 自动检测所有可用数据（跨年度支持）

3. **`dim-compare` 输出增强**
   - ✅ 自动保存 `top_factors.json`（兼容格式）
   - ✅ 自动保存 `representative_factors.json`
   - ✅ 自动保存 `production_autoencoder.pth`（最佳模型）

### 📝 工作流程示例

#### 完整流程示例

```bash
# 1. 研究降维（一个季度数据）
make dim-compare SYMBOL=BTCUSDT \
  START_DATE=2025-05-01 END_DATE=2025-07-31 \
  ENCODING_DIM=32

# 输出目录（记录这个路径）
DIM_DIR=results/production_dimensionality_20250501_20250731

# 2. 训练生产模型（使用降维特征）
make train SYMBOL=BTCUSDT \
  START_DATE=2025-05-01 END_DATE=2025-07-31 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32

# 3. 滚动更新（使用降维特征，自动检测所有数据）
make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32

# 4. 定期更新（每周/每月运行一次）
make auto-rolling-update-only SYMBOL=BTCUSDT \
  OUTPUT=results/auto_rolling_btcusdt_XXX \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

## 文件结构

### `dim-compare` 输出结构

```
results/production_dimensionality_20250501_20250731/
├── production_results.json           # 详细的4阶段对比结果
├── dimensionality_report.html        # HTML 可视化报告
├── top_factors.json                  # ✅ 代表性特征列表（兼容格式）
├── representative_factors.json      # ✅ 代表性特征列表（原始格式）
└── production_autoencoder.pth       # ✅ 最佳 Autoencoder 模型
```

### `top_factors.json` 格式

```json
{
  "top_factors": [
    {"name": "feature_1"},
    {"name": "feature_2"},
    ...
  ],
  "count": 60,
  "source": "dim-compare",
  "stage": "Stage 3: Representative features"
}
```

### `production_autoencoder.pth`
- PyTorch 模型文件
- 包含 Autoencoder 的状态字典
- 用于压缩代表性特征（60-100个）到压缩维度（如32维）

## 常见问题

### Q1: 如何查找所有可用数据？

**A**: `make auto-rolling-update` 会自动检测所有可用数据：
- 支持跨年度数据查找
- 支持多种文件命名格式
- 按时间顺序排序

### Q2: 如何从 `dim-compare` 结果中提取配置？

**A**: `dim-compare` 现在会自动保存：
- `top_factors.json` - 可直接用于 `--use-top-factors`
- `production_autoencoder.pth` - 可直接用于 `--use-autoencoder`
- `production_results.json` - 包含压缩维度信息（用于 `--encoding-dim`）

### Q3: 是否需要在每个训练中都指定降维参数？

**A**: **是的，推荐所有生产训练都使用降维特征**。这样做的好处：
- 性能更好（从研究结果可以看出）
- 训练更快（特征更少）
- 减少过拟合风险
- 与研究阶段保持一致

### Q4: `dim-compare` 是否应该同时训练生产模型？

**A**: **不建议**。`dim-compare` 的目的是研究评估，不应该训练生产模型：
- `dim-compare`: 评估降维效果，找到最优配置
- `make train`: 使用最优配置训练生产模型
- `make auto-rolling-update`: 定期更新生产模型

这样分离的好处：
- 研究和生产职责清晰
- 可以多次研究，只训练一次生产模型
- 可以研究不同的时间范围，选择最优配置后再训练

## 最佳实践

1. **定期重新研究**: 每季度运行一次 `dim-compare`，评估最新的降维效果
2. **使用最新配置**: 每次生产训练都使用最新的 `top_factors.json` 和 `production_autoencoder.pth`
3. **保持一致性**: 研究、训练、滚动更新都使用相同的降维配置
4. **记录配置**: 保存 `dim-compare` 的输出目录，用于后续训练

## 当前状态总结

✅ **已完成的改进**:
- `dim-compare` 自动保存代表性特征列表和 Autoencoder
- `auto-rolling-update` 支持降维特征和跨年度数据查找
- 所有滚动训练都支持 Top-K 和 Autoencoder 参数

✅ **推荐的工作流程**:
1. `make dim-compare` → 找到最优配置
2. `make train` + 降维参数 → 训练生产模型
3. `make auto-rolling-update` + 降维参数 → 滚动更新

现在可以按照这个流程进行研究和生产训练了！

