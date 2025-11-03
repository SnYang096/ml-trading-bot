# `make dim-compare` 完整指南

## 概述

`make dim-compare` 是研究和选择最优特征的关键步骤。它通过4阶段对比（全量特征 → IC筛选 → 相关性筛选 → Autoencoder压缩）找到最优的特征集和压缩维度，用于后续的生产模型训练。

## 核心问题：训练时间窗口选择

### 原则 1: 使用足够长的时间窗口

**推荐**: 使用**一个季度（3个月）到半年（6个月）**的数据进行研究。

**原因**:
- 太短（< 3个月）: 数据不足，特征选择不稳定，可能过拟合
- 太长（> 6个月）: 市场状态可能发生变化，选择出的特征可能不适用于当前市场
- **最佳**: 3-4个月，覆盖不同市场状态，但不会太旧

### 原则 2: 使用近期数据

**推荐**: 使用**最近3-4个月**的数据，而不是历史数据。

**原因**:
- 市场特征会随时间变化
- 近期数据更能反映当前市场状态
- 使用过旧的数据可能导致特征选择不适用于当前市场

### 原则 3: 包含不同市场状态

**推荐**: 尽量包含**上涨、下跌、震荡**等不同市场状态。

**原因**:
- 只在单一市场状态下选择的特征可能在其他状态下失效
- 包含多种市场状态可以提高特征的鲁棒性

## 参数配置指南

### 必需参数

```bash
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-04-01 \
  END_DATE=2025-07-31 \
  ENCODING_DIM=32
```

**参数说明**:
- `SYMBOL`: 交易对（如 BTCUSDT, ETHUSDT）
- `START_DATE`: 训练开始日期（格式: YYYY-MM-DD）
- `END_DATE`: 训练结束日期（格式: YYYY-MM-DD）
- `ENCODING_DIM`: 初始压缩维度（可选，如果不指定会尝试多个维度）

### 可选参数

```bash
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-04-01 \
  END_DATE=2025-07-31 \
  ENCODING_GRID="16,32,64" \
  DIM_COMPARE_ARGS="--autoencoder-epochs 50 --export-model models/best_ae.pkl"
```

**参数说明**:
- `ENCODING_GRID`: 尝试多个压缩维度（如 "16,32,64"），系统会自动选择最优的
- `DIM_COMPARE_ARGS`: 额外的参数
  - `--autoencoder-epochs`: Autoencoder 训练轮数（默认50）
  - `--export-model`: 导出最佳 Autoencoder 模型路径

## 完整流程示例

### 场景：为 BTCUSDT 找到最优特征配置（2025年7月）

假设现在是 2025年7月，我们需要为 BTCUSDT 找到最优的特征配置，用于后续的生产模型训练。

#### 步骤 1: 研究降维（`make dim-compare`）

```bash
# 使用最近一个季度（2025-04-01 到 2025-07-31）的数据进行研究
# 尝试多个压缩维度，系统会自动选择最优的
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-04-01 \
  END_DATE=2025-07-31 \
  ENCODING_GRID="16,32,64,128" \
  DIM_COMPARE_ARGS="--autoencoder-epochs 50"
```

**为什么选择这个时间窗口？**
- ✅ **足够长**: 4个月，覆盖不同市场状态
- ✅ **近期数据**: 使用最新的4个月数据，反映当前市场特征
- ✅ **包含多种状态**: 4个月通常包含上涨、下跌、震荡等不同状态

**输出目录**:
```
results/production_dimensionality_20250401_20250731/
├── production_results.json           # 详细的4阶段对比结果
├── dimensionality_report.html        # HTML 可视化报告
├── top_factors.json                  # 代表性特征列表（60-100个）
├── representative_factors.json      # 代表性特征列表（另一种格式）
└── production_autoencoder.pth       # 最佳 Autoencoder 模型
```

#### 步骤 2: 查看研究结果

查看 `dimensionality_report.html` 报告，重点关注：

1. **4阶段对比表**:
   - Stage 1 (全量特征): 482个特征
   - Stage 2 (IC筛选): ~120个特征
   - Stage 3 (相关性筛选): 60-100个特征
   - Stage 4 (Autoencoder压缩): 16/32/64维

2. **最佳压缩维度**:
   - 查看 `production_results.json` 中的 `data_info.stage4_compressed_dim`
   - 例如: 32维可能是最优的（在压缩比和性能之间取得平衡）

3. **性能指标**:
   - R²、RMSE、MAE
   - 金融指标：Sharpe Ratio、Total Return、Max Drawdown
   - Stage 4 应该接近 Stage 3 的性能（说明压缩有效）

#### 步骤 3: 提取配置信息

从 `production_results.json` 中提取关键信息：

```json
{
  "data_info": {
    "stage3_representatives": 60,
    "stage4_compressed_dim": 32,
    "compression_ratio": 15.03125,
    ...
  },
  "performance": {
    "stage4_vs_stage3": {
      "delta_r2": 0.001,  // 性能下降很小
      ...
    }
  }
}
```

**关键信息**:
- 代表性特征数量: 60个
- 最佳压缩维度: 32维
- 压缩比: 15x（482 → 32）
- 性能下降: 很小（说明压缩有效）

#### 步骤 4: 使用配置进行生产训练

```bash
DIM_DIR=results/production_dimensionality_20250401_20250731

# 方式 1: 滚动更新（推荐，已包含训练）
make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**注意**: 
- `ENCODING_DIM=32` 从 `production_results.json` 中读取 `data_info.stage4_compressed_dim`
- 使用从研究中找到的最优配置

## 时间窗口选择示例

### 示例 1: 保守策略（推荐）

```bash
# 使用最近一个季度（3个月）的数据
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-05-01 \
  END_DATE=2025-07-31 \
  ENCODING_GRID="16,32,64"
```

**适用场景**:
- 首次研究
- 需要快速验证
- 数据有限

### 示例 2: 标准策略（推荐）

```bash
# 使用最近一个季度（4个月）的数据
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-04-01 \
  END_DATE=2025-07-31 \
  ENCODING_GRID="16,32,64,128"
```

**适用场景**:
- 大部分情况下使用
- 需要平衡稳定性和时效性
- 有足够的数据

### 示例 3: 稳健策略

```bash
# 使用半年（6个月）的数据
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-02-01 \
  END_DATE=2025-07-31 \
  ENCODING_GRID="16,32,64,128"
```

**适用场景**:
- 需要更稳定的特征选择
- 可以接受稍微过时的特征
- 数据充足

### 示例 4: 快速验证策略

```bash
# 使用2个月的数据（不推荐，仅用于快速验证）
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-06-01 \
  END_DATE=2025-07-31 \
  ENCODING_DIM=32
```

**适用场景**:
- 快速验证配置是否有效
- 不用于生产
- 数据有限时临时使用

## 压缩维度选择策略

### 策略 1: 自动选择（推荐）

```bash
ENCODING_GRID="16,32,64,128"
```

系统会自动尝试这些维度，并选择性能最好的。

### 策略 2: 固定维度

```bash
ENCODING_DIM=32
```

如果已经知道最优维度（从之前的研究中），可以直接指定。

### 维度选择原则

| 维度 | 压缩比 | 适用场景 |
|------|--------|---------|
| 8-16 | 高（30-60x） | 极简模型，计算资源受限 |
| 32 | 中（15x） | **推荐**，平衡压缩比和性能 |
| 64 | 中低（7.5x） | 需要更好性能 |
| 128+ | 低（< 4x） | 接近原始特征，压缩效果有限 |

**推荐**: 从 32 开始尝试，如果性能不满足要求，可以尝试 64。

## 完整工作流程示例

### 场景：为 BTCUSDT 建立生产模型（2025年7月）

#### 第1步: 数据准备（如果还没有）

```bash
# 下载数据（如果需要）
make data-pipeline DOWNLOAD_SYMBOLS="BTCUSDT" \
  DOWNLOAD_START_YEAR=2025 DOWNLOAD_START_MONTH=4
```

#### 第2步: 研究降维（找到最优配置）

```bash
# 使用最近4个月的数据进行研究
# 尝试多个压缩维度，自动选择最优的
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-04-01 \
  END_DATE=2025-07-31 \
  ENCODING_GRID="16,32,64" \
  DIM_COMPARE_ARGS="--autoencoder-epochs 50"

# 输出目录
DIM_DIR=results/production_dimensionality_20250401_20250731
```

#### 第3步: 查看研究结果

```bash
# 查看 HTML 报告
open $(DIM_DIR)/dimensionality_report.html

# 或者查看 JSON 结果
cat $(DIM_DIR)/production_results.json | jq '.data_info.stage4_compressed_dim'
cat $(DIM_DIR)/production_results.json | jq '.data_info.stage3_representatives'
cat $(DIM_DIR)/production_results.json | jq '.performance.stage4_vs_stage3'
```

**预期结果**:
- 最佳压缩维度: 32
- 代表性特征数量: 60
- Stage 4 性能接近 Stage 3（压缩有效）

#### 第4步: 使用配置进行滚动更新

```bash
# 使用研究找到的最优配置进行滚动更新
# 这会训练多个模型（每月一个），评估模型稳定性
make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

**说明**:
- `INITIAL_TRAIN_MONTHS=6`: 初始训练使用6个月数据
- `USE_TOP_FACTORS`: 使用研究找到的代表性特征
- `USE_AUTOENCODER`: 使用研究找到的最佳 Autoencoder
- `ENCODING_DIM=32`: 使用研究找到的最佳压缩维度

#### 第5步: 查看滚动更新结果

```bash
# 查看结果目录（自动生成）
ROLLING_DIR=results/auto_rolling_btcusdt_*

# 查看 HTML 报告
open $(ROLLING_DIR)/monthly_rolling_report.html

# 查看汇总结果
cat $(ROLLING_DIR)/summary.json
```

**评估指标**:
- 平均收益率
- 平均 Sharpe Ratio
- 平均最大回撤
- 胜率
- 模型稳定性（各月份性能是否稳定）

#### 第6步: 定期更新（可选）

```bash
# 每周/每月运行一次，更新到最新数据
make auto-rolling-update-only SYMBOL=BTCUSDT \
  OUTPUT=results/auto_rolling_btcusdt_XXX \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=32
```

## 常见问题和解决方案

### Q1: 应该多久重新运行一次 `dim-compare`？

**A**: **每季度重新运行一次**（或当市场状态发生重大变化时）。

**原因**:
- 市场特征会随时间变化
- 每季度更新特征选择，保持特征的新鲜度
- 如果性能下降，可能需要更频繁地更新

### Q2: 时间窗口选择多少最合适？

**A**: **推荐3-4个月**。

**权衡**:
- **太短**（< 3个月）: 数据不足，选择不稳定
- **太长**（> 6个月）: 可能包含过时的市场状态
- **最佳**（3-4个月）: 平衡稳定性和时效性

### Q3: 如何判断特征选择是否有效？

**A**: 查看 `dimensionality_report.html` 中的指标：

1. **压缩有效性**:
   - Stage 4 的 R² 应该接近 Stage 3（性能下降 < 5%）
   - 压缩比应该在 10-20x 之间

2. **特征质量**:
   - Stage 3 → Stage 4 的性能下降应该很小
   - 如果性能下降 > 10%，可能需要增加压缩维度

3. **稳定性**:
   - 各阶段的性能应该相对稳定
   - 不应该出现大幅波动

### Q4: 应该使用多少个压缩维度？

**A**: **从 32 开始，根据性能调整**。

**策略**:
- 尝试多个维度: `ENCODING_GRID="16,32,64"`
- 选择性能最好的（通常是在压缩比和性能之间的平衡）
- 如果 32 性能不满足要求，可以尝试 64

### Q5: 如何知道选择的特征是否适用于生产？

**A**: 通过滚动更新评估模型稳定性。

**步骤**:
1. 使用 `dim-compare` 找到最优配置
2. 使用 `auto-rolling-update` 评估模型在不同时间段的稳定性
3. 如果各月份性能稳定，说明特征选择有效
4. 如果性能波动大，可能需要重新运行 `dim-compare`

## 最佳实践总结

1. **时间窗口**: 使用**最近3-4个月**的数据
2. **压缩维度**: 从**32开始**，尝试多个维度自动选择
3. **更新频率**: **每季度**重新运行一次 `dim-compare`
4. **验证方法**: 使用 `auto-rolling-update` 评估模型稳定性
5. **评估指标**: 关注压缩比、性能下降、稳定性

## 快速参考

### 标准工作流程

```bash
# 1. 研究降维（使用最近4个月数据）
make dim-compare \
  SYMBOL=BTCUSDT \
  START_DATE=2025-04-01 \
  END_DATE=2025-07-31 \
  ENCODING_GRID="16,32,64"

# 2. 提取配置
DIM_DIR=results/production_dimensionality_20250401_20250731
BEST_DIM=$(cat $(DIM_DIR)/production_results.json | jq -r '.data_info.stage4_compressed_dim')

# 3. 使用配置进行滚动更新
make auto-rolling-update SYMBOL=BTCUSDT \
  INITIAL_TRAIN_MONTHS=6 \
  USE_TOP_FACTORS=$(DIM_DIR)/top_factors.json \
  USE_AUTOENCODER=$(DIM_DIR)/production_autoencoder.pth \
  ENCODING_DIM=$BEST_DIM
```

### 时间窗口推荐

| 场景 | 时间窗口 | 命令示例 |
|------|---------|---------|
| 快速验证 | 2个月 | `START_DATE=2025-06-01 END_DATE=2025-07-31` |
| **标准推荐** | **3-4个月** | `START_DATE=2025-04-01 END_DATE=2025-07-31` |
| 稳健策略 | 6个月 | `START_DATE=2025-02-01 END_DATE=2025-07-31` |

### 压缩维度推荐

| 维度 | 压缩比 | 推荐场景 |
|------|--------|---------|
| 16 | 30x | 极简模型 |
| **32** | **15x** | **推荐默认值** |
| 64 | 7.5x | 需要更好性能 |
| 128 | 3.75x | 接近原始特征 |

