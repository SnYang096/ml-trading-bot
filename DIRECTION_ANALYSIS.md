# 方向判断问题分析和改进方案

## 🔍 当前方向判断逻辑

### 当前方法（Quantile 方法）

**代码位置**：`src/time_series_model/pipeline/training/rank_ic_utils.py`

**逻辑**：
1. 计算预测值的分位数（`pred_quantile`）：预测值在最近 30 个样本中的排名（0-1）
2. 计算置信度（`confidence_score`）：`|pred_quantile - 0.5| * 2`
3. 生成信号：
   - **Long**: `pred_quantile >= 0.9` 且 `confidence >= 0.85`
   - **Short**: `pred_quantile <= 0.1` 且 `confidence >= 0.85`
   - **Hold**: 其他情况

### 当前表现

从最新训练结果（`rank_ic_results_20251119_172617.json`）：
- **OOS Rank IC**: 0.2177（高）
- **Direction Accuracy**: 50.3%（接近随机）
- **Win Rate**: 13.5%（极低）
- **Total Return**: 0.31%（几乎为 0）
- **Sharpe Ratio**: 0.0037（极低）

### 问题诊断

**核心矛盾**：
- ✅ IC 高（0.2177）→ 说明模型能正确排序
- ❌ Win Rate 低（13.5%）→ 说明交易表现差
- ❌ Direction Accuracy 低（50.3%）→ 说明方向判断不准确

**可能原因**：
1. **排序正确但预测值不准确**
   - 模型能区分好坏（IC 高），但预测值本身可能不准确
   - 预测值的符号可能错误，或幅度不对

2. **分位数方法的问题**
   - 只关注相对排名，不关注绝对值
   - 如果预测值分布与真实收益分布不匹配，方向判断会出错

3. **阈值太极端**
   - 只有最极端的 10% 预测值才会产生信号
   - 可能错过了很多有效的交易机会

## ✅ 改进方案

### 方案 1：直接使用预测值符号（Sign 方法）⭐ 推荐

**逻辑**：
- 如果 `pred > 0` → Long
- 如果 `pred < 0` → Short
- 仍然使用置信度过滤（`confidence >= 0.85`）

**优点**：
- ✅ 简单直接，不依赖分位数
- ✅ 直接使用预测值的符号，更符合直觉
- ✅ 适用于预测值本身有方向性的情况

**使用方法**：
```bash
make ts-r-rank-ic-train \
  RANK_IC_TOP_FACTORS=results/feature_evaluation/top_factors.json \
  RANK_IC_SIGNAL_METHOD=sign
```

### 方案 2：结合预测值符号和分位数（Hybrid 方法）

**逻辑**：
- 只有当**预测值符号和分位数方向一致**时才交易
- Long: `pred > 0` 且 `pred_quantile >= 0.9` 且 `confidence >= 0.85`
- Short: `pred < 0` 且 `pred_quantile <= 0.1` 且 `confidence >= 0.85`

**优点**：
- ✅ 双重验证，减少误信号
- ✅ 结合了两种方法的优点

**使用方法**：
```bash
make ts-r-rank-ic-train \
  RANK_IC_TOP_FACTORS=results/feature_evaluation/top_factors.json \
  RANK_IC_SIGNAL_METHOD=hybrid
```

### 方案 3：基于历史表现优化阈值（Optimized 方法）

**逻辑**：
- 在训练数据上优化预测值阈值，最大化方向准确率
- 使用优化后的阈值生成信号

**优点**：
- ✅ 自动优化，适应数据分布
- ✅ 基于历史表现，更可靠

**使用方法**：
```bash
make ts-r-rank-ic-train \
  RANK_IC_TOP_FACTORS=results/feature_evaluation/top_factors.json \
  RANK_IC_SIGNAL_METHOD=optimized
```

### 方案 4：校准预测值

**逻辑**：
- 使用 sigmoid 或 Platt scaling 校准预测值
- 使预测值分布与真实收益分布匹配

**优点**：
- ✅ 使预测值更准确地反映真实收益
- ✅ 可以改善方向判断的准确性

**使用方法**：
```bash
make ts-r-rank-ic-train \
  RANK_IC_TOP_FACTORS=results/feature_evaluation/top_factors.json \
  RANK_IC_SIGNAL_METHOD=sign \
  RANK_IC_CALIBRATE_PREDICTIONS=1
```

## 🚀 推荐使用顺序

### 第一步：尝试 Sign 方法（最简单，最可能有效）

```bash
make ts-r-rank-ic-train \
  RANK_IC_TOP_FACTORS=results/feature_evaluation/top_factors.json \
  RANK_IC_SIGNAL_METHOD=sign
```

**为什么推荐**：
- 当前 Direction Accuracy 只有 50.3%，说明分位数方法的方向判断不准确
- 直接使用预测值符号可能更准确
- 最简单，最容易验证

### 第二步：如果 Sign 方法不够好，尝试 Hybrid 方法

```bash
make ts-r-rank-ic-train \
  RANK_IC_TOP_FACTORS=results/feature_evaluation/top_factors.json \
  RANK_IC_SIGNAL_METHOD=hybrid
```

### 第三步：如果还是不够好，尝试 Optimized 方法

```bash
make ts-r-rank-ic-train \
  RANK_IC_TOP_FACTORS=results/feature_evaluation/top_factors.json \
  RANK_IC_SIGNAL_METHOD=optimized
```

### 第四步：如果预测值分布不匹配，尝试校准

```bash
make ts-r-rank-ic-train \
  RANK_IC_TOP_FACTORS=results/feature_evaluation/top_factors.json \
  RANK_IC_SIGNAL_METHOD=sign \
  RANK_IC_CALIBRATE_PREDICTIONS=1
```

## 📊 预期改善

使用改进方法后，应该看到：

1. **Win Rate 提升**
   - 从 13.5% → **40-50%**（至少高于随机）

2. **方向准确率提升**
   - 从 50.3% → **55-60%**

3. **交易表现改善**
   - Sharpe Ratio 从 0.0037 → **> 0.5**
   - Total Return 从 0.31% → **> 5%**

## 📝 相关文件

- **改进代码**：`src/time_series_model/pipeline/training/rank_ic_utils_improved.py`
- **主训练代码**：`src/time_series_model/pipeline/training/rank_ic_trainer.py`
- **诊断脚本**：`scripts/test_direction_methods.py`
- **详细文档**：`docs/方向判断改进方案.md`

