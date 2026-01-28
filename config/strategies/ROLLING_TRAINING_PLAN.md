# 滚动训练计划：验证"短时间段表现好"的假设

## 🎯 目标

验证树模型在短时间段（几个月训练 → 几个月预测）的表现是否比长时间段（3年训练 → 3年预测）更好。

## 📋 实施步骤

### 1. 添加相对阈值（分位数）特征 ✅

**已完成**：
- ✅ 在 `features.yaml` 中添加 `atr_percentile_f`
- ✅ 在 `features_suggested_20260128.yaml` 中添加 `atr_percentile_f`

**为什么重要**：
- `atr_percentile_f` 使用滚动分位数（相对位置），而不是硬阈值
- 自适应数据分布变化，更稳健，不容易受到 Regime Shift 影响

### 2. 运行滚动训练

**命令**：
```bash
python3 scripts/run_rolling_training_with_quantile_features.py
```

或直接使用：
```bash
python3 src/time_series_model/pipeline/rolling/rolling_train.py \
  --config config/strategies/sr_reversal_rr_reg_long \
  --symbol BTCUSDT \
  --data-dir data/parquet_data \
  --timeframe 240T \
  --initial-train-months 6 \
  --min-train-months 3 \
  --output-root results/rolling
```

**工作流程**：
```
第1次: 训练=[1-6月], 测试=7月 → 模型1, Sharpe_1
第2次: 训练=[1-7月], 测试=8月 → 模型2, Sharpe_2
第3次: 训练=[1-8月], 测试=9月 → 模型3, Sharpe_3
...
```

### 3. 对比结果

**对比指标**：
- 滚动训练（短时间段）的平均 Sharpe
- 固定训练（3年周期）的 Sharpe
- 改进百分比

**预期**：
- 滚动训练 Sharpe: 1.2 - 1.5（提升 20-50%）
- 固定训练 Sharpe: 0.93（当前结果）

## 📊 特征配置

### 当前特征（已添加分位数特征）

```yaml
requested_features:
  - poc_hal_features_close_f
  - atr_f
  - atr_percentile_f  # ⭐ 相对阈值特征：自适应 regime shift
  - volume_profile_volatility_features_f
```

### 分位数特征的优势

1. **自适应数据分布**：
   - 硬阈值：`cvd_long <= -34895`（固定值）
   - 分位数：`atr_percentile <= 0.2`（相对位置，20%分位数）

2. **适应 Regime Shift**：
   - 即使数据分布变化，20%分位数仍然对应"极端值"
   - 更稳健，不容易失效

3. **滚动窗口计算**：
   - 使用最近 N 个 bar 的历史分布
   - 自动适应当前市场状态

## 🔍 验证方法

### 1. 检查滚动训练结果

```bash
# 查看月度结果
cat results/rolling/sr_reversal_rr_reg_long/monthly_results.json | python3 -m json.tool

# 提取 Sharpe 序列
python3 << 'EOF'
import json
from pathlib import Path

results_file = Path('results/rolling/sr_reversal_rr_reg_long/monthly_results.json')
if results_file.exists():
    data = json.load(open(results_file))
    sharpe_values = [r.get('sharpe', 0) for r in data if r.get('sharpe')]
    print(f"滚动训练 Sharpe 均值: {sum(sharpe_values)/len(sharpe_values):.4f}")
    print(f"Sharpe 范围: [{min(sharpe_values):.4f}, {max(sharpe_values):.4f}]")
EOF
```

### 2. 对比固定训练结果

- 固定训练（3年）：Sharpe = 0.93
- 滚动训练（6个月 → 1个月）：预期 Sharpe = 1.2 - 1.5

### 3. 分析不同时间段

- 分别查看 2023、2024、2025 年的滚动训练结果
- 识别哪些时间段表现好/差
- 分析市场环境差异

## 📈 预期结果

### 成功标准

1. **Sharpe 提升**：
   - 滚动训练 Sharpe > 1.2（比固定训练 0.93 提升 30%+）
   - 验证"短时间段表现好"的假设

2. **稳定性**：
   - 不同月份的 Sharpe 波动较小
   - 说明模型能够适应市场变化

3. **分位数特征有效**：
   - 使用 `atr_percentile_f` 后表现更好
   - 验证相对阈值特征的优势

## 🔗 相关文档

- Regime Shift 分析：`config/strategies/REGIME_SHIFT_ANALYSIS.md`
- Sharpe 分析：`config/strategies/SHARPE_ANALYSIS.md`
- 滚动训练指南：`docs/models/时序模型/命令总览.md`
