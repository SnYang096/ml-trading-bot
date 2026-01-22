# FR/ET优化验证实验

## 优化状态

✅ **MEAN_REGIME优化已应用**:
- `mean_deviation_z_abs_min_pct`: 0.85 → 0.6 (放宽)
- `mean_path_length_min_pct`: 0.7 → 0.5 (放宽)
- `mean_atr_percentile_min`: 0.8 → 0.5 (放宽)
- 新增 `mean_path_efficiency_max_pct`: 0.4 (低效率路径)
- 新增 `mean_price_dir_consistency_max_pct`: 0.5 (不稳定方向)
- 新增 `mean_jump_risk_max_pct`: 0.3 (低跳空风险)

✅ **FR/ET Gate Rules优化已应用**:
- FR新增: `path_efficiency_pct > 0.4` (拒绝高效率路径)
- FR新增: `price_dir_consistency_pct > 0.5` (拒绝稳定方向)
- FR新增: `deviation_z_abs_pct < 0.6` (拒绝低偏离)
- ET新增: 同样的三个约束

## 实验配置说明

**不是关闭了regime**，而是有两个配置来测试不同效果：

### 1. baseline (有Regime + 优化后的Gate Rules)
- **启用**: Regime过滤 + Gate Veto + Semantic Veto
- **使用**: 优化后的MEAN_REGIME判断（更宽松，能识别更多适合FR/ET的场景）
- **使用**: 优化后的FR/ET Gate Rules（更严格的物理特征约束）
- **目的**: 测试完整优化效果（Regime + Gate Rules）

### 2. no_regime_filter (无Regime + 优化后的Gate Rules)
- **启用**: Gate Veto + Semantic Veto
- **关闭**: Regime过滤
- **使用**: 优化后的FR/ET Gate Rules（更严格的物理特征约束）
- **目的**: 测试Gate Rules单独的效果（不依赖Regime）

### 3. only_gate_rules (有Regime + 优化后的Gate Rules + 无Semantic Veto)
- **启用**: Regime过滤 + Gate Veto
- **关闭**: Semantic Veto
- **使用**: 优化后的MEAN_REGIME和Gate Rules
- **目的**: 测试Semantic Veto的影响

## 运行实验

### 前置条件

需要准备logs文件（`logs_3action.parquet`），通常通过以下步骤生成：

```bash
# 1. 训练模型
python3 scripts/train_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-09-30 \
  --epochs 30 \
  --output-dir results/real_btc_eth_2024 \
  --features-store-root feature_store

# 2. 预测
python3 scripts/predict_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --model results/real_btc_eth_2024/path_primitives_4h_80h_min_multi_240T/model.pt \
  --output results/real_btc_eth_2024/preds \
  --features-store-root feature_store

# 3. 生成logs
python3 scripts/rule_mode_3action.py \
  --preds results/real_btc_eth_2024/preds \
  --output results/real_btc_eth_2024/mode_3action.parquet

python3 scripts/rl_build_logs_3action.py \
  --preds results/real_btc_eth_2024/preds \
  --mode results/real_btc_eth_2024/mode_3action.parquet \
  --data-path data/parquet_data \
  --timeframe 240T \
  --output results/real_btc_eth_2024/logs_3action.parquet
```

### 运行实验

```bash
# 方法1: 使用Python脚本
python3 scripts/experiment_regime_gate.py \
  --logs results/real_btc_eth_2024/logs_3action.parquet \
  --output-dir results/experiments_optimized \
  --features-store-root feature_store \
  --features-store-layer <layer_name> \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-10-01 \
  --end-date 2024-12-31

# 方法2: 使用CLI命令
mlbot experiment regime-gate \
  --logs results/real_btc_eth_2024/logs_3action.parquet \
  --output-dir results/experiments_optimized \
  --features-store-root feature_store \
  --features-store-layer <layer_name> \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-10-01 \
  --end-date 2024-12-31
```

## 预期结果对比

### 优化前（baseline）
- FR: Sharpe 0.000, 交易数 0
- ET: Sharpe 0.000, 交易数 0
- 整体: Sharpe 4.657, 交易数 660

### 优化后预期（baseline）
- FR: Sharpe > 0, 交易数 > 0（MEAN_REGIME样本增加）
- ET: Sharpe > 0, 交易数 > 0（MEAN_REGIME样本增加）
- 整体: Sharpe 保持或提升

### 优化后预期（no_regime_filter）
- FR: Sharpe 改善（从-1.641提升）
- ET: Sharpe 改善（从-2.398提升）
- 整体: Sharpe 提升（Gate Rules单独效果）

## 验证要点

1. **MEAN_REGIME样本数**: 应该从2个增加到更多
2. **FR/ET交易数**: baseline配置下应该有FR/ET交易
3. **FR/ET Sharpe**: 应该从负值改善到正值或接近0
4. **整体KPI**: 应该保持或提升
