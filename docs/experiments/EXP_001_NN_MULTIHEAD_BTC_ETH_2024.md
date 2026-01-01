# 实验 001: NN Multihead 模型首次回测 (BTC+ETH, 2024)

## 实验时间
2025-01-01

## 目标
使用 NN Multihead (Path Primitives) 模型训练并回测，评估 Sharpe Ratio 等指标。

## 数据配置
- **训练集**: 2024-01-01 ~ 2024-09-30 (9个月)
- **测试集**: 2024-10-01 ~ 2024-12-31 (3个月, OOS)
- **标的**: BTCUSDT, ETHUSDT
- **时间框架**: 4H (240T)
- **FeatureStore Layer**: `features_291404fba6`

## 执行命令

### 1. 构建 FeatureStore
```bash
mlbot feature-store build \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --no-docker
```

### 2. 训练模型
```bash
python3 scripts/train_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-09-30 \
  --epochs 30 \
  --output-dir results/real_btc_eth_2024 \
  --features-store-root feature_store \
  --features-store-layer features_291404fba6
```

### 3. OOS 预测
```bash
python3 scripts/predict_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --model results/real_btc_eth_2024/path_primitives_4h_80h_min_multi_240T/model.pt \
  --output results/real_btc_eth_2024/preds \
  --features-store-root feature_store \
  --features-store-layer features_291404fba6
```

### 4. Rule Router + RL Logs
```bash
python3 scripts/rule_mode_3action.py \
  --preds results/real_btc_eth_2024/preds \
  --output results/real_btc_eth_2024/mode_3action.parquet

python3 scripts/rl_build_logs_3action.py \
  --preds results/real_btc_eth_2024/preds \
  --mode results/real_btc_eth_2024/mode_3action.parquet \
  --timeframe 240T \
  --output results/real_btc_eth_2024/logs_3action.parquet
```

## 输出文件位置

| 文件 | 路径 |
|------|------|
| 模型 | `results/real_btc_eth_2024/path_primitives_4h_80h_min_multi_240T/model.pt` |
| 训练报告 | `results/real_btc_eth_2024/path_primitives_4h_80h_min_multi_240T/report.html` |
| 训练指标 | `results/real_btc_eth_2024/path_primitives_4h_80h_min_multi_240T/metrics.json` |
| BTC 预测 | `results/real_btc_eth_2024/preds/preds_BTCUSDT.parquet` |
| ETH 预测 | `results/real_btc_eth_2024/preds/preds_ETHUSDT.parquet` |
| Mode 标签 | `results/real_btc_eth_2024/mode_3action.parquet` |
| RL Logs | `results/real_btc_eth_2024/logs_3action.parquet` |

## 训练指标
```
Dir Accuracy: 0.5717
Dir AUC: 0.5788 (接近随机)
MFE Spearman: 0.0162 (几乎无预测能力)
```

## 回测结果 (OOS: 2024-10 ~ 2024-12)

| 指标 | BTC | ETH | 组合 |
|------|-----|-----|------|
| **年化 Sharpe** | 2.68 | -0.29 | **1.13** |
| **总收益** | +45.17% | -10.38% | +15.14% |
| **最大回撤** | -14.12% | - | - |
| **信号分布** | 100% Long | 87% Long, 13% Short | - |

## 问题分析

1. **预测头失效**: 
   - `pred_dir_prob` 全部是 1.0（模型只预测做多）
   - `pred_mfe_atr` 和 `pred_mae_atr` 都是 0

2. **模型能力弱**:
   - Dir AUC 0.58 接近随机猜测
   - MFE Spearman 0.016 几乎无预测能力

3. **BTC Sharpe 虚高**:
   - 2024 Q4 是 BTC 牛市（从 $60k 涨到 $100k+）
   - 模型做多的收益主要来自市场趋势，不是预测能力

## 根因

1. **训练数据太少**: 只有 9 个月 2 个标的
2. **样本多样性不足**: BTC 和 ETH 高度相关
3. **模型可能过拟合**: 需要更多正则化或 dropout

## 下一步优化

- [ ] 扩大训练数据 (2023-2024 全年)
- [ ] 增加更多 symbol (TOP 10 代币)
- [ ] 调整模型架构/超参数
- [ ] 增加数据增强或正则化

---
*实验记录自动生成于 2025-01-01*

