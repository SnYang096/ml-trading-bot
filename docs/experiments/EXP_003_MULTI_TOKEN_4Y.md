# EXP 003: 多 Token 4 年训练实验

## 实验目标

1. **扩大样本量**: 从 2 tokens 扩展到 20 tokens
2. **扩大时间范围**: 从 1 年扩展到 4 年 (2021-2024)
3. **多样化标的**: 混合 HighCap/Alt/Meme 三类

## Token 列表 (20个)

| 类型 | Tokens | 数量 |
|------|--------|------|
| **HighCap** | BTC, ETH, BNB, SOL, XRP, ADA, AVAX, LTC | 8 |
| **Alt** | LINK, DOT, ATOM, NEAR, UNI, AAVE, FTM | 7 |
| **Meme** | DOGE, SHIB, PEPE, WIF, FLOKI | 5 |

## 数据配置

- **时间范围**: 2021-01-01 ~ 2024-12-31 (4年)
- **训练集**: 2021-01 ~ 2023-12 (3年)
- **OOS 测试集**: 2024-01 ~ 2024-12 (1年)
- **Timeframe**: 4H (240分钟)
- **预估样本量**: ~105,000 bars (20 symbols × 365 × 6 bars/day × 3 years)

## 实验步骤

### Step 1: 数据下载

```bash
./scripts/run_exp003_data_download.sh
```

### Step 2: 数据转换

```bash
mlbot data convert --cleanup yes --no-docker
```

### Step 3: FeatureStore 构建

```bash
SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,LTCUSDT,LINKUSDT,DOTUSDT,ATOMUSDT,NEARUSDT,UNIUSDT,AAVEUSDT,FTMUSDT,DOGEUSDT,SHIBUSDT,PEPEUSDT,WIFUSDT,FLOKIUSDT"

mlbot feature-store build \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols $SYMBOLS \
  --timeframe 240T \
  --start-date 2021-01-01 \
  --end-date 2024-12-31 \
  --no-docker
```

### Step 4: 训练

```bash
python3 scripts/train_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols $SYMBOLS \
  --start-date 2021-01-01 \
  --end-date 2023-12-31 \
  --out-dir results/exp003_20tokens_4y \
  --epochs 100 \
  --hidden 512 \
  --depth 3
```

### Step 5: OOS 预测

```bash
python3 scripts/predict_path_primitives_mlp.py \
  --model-dir results/exp003_20tokens_4y/path_primitives_* \
  --symbols $SYMBOLS \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --out-dir results/exp003_20tokens_4y/preds
```

### Step 6: 回测

```bash
# Rule Router + E2E 评估
mlbot rule mode-3action --preds-dir results/exp003_20tokens_4y/preds --out-dir results/exp003_20tokens_4y/mode

mlbot rl build-logs-3action \
  --preds-dir results/exp003_20tokens_4y/preds \
  --mode-dir results/exp003_20tokens_4y/mode \
  --out-dir results/exp003_20tokens_4y/logs

mlbot rl run-e2e-3action --logs-dir results/exp003_20tokens_4y/logs --out-dir results/exp003_20tokens_4y/e2e
```

---

## 实验记录

### 数据下载状态

| Token | 2021 | 2022 | 2023 | 2024 | 状态 |
|-------|------|------|------|------|------|
| BTCUSDT | ✓ | ✓ | ✓ | ✓ | 待下载 |
| ETHUSDT | ✓ | ✓ | ✓ | ✓ | 待下载 |
| ... | | | | | |

### 训练指标

| 指标 | 值 | 备注 |
|------|----|----|
| Dir Acc | - | |
| Dir AUC | - | |
| MFE Spearman | - | |
| MAE Spearman | - | |

### OOS 回测结果 (2024)

| Symbol | Sharpe | Return | MaxDD | 信号分布 |
|--------|--------|--------|-------|---------|
| BTCUSDT | - | - | - | - |
| ETHUSDT | - | - | - | - |
| ... | | | | |

**总体 Sharpe**: -
**总体 Return**: -
**总体 MaxDD**: -

---

## 待办事项

- [ ] 下载 20 tokens 数据
- [ ] 转换为 parquet
- [ ] 构建 FeatureStore
- [ ] 训练模型
- [ ] OOS 回测
- [ ] 分析结果

---

## 后续改进 (TODO)

### 特征归一化重构

当前问题：
- 混合使用归一化和非归一化特征
- 全局 StandardScaler 可能导致信息泄露

改进计划：
1. 全部使用归一化特征（参见 `docs/architecture/FEATURE_NORMALIZATION_POLICY.md`）
2. 移除原始价格类特征（ema, sma, macd 等）
3. 使用滚动窗口归一化避免信息泄露

---

*创建时间: 2026-01-01*

