# 训练脚本（Training）

## ⚠️ 重要变更

**`make train` 已被移除**，统一使用 `make rolling` 进行训练。

**原因**：
- 滚动训练提供更好的评估（通过扩展窗口训练和多个模型检查点）
- 可以观察模型在不同时间段的性能变化
- 更接近真实交易场景

## 使用方式

```bash
# 推荐：使用 make rolling 进行滚动训练
make rolling SYMBOLS=BTCUSDT \
  ROLLING_START=2024-01 ROLLING_END=2024-12 \
  INITIAL_TRAIN_MONTHS=6 \
  ROLLING_FEATURE_TYPE=comprehensive

# 单个月训练（相当于原来的 make train）
make rolling SYMBOLS=BTCUSDT \
  ROLLING_START=2024-11 ROLLING_END=2024-11 \
  INITIAL_TRAIN_MONTHS=1
```

> 提示：训练完成后模型存放在 `results/rolling_*/latest/` 目录，供 `scripts/backtesting/` 中的回测脚本使用。

