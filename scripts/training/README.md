# 训练脚本（Training）

该目录现在仅包含生产级的训练入口：

- `train_model_enhanced.py` — 使用 `EnhancedFeatureEngineer` + LightGBM 的生产流水线。脚本会保存训练好的策略、特征工程器、指标信息，并生成模型说明 JSON。

## 使用方式

```bash
# 推荐通过 Makefile 调度（会自动设置 PYTHONPATH）
make train-enhanced

# 或者手动执行并覆盖数据路径
PYTHONPATH=src TRAIN_DATA=/path/to/BTCUSDT-aggTrades-2025-05.parquet \
python scripts/training/train_model_enhanced.py
```

> 提示：脚本会优先读取环境变量 `TRAIN_DATA`（默认：`data/parquet_data/BTCUSDT-aggTrades-2025-05.parquet`）。如需使用旧的 ZIP 文件，可改用 `TRAIN_ZIP=/path/to/file.zip`，脚本会自动解压并处理。训练完成后模型保存在 `models/` 目录下，供 `scripts/backtesting/` 中的回测/OOS 脚本复用。

