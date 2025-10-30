# 训练脚本（Training）

该目录现在仅包含生产级的训练入口：

- `train_model.py` — 使用综合特征工程 + LightGBM 的生产流水线。脚本会保存训练好的策略、特征工程器、指标信息，并生成模型说明 JSON。

## 使用方式

```bash
# 推荐通过 Makefile 调度（会自动设置 PYTHONPATH）
make train SYMBOL=BTCUSDT START_DATE=2025-05-01 END_DATE=2025-05-31

# 或者手动执行，指定标的、时间范围和数据目录
PYTHONPATH=src python scripts/training/train_model.py \
    --symbol BTCUSDT \
    --start-date 2025-05-01 \
    --end-date 2025-05-31 \
    --data-dir data/parquet_data \
    --output-dir models \
    --model-name trained_model

# 同时训练多个标的
make train SYMBOLS="BTCUSDT ETHUSDT" START_DATE=2024-01-01 END_DATE=2024-12-31 OVERWRITE=1
```

> 提示：可以使用 `--train-data file1.parquet file2.parquet` 明确指定文件；如需强制覆盖已有模型，添加 `--overwrite` 或 `OVERWRITE=1 make train`。脚本仍支持 `.zip` / `.csv` 文件并自动解压。训练完成后模型存放在 `models/` 目录，供 `scripts/backtesting/` 中的回测 / 实盘脚本复用。

