# 回测脚本（Backtesting）

该目录保留两类与生产训练结果直接关联的回测工具：

- `ml_trading.backtesting.vectorbot` — 基于 `MLTradingStrategy` 的逐笔交易回放，带止损/止盈/仓位控制，可输出交易清单和资金曲线。
- `ml_trading.backtesting.nautilus_dim` — 使用 Nautilus Trader 引擎及 LightGBM 模型的多资产维度压缩回测（需在宿主环境安装 nautilus-trader）。
- `oos_june.py` — 固定窗口的 2025 年 6 月样本外测试，加载训练阶段保存的策略与特征工程器，对持出数据进行评估并生成报告。

## 运行示例

```bash
# VectorBot 回测（默认使用 Makefile 中的 MODEL_PATH 变量）
make vectorbot-backtest

# Nautilus Dim 回测（需宿主环境安装 nautilus-trader）
make nautilus-backtest SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT \
    DATA_DIR=data/parquet_data \
    RESULTS_DIR=results/production_dimensionality_20251109_140314 \
    START_DATE=2024-10-01 END_DATE=2024-12-31

# 2025 年 6 月 OOS 评估，覆盖模型与数据路径
make oos-june MODEL_PATH=models/trained_model_btcusdt_20250501_20250531.pkl \
             SCALER_PATH=models/trained_model_btcusdt_20250501_20250531_scalers.pkl \
             OOS_DATA=data/parquet_data/BTCUSDT-aggTrades-2025-06.parquet
```

回测输出目录：
- VectorBot：`results/vectorbot_backtests/{symbol}_{start}_{end}_{timestamp}/`
- Nautilus Dim：`results/nautilus_backtests/{symbols}_{timeframe}_{start}_{end}_{timestamp}/`

> 训练脚本会生成所需的模型与特征缩放器（参见 `ml_trading.models.train_model`），默认从 `data/parquet_data/*.parquet` 读取。若仍使用 ZIP，可将 `OOS_DATA` 指向 `.zip` 文件，脚本会自动解压处理。确保在运行回测/OOS 前已完成训练。

