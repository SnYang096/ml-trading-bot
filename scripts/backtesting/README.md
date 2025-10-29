# 回测脚本（Backtesting）

该目录保留两类与生产训练结果直接关联的回测工具：

- `vectorbot_backtest.py` — 基于 `MLTradingStrategy` 的逐笔交易回放，带止损/止盈/仓位控制，可输出交易清单和资金曲线。
- `oos_june.py` — 固定窗口的 2025 年 6 月样本外测试，加载训练阶段保存的策略与特征工程器，对持出数据进行评估并生成报告。

## 运行示例

```bash
# VectorBot 回测（默认使用 models/trained_model_enhanced_may_2025.pkl）
make vectorbot-backtest

# 2025 年 6 月 OOS 评估，覆盖模型与数据路径
make oos-june MODEL_PATH=models/trained_model_enhanced_may_2025.pkl \
             SCALER_PATH=models/feature_scalers_enhanced_may_2025.pkl \
             OOS_DATA=data/parquet_data/BTCUSDT-aggTrades-2025-06.parquet
```

> 训练脚本会生成所需的模型与特征缩放器（参见 `scripts/training/train_model_enhanced.py`），默认从 `data/parquet_data/*.parquet` 读取。若仍使用 ZIP，可将 `OOS_DATA` 指向 `.zip` 文件，脚本会自动解压处理。确保在运行回测/OOS 前已完成训练。

