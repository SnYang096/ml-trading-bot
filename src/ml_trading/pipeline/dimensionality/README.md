# 🚀 降维训练系统

该目录下的模块组成了 Autoencoder + SHAP Top-K 选因的生产级流水线。核心目标是将复杂的因子集合压缩成可解释的低维特征，并使用 LightGBM 构建最终的交易因子模型。

```
src/ml_trading/pipeline/dimensionality/
├── pipeline.py         # 端到端管道入口（CLI：python -m ml_trading.pipeline.dimensionality.pipeline）
├── rolling_training.py # 滚动/季度训练与漂移触发逻辑
└── __init__.py
```

## 模块概要

- `pipeline.py`
  - 数据加载：`UnifiedFeatureDataLoader`
  - Autoencoder 训练：`InterpretableFactorEngine`（内部调用 `UnifiedAutoencoder` + `AutoencoderTrainer`）
  - SHAP Distillation：提取 Top-K 因子并训练轻量级 LightGBM
  - 报告输出：模型指标、Top-K 因子、可视化、JSON 归档
  - 支持命令行参数：`--use-real-data`、`--data-path`、`--symbol`、`--encoding-dim`、`--top-k` 等

- `rolling_training.py`
  - `RollingConfig`：集中管理滚动训练参数
  - `run_quarterly_rolling_training`：季度滑窗训练、性能比较
  - `run_drift_triggered_training`：基于漂移检测触发再训练
  - 依赖 `ml_trading.utils.drift.DriftDetector` 与 `ml_trading.utils.feature_evaluation.FeatureEvaluator`

## 快速使用

```bash
# Demo：使用合成数据验证全流程
make dimensionality-demo

# 真实数据：读取 data/parquet_data 下的聚合数据，保存模型与报告
make dimensionality-real DATA_DIR=/mnt/data/parquet_data SYMBOL=ETHUSDT

# 手动执行（可自定义参数）
PYTHONPATH=src python -m ml_trading.pipeline.dimensionality.pipeline \
    --use-real-data \
    --data-path /mnt/data/parquet_data \
    --symbol BTCUSDT \
    --encoding-dim 12 \
    --top-k 40 \
    --generate-report \
    --save-model
```

## 依赖组件

- `ml_trading/models/autoencoder.py`：统一的 Autoencoder 结构与训练器
- `ml_trading/models/interpretable_factor_engine.py`：整合 Autoencoder、SHAP、LightGBM 的高层封装
- `ml_trading/utils/training.py`：LightGBM 训练封装，自动处理 GPU/CPU 回退
- `ml_trading/utils/sample_data.py`：生成合成数据集用于快速验证
- `ml_trading/utils/drift.py` & `ml_trading/utils/feature_evaluation.py`：漂移检测与特征表现评估工具

> 记得先执行 `pip install -e .`，确保 `ml_trading` 包处于可导入状态。
