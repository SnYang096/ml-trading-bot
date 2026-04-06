# ModelArtifact 使用指南

## 概述

`ModelArtifact` 是统一管理模型部署所需所有组件的类，确保训练和部署的一致性。它封装了：
- **model**: 训练好的模型
- **preprocessor**: 特征预处理器（确保特征转换逻辑一致）
- **used_features**: 模型实际使用的特征列表
- **feature_config**: 特征配置（可选）
- **metadata**: 元数据（策略名称、模型类型等）

## 在训练 Pipeline 中的使用

### 自动保存

`ModelArtifact` 已经在训练脚本中自动集成：

1. **`train_strategy_pipeline.py`**: 训练单个策略时自动保存 ModelArtifact
2. **`rolling_train.py`**: 滚动训练时自动保存 ModelArtifact

**保存位置**：
- 默认：`results/<strategy_name>/`
- 滚动训练：`results/<strategy_name>/<test_month>/`

**保存的文件**：
```
results/sr_reversal/
├── model.pkl                          # 模型文件
├── preprocessor.pkl                   # 预处理器
├── used_features.json                 # 使用的特征列表
├── feature_config.json                # 特征配置（可选）
├── model_artifact_metadata.json      # 元数据
└── results.json                       # 训练结果
```

### 手动保存

如果需要手动创建和保存 ModelArtifact：

```python
from src.time_series_model.strategies.models import ModelArtifact

# 创建 ModelArtifact
artifact = ModelArtifact(
    model=trained_model,
    preprocessor=preprocessor,
    used_features=["feature1", "feature2", "feature3"],
    feature_config={"requested_features": [...]},
    metadata={
        "strategy": "sr_reversal",
        "model_type": "lightgbm",
        "task_type": "regression",
        "avg_cv_metric": 0.85,
    }
)

# 保存到目录
artifact.save(output_dir=Path("results/sr_reversal"))
```

## 在 Nautilus Trader 回测中的使用

### 更新后的代码

`EventDrivenStrategy` 已经更新，支持自动检测和加载 ModelArtifact：

```python
from src.time_series_model.live.event_driven_strategy import EventDrivenStrategy
from nautilus_trader.model import InstrumentId, BarType

# 创建策略实例
strategy = EventDrivenStrategy(
    strategy_name="sr_reversal",
    instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
    bar_types={"15T": bar_type_15m, "1H": bar_type_1h},
    trade_size=0.001,
    model_path="results/sr_reversal",  # ModelArtifact 目录路径
)
```

**自动检测逻辑**：
1. 如果 `model_path` 指向的目录包含 `model_artifact_metadata.json`，则加载 ModelArtifact
2. 否则，尝试加载旧格式的 `model.pkl`（向后兼容）

### 回测命令

使用 `mlbot backtest nautilus` 命令：

```bash
mlbot backtest nautilus \
  --strategy sr_reversal \
  --symbol BTCUSDT \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --model-path results/sr_reversal \
  --output-dir results/nautilus_backtest
```

**参数说明**：
- `--model-path`: ModelArtifact 目录路径（包含 `model_artifact_metadata.json` 的目录）
- 如果未指定，会尝试从默认路径加载：`results/<strategy_name>/`

## 在实盘交易中的使用

### 策略初始化

实盘策略使用与回测相同的代码，自动支持 ModelArtifact：

```python
from src.time_series_model.live.event_driven_strategy import EventDrivenStrategy

# 创建实盘策略
strategy = EventDrivenStrategy(
    strategy_name="sr_reversal",
    instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
    bar_types={"15T": bar_type_15m},
    trade_size=0.001,
    model_path="results/sr_reversal",  # ModelArtifact 目录
)
```

### 预测流程

策略内部使用 ModelArtifact 进行预测：

```python
# 在 _evaluate_entry_signal 方法中
if self.model_artifact is not None:
    # 使用 ModelArtifact 进行预测（自动使用 preprocessor）
    feature_df = pd.DataFrame([feature_dict])
    predictions = self.model_artifact.predict(feature_df)
    prediction = predictions[0]
```

**优势**：
- 自动使用训练时的 preprocessor，确保特征转换一致
- 自动使用 `used_features`，只使用模型需要的特征
- 支持集成模型（模型列表）

## 完整使用示例

### 1. 训练模型（自动保存 ModelArtifact）

```bash
mlbot train strategy \
  --config config/strategies/sr_reversal \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-10-31 \
  --test-size 0.3
```

训练完成后，ModelArtifact 自动保存到 `results/sr_reversal/`

### 2. 验证 ModelArtifact

```python
from pathlib import Path
from src.time_series_model.strategies.models import ModelArtifact

# 加载 ModelArtifact
artifact_dir = Path("results/sr_reversal")
artifact = ModelArtifact.load(artifact_dir)

# 查看信息
info = artifact.get_artifact_info()
print(f"Features: {info['n_features']}")
print(f"Metadata: {info['metadata']}")

# 测试预测
import pandas as pd
test_df = pd.DataFrame({
    "feature1": [1.0, 2.0],
    "feature2": [3.0, 4.0],
    # ... 其他特征
})
predictions = artifact.predict(test_df)
print(f"Predictions: {predictions}")
```

### 3. 运行 Nautilus 回测

```bash
mlbot backtest nautilus \
  --strategy sr_reversal \
  --symbol BTCUSDT \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --model-path results/sr_reversal
```

### 4. 部署到实盘

```python
# 在实盘策略中
strategy = EventDrivenStrategy(
    strategy_name="sr_reversal",
    instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
    bar_types={"15T": bar_type_15m},
    trade_size=0.001,
    model_path="results/sr_reversal",  # 使用训练时保存的 ModelArtifact
)
```

## 向后兼容性

为了保持向后兼容，代码支持两种模式：

1. **ModelArtifact 模式**（推荐）：
   - 目录包含 `model_artifact_metadata.json`
   - 自动加载所有组件（model, preprocessor, used_features, etc.）

2. **旧格式模式**（兼容）：
   - 只有 `model.pkl` 文件
   - 直接加载模型，不使用 preprocessor

## 最佳实践

1. **始终使用 ModelArtifact**：
   - 训练时自动保存，无需额外操作
   - 确保训练和部署的一致性

2. **版本管理**：
   - 为每个训练运行创建独立的输出目录
   - 使用时间戳或版本号区分不同模型版本

3. **验证一致性**：
   - 在部署前验证 ModelArtifact 的完整性
   - 检查 `used_features` 是否与特征计算逻辑匹配

4. **文档记录**：
   - 在 `metadata` 中记录训练参数和配置
   - 便于后续追溯和调试

## 故障排查

### 问题：找不到 ModelArtifact

**错误**：`FileNotFoundError: Model file not found`

**解决**：
1. 检查 `model_path` 是否正确
2. 确认目录包含 `model_artifact_metadata.json`
3. 检查文件权限

### 问题：特征不匹配

**错误**：预测时特征列缺失

**解决**：
1. 检查 `used_features.json` 中的特征列表
2. 确保特征计算逻辑与训练时一致
3. 使用 `ModelArtifact.preprocessor` 确保特征转换正确

### 问题：预测结果不一致

**可能原因**：
1. 使用了不同的 preprocessor
2. 特征计算逻辑不一致
3. 模型版本不匹配

**解决**：
1. 始终使用 ModelArtifact 进行预测
2. 确保特征计算使用相同的配置
3. 验证模型版本和训练参数

## 相关文档

- [系统架构文档](../../../ARCHITECTURE.md) - ModelArtifact 在架构中的位置
- [工作流文档](../../../models/时序模型/工作流："预处理 + 模型 + 后处理"一体化保存与部署.md) - 详细的设计说明
- [测试文件](../../../../tests/unit/test_model_artifact.py) - 完整的使用示例

