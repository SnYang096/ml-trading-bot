# 特征加载器模块使用说明

## 概述

特征加载器模块提供了基于配置文件的特征加载、并行计算和缓存功能。

## 核心组件

1. **feature_function_mapping.py**: 特征计算函数映射表
2. **parallel_computer.py**: 并行特征计算器
3. **strategy_feature_loader.py**: 策略特征加载器

## 配置文件

1. **config/feature_dependencies.yaml**: 特征依赖配置
2. **config/strategy_features.yaml**: 策略特征配置

## 使用示例

### 基本使用

```python
from src.features.loader import StrategyFeatureLoader
import pandas as pd

# 初始化加载器
loader = StrategyFeatureLoader(
    feature_deps_path="config/feature_dependencies.yaml",
    strategy_config_path="config/strategy_features.yaml",
    cache_dir="cache/features",
    use_disk_cache=True,
    use_memory_cache=True,
    max_workers=4,
    parallel_backend="process",
)

# 研究阶段
df_train = loader.load_strategy_features(df_raw, "sr_reversal", fit=True)

# 实盘阶段
df_live = loader.load_strategy_features(df_live_raw, "sr_reversal", fit=False)

# 清除缓存
loader.clear_cache(memory=True, disk=False)
```

### 获取策略特征列表

```python
# 获取策略需要的所有特征（包括依赖）
features = loader.get_strategy_features("sr_reversal")
print(f"Strategy features: {features}")
```

## 配置说明

### feature_dependencies.yaml

定义特征及其依赖关系：

```yaml
features:
  atr:
    module: baseline
    compute_func: BaselineFeatureEngineer._compute_atr
    dependencies: []
    required_columns: ["high", "low", "close"]
    output_columns: ["atr"]
    category: technical_indicator
    description: "Average True Range"
```

### strategy_features.yaml

定义每个策略的特征集：

```yaml
strategies:
  sr_reversal:
    base_feature_types: ["baseline", "default"]
    requested_features:
      - sr_strength_max
      - sqs_hal_high
      - sqs_hal_low
```

## 性能优化

- **并行计算**: 按依赖层级并行计算，充分利用多核 CPU
- **内存缓存**: 同一会话内快速复用
- **磁盘缓存**: 跨会话持久化，避免重复计算

预计性能提升：**2-10倍**（取决于特征数量和 CPU 核心数）

## 注意事项

1. 确保配置文件中的 `compute_func` 在 `feature_function_mapping.py` 中有对应映射
2. 确保 `required_columns` 在 DataFrame 中存在
3. 并行计算时确保计算函数是线程安全的

