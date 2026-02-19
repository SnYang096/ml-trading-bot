# CI/CD测试配置

## 概述

测试系统支持CI/CD模式，通过环境变量自动启用更激进的优化配置，大幅减少测试时间。

## 启用CI/CD模式

设置环境变量 `CI=true`：

```bash
# 方式1：直接设置
export CI=true
pytest tests/live_data_stream/

# 方式2：在CI/CD配置中设置
# GitHub Actions示例
env:
  CI: true

# GitLab CI示例
variables:
  CI: "true"
```

## CI/CD模式优化

### 数据量优化

- **正常模式**: 5000 ticks/symbol
- **CI/CD模式**: 100 ticks/symbol（减少98%）

### 测试时间对比

- **正常模式**: ~60秒/测试
- **CI/CD模式**: ~6秒/测试（约90%提速）

### 验证逻辑调整

CI/CD模式下：
- 放宽了bar生成的验证（因为100条tick可能不足以生成完整的1分钟bar）
- 重点验证tick数据处理正确性
- 仍然验证特征计算功能

## 使用建议

### 本地开发

```bash
# 正常模式（完整测试）
pytest tests/live_data_stream/

# 快速模式（手动减少数据量）
pytest tests/live_data_stream/ -k "not slow"
```

### CI/CD环境

```bash
# 自动使用CI/CD模式（100 ticks/symbol）
CI=true pytest tests/live_data_stream/
```

### 手动控制

如果需要手动控制数据量，可以直接修改 `TestConfig.MAX_TICKS_PER_SYMBOL`：

```python
# tests/live_data_stream/test_config.py
MAX_TICKS_PER_SYMBOL = 200  # 手动设置
```

## 配置说明

### TestConfig.get_max_ticks_per_symbol()

自动根据环境返回合适的数据量：

```python
from tests.live_data_stream.test_config import TestConfig

max_ticks = TestConfig.get_max_ticks_per_symbol()
# CI模式下返回100，否则返回5000
```

## 注意事项

1. **测试覆盖度**：CI/CD模式减少了数据量，可能无法覆盖所有边界情况
2. **特征计算**：仍然会计算特征，但数据量较少
3. **Bar生成**：100条tick可能不足以生成完整的1分钟bar，测试会相应调整验证逻辑

## 相关文件

- `tests/live_data_stream/test_config.py` - 测试配置（包含CI/CD模式）
- `tests/live_data_stream/test_multi_symbol.py` - 多symbol测试（支持CI/CD模式）
