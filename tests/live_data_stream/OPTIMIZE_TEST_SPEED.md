# 测试速度优化指南

## 当前优化措施

### 1. 减少测试数据量

- `MAX_TICKS_PER_SYMBOL`: 从5000降到1000（每个symbol）
- `MAX_TICKS_PER_TEST`: 从10万降到1万（每个测试）
- 快速测试模式：进一步减少50%数据量

### 2. 优化特征计算

- 测试时使用更长的特征计算间隔（60分钟，实际不会触发）
- 使用更小的内存窗口（1小时 vs 4小时）

### 3. 减少测试symbol数量

- 快速测试模式：只使用2个symbol（vs 3个）

## 进一步优化选项

### 选项1：跳过特征计算（极速测试）

在 `test_config.py` 中设置：
```python
TEST_SKIP_FEATURE_COMPUTE = True
```

### 选项2：使用更少的数据

在 `test_config.py` 中设置：
```python
MAX_TICKS_PER_SYMBOL = 500  # 进一步减少
```

### 选项3：只运行关键测试

```bash
# 只运行基本功能测试
pytest tests/live_data_stream/test_multi_symbol.py::test_multi_symbol_basic_functionality -v

# 跳过慢速测试
pytest tests/live_data_stream/test_multi_symbol.py -v -k "not interruption and not gap_fill"
```

### 选项4：并行运行测试

```bash
# 使用pytest-xdist并行运行
pytest tests/live_data_stream/test_multi_symbol.py -n auto
```

## 性能对比

- **优化前**: ~101秒/测试（5000 ticks/symbol, 3 symbols）
- **优化后**: ~67秒/测试（300 ticks/symbol, 2 symbols）
- **提速**: 约34%的改进

## 进一步优化建议

如果还需要更快，可以：

1. **减少到200 ticks/symbol**: 预计~45秒
2. **只测试1个symbol**: 预计~30秒
3. **跳过特征计算**: 预计~10秒（但会减少测试覆盖度）

## 注意事项

- 快速测试模式可能会减少测试覆盖度
- 建议在CI/CD中使用快速模式，本地开发时使用完整模式
