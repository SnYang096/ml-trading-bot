# 订单流监听和补数据系统测试

## 概述

本测试套件验证订单流监听和补数据系统的各项功能，包括：
1. 基本功能测试（1分钟聚合、特征计算、数据保存）
2. Socket中断恢复测试
3. 1天内补数据测试
4. 1天以上补数据测试
5. Feature Store恢复测试

## 文件结构

```
tests/live_data_stream/
├── __init__.py
├── test_config.py                  # 测试配置
├── test_data_simulator.py          # 数据模拟器
├── test_data_downloader.py         # 数据下载辅助
├── test_order_flow_integration.py  # 集成测试
├── test_multi_symbol.py            # 多symbol测试
└── README.md                       # 本文档
```

## 运行测试

### 运行所有测试

```bash
# 使用pytest运行所有测试
python -m pytest tests/live_data_stream/test_order_flow_integration.py -v

# 或者直接运行
cd tests/live_data_stream
python test_order_flow_integration.py
```

### 运行特定测试场景

```bash
# 基本功能测试
python -m pytest tests/live_data_stream/test_order_flow_integration.py::test_basic_functionality -v

# Socket中断恢复测试
python -m pytest tests/live_data_stream/test_order_flow_integration.py::test_socket_interruption_recovery -v

# 1天内补数据测试
python -m pytest tests/live_data_stream/test_order_flow_integration.py::test_gap_fill_within_day -v

# 1天以上补数据测试（需要网络连接）
python -m pytest tests/live_data_stream/test_order_flow_integration.py::test_gap_fill_over_day -v

# Feature Store恢复测试（需要Feature Store支持）
python -m pytest tests/live_data_stream/test_order_flow_integration.py::test_feature_store_recovery -v
```

## 测试数据

### 数据源

测试使用 `data/parquet_data_1s/` 目录下的1秒tick数据：
- 格式：`{SYMBOL}_{YEAR}-{MONTH}.parquet`
- 列：`timestamp, price, volume, side, symbol`
- `side`: 1=buy, -1=sell

### 测试存储

测试数据保存在 `data/test_live_storage/` 目录，避免污染生产数据：
- `ticks/{symbol}/{YYYY-MM-DD}.parquet` - 1分钟聚合tick数据
- `features_15min/{symbol}/{YYYY-MM-DD}.parquet` - 15分钟特征
- `features_4h/{symbol}/{YYYY-MM-DD}.parquet` - 4小时特征

## 测试配置

在 `test_config.py` 中可以配置：
- 测试数据路径：`PARQUET_DATA_1S_DIR`
- 测试存储路径：`TEST_STORAGE_DIR`
- 测试symbol：`TEST_SYMBOL`（默认：BTCUSDT）
- 测试时间范围：`TEST_START_DATE`, `TEST_END_DATE`
- Socket中断时间点：`INTERRUPT_AT`
- 最大tick数量：`MAX_TICKS_PER_TEST`

## 测试场景说明

### 1. 基本功能测试 (`test_basic_functionality`)

验证核心功能：
- 从parquet_data_1s加载1秒tick数据
- 使用OrderFlowListener处理tick数据
- 验证1分钟聚合是否正确
- 验证特征计算是否正常
- 验证数据保存到Parquet

**预期结果**：
- 成功处理tick数据
- 生成正确的1分钟bar
- 特征计算正常
- 数据正确保存

### 2. Socket中断恢复测试 (`test_socket_interruption_recovery`)

验证恢复机制：
- 模拟数据流运行一段时间
- 在指定时间点模拟socket中断
- 验证未完成的bar是否正确保存
- 验证从Parquet恢复状态
- 验证内存窗口恢复
- 验证特征计算器状态恢复
- 继续处理剩余数据，验证数据连续性

**预期结果**：
- 中断时未完成的bar已保存
- 恢复后状态正确
- 数据连续性无问题

### 3. 1天内补数据测试 (`test_gap_fill_within_day`)

验证1天内补数据：
- 模拟数据缺失（1天内）
- 验证从Parquet warmup补数据
- 验证补数据后系统继续正常运行

**预期结果**：
- 成功从Parquet补数据
- 补数据后系统正常运行
- 数据连续性无问题

### 4. 1天以上补数据测试 (`test_gap_fill_over_day`)

验证1天以上补数据：
- 模拟数据缺失（1天以上）
- 使用BinanceMultiSymbolDownloader下载缺失数据
- 验证从币安API补数据
- 验证补数据后系统继续正常运行

**注意**：此测试需要网络连接，可能需要较长时间。

**预期结果**：
- 成功从币安API下载数据
- 成功补数据
- 补数据后系统正常运行

### 5. Feature Store恢复测试 (`test_feature_store_recovery`)

验证Feature Store恢复：
- 模拟1天以上数据缺失
- 下载日交易ticks数据
- 计算特征并保存到Feature Store
- 验证从Feature Store恢复特征

**注意**：此测试需要Feature Store支持，可能需要先构建Feature Store。

**预期结果**：
- 成功下载数据
- 成功计算特征
- 成功保存到Feature Store
- 成功从Feature Store恢复特征

## 依赖

### 必需依赖

- `pandas` - 数据处理
- `pytest` - 测试框架

### 可选依赖

- `ccxt` - 用于从币安API下载数据（测试场景4和5需要）
- `nautilus_trader` - Nautilus Trader支持（如果可用，使用真实对象；否则使用Mock对象）

## 故障排除

### 问题：找不到测试数据文件

**解决方案**：
- 确保 `data/parquet_data_1s/` 目录存在
- 确保有对应symbol和日期的parquet文件
- 检查 `test_config.py` 中的路径配置

### 问题：测试时间过长

**解决方案**：
- 减少 `MAX_TICKS_PER_TEST` 的值
- 使用更小的测试时间范围
- 跳过需要网络连接的测试（使用 `pytest -k "not test_gap_fill_over_day"`）

### 问题：网络连接失败

**解决方案**：
- 检查网络连接
- 跳过需要网络连接的测试
- 使用 `pytest --skip-network` 标记（需要自定义）

### 问题：Feature Store不可用

**解决方案**：
- 跳过Feature Store相关测试
- 先构建Feature Store
- 检查Feature Store配置

## 注意事项

1. **测试数据隔离**：测试使用独立的存储路径 `data/test_live_storage/`，不会污染生产数据
2. **性能平衡**：使用1秒tick数据聚合1分钟，避免处理过大数据量
3. **时间控制**：测试中控制时间范围和数据量，避免测试时间过长
4. **网络依赖**：部分测试需要网络连接，如果网络不可用，相关测试会被跳过
5. **数据清理**：测试后可以选择清理测试数据，避免占用磁盘空间

## 扩展测试

如果需要添加新的测试场景：

1. 在 `test_order_flow_integration.py` 中添加新的测试函数
2. 使用 `@pytest.fixture` 创建必要的测试资源
3. 使用 `pytest.skip()` 跳过不可用的测试
4. 添加适当的断言验证测试结果

## 相关文档

- `src/live_data_stream/README_STORAGE.md` - 存储系统说明
- `src/live_data_stream/USAGE_EXAMPLE.md` - 使用示例
