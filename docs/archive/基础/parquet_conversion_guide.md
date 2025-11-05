# ZIP to Parquet 数据转换指南

## 概述

为了提高数据加载速度和训练效率，我们实现了从 ZIP 格式到 Parquet 格式的数据转换功能。

## 主要特性

- ✅ **批量转换**：一次性转换所有 ZIP 文件到 Parquet 格式
- ✅ **订单流特征**：自动计算并保存 CVD、taker_buy_ratio 等订单流特征
- ✅ **多币种支持**：支持 ETH、BTC、SOL 等多种币种
- ✅ **自动备份**：转换前自动备份原始 ZIP 文件
- ✅ **自动加载**：训练脚本会优先使用 Parquet 文件（如果存在）

## 使用方法

### 1. 批量转换所有 ZIP 文件（推荐）

```bash
cd ml_project
make convert-all-zip-to-parquet
```

这将：
- 读取 `data/agg_data` 目录下所有 `*aggTrades-*.zip` 文件
- 转换为 Parquet 格式并保存到 `data/parquet_data` 目录
- 自动备份原始 ZIP 文件到 `data/backup_zip` 目录

### 2. 交互式转换

```bash
cd ml_project
make convert-zip-to-parquet
```

这将启动交互式转换脚本，可以选择性地清理已转换的 ZIP 文件。

## 输出格式

转换后的 Parquet 文件包含以下列：

### 基础 OHLCV 数据
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `timestamp`: 时间戳（索引）
- `trade_count`: 交易笔数
- `symbol`: 交易对符号（如 BTC-USD）

### 订单流特征
- `buy_qty`: 买方成交量
- `sell_qty`: 卖方成交量
- `taker_buy_ratio`: 主动买入比例
- `cvd`: 累积成交量差（Cumulative Volume Delta）
- `cvd_short`: 短期 CVD（20 个周期）
- `cvd_medium`: 中期 CVD（60 个周期）
- `cvd_long`: 长期 CVD（288 个周期）
- `cvd_change_1`: 当前周期 delta
- `cvd_change_5`: 5 周期累计 delta
- `cvd_change_20`: 20 周期累计 delta
- `cvd_normalized`: 归一化 CVD

## 文件命名规则

- 输入: `BTCUSDT-aggTrades-2021-01.zip`
- 输出: `BTC-USD_2021-01.parquet`

## 性能提升

使用 Parquet 格式相比 ZIP 格式：
- ⚡ 数据加载速度提升 10-20 倍
- 💾 支持列式读取，只加载需要的列
- 🗜️ 更高的压缩率（使用 Snappy 压缩）
- 📊 内置元数据，快速统计信息

## 训练脚本自动使用 Parquet

所有使用 `ml_trading.data_tools.rolling_data` 中 `load_and_process_file()` 函数的训练脚本会自动：
1. 首先检查是否存在对应的 Parquet 文件
2. 如果存在，直接加载 Parquet（快速）
3. 如果不存在，则回退到加载 ZIP 文件（兼容性）

示例：
```python
from ml_trading.data_tools.rolling_data import load_and_process_file

# 会自动检查并优先使用 parquet
df = load_and_process_file("data/agg_data/BTCUSDT-aggTrades-2021-01.zip")
```

## 目录结构

```
data/
├── agg_data/           # 原始 ZIP 文件
│   ├── BTCUSDT-aggTrades-2021-01.zip
│   ├── ETHUSDT-aggTrades-2021-01.zip
│   └── ...
├── parquet_data/       # 转换后的 Parquet 文件
│   ├── BTC-USD_2021-01.parquet
│   ├── ETH-USD_2021-01.parquet
│   └── ...
└── backup_zip/         # 备份的 ZIP 文件
    ├── BTCUSDT-aggTrades-2021-01.zip
    └── ...
```

## 注意事项

1. **磁盘空间**：Parquet 文件通常比 ZIP 小 20-30%，但转换过程需要临时空间
2. **订单流特征**：只有当 ZIP 文件包含 `is_buyer_maker` 列时才会计算订单流特征
3. **备份管理**：定期清理 `backup_zip` 目录以节省空间
4. **增量转换**：如果 Parquet 文件已存在，转换脚本会跳过该文件

## 故障排查

### 问题：转换失败
- 检查 ZIP 文件是否损坏
- 确保有足够的磁盘空间
- 查看日志输出中的错误信息

### 问题：训练脚本仍使用 ZIP
- 确认 Parquet 文件存在于 `data/parquet_data` 目录
- 检查文件命名是否符合规则
- 查看训练脚本的日志输出

## 相关命令

```bash
# 批量转换
make convert-all-zip-to-parquet

# 查看 Parquet 文件
ls -lh data/parquet_data/

# 查看备份
ls -lh data/backup_zip/

# 清理备份（如果需要）
rm -rf data/backup_zip/*.zip
```

## 更多信息

- 转换脚本: `scripts/data_conversion/convert_zip_to_parquet.py`
- 数据加载工具: `scripts/common/data_utils.py`
- Makefile 命令: `Makefile` (line 234-256)

