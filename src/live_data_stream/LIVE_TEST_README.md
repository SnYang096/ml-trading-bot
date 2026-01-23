# 币安实盘测试说明

## 概述

实盘测试脚本用于验证多symbol数据接收和订单流特征计算功能，使用空策略（不执行交易）。

## 功能验证

测试以下订单流特征的计算：
- **VPIN** (Volume-synchronized Probability of Informed Trading)
- **CVD** (Cumulative Volume Delta)
- **Trade Cluster** (交易聚类)
- **Volume Profile** (成交量分布)
- **VWAP** (Volume Weighted Average Price)

## 使用方法

### 基本使用

```bash
# 使用默认symbol（BTCUSDT, ETHUSDT, SOLUSDT），运行10分钟
python scripts/run_live_test.py

# 指定symbol和运行时长
python scripts/run_live_test.py --symbols BTCUSDT ETHUSDT --duration 5

# 使用测试网
python scripts/run_live_test.py --testnet

# 详细输出
python scripts/run_live_test.py --verbose
```

### 参数说明

- `--symbols`: 交易对符号列表（默认: BTCUSDT ETHUSDT SOLUSDT）
- `--duration`: 运行时长（分钟，默认: 10）
- `--testnet`: 使用测试网（默认: 主网）
- `--storage`: 存储路径（默认: data/live_storage）
- `--verbose`: 详细输出

## API Key配置

### 主网（默认）

API key从 `config/local/binance_mainnet.env` 加载：

```bash
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

### 测试网

使用 `--testnet` 参数时，从 `config/local/binance_testnet.env` 加载。

## 输出说明

### 实时输出

- 每1000条tick输出一次统计
- 每15分钟输出一次特征摘要

### 特征摘要格式

```
================================================================================
📊 订单流特征摘要
================================================================================

🔹 BTCUSDT:
   已处理tick数: 15000
   VPIN:
     vpin: 0.123456
   CVD:
     cvd: 1234.56
     cvd_change_1: 12.34
   Trade Cluster:
     trade_cluster_*: ...
   Volume Profile:
     vp_*: ...
   VWAP:
     vwap: 50000.00
   内存窗口: 240 条bar
   最新1分钟bar: 2024-01-01 12:00:00
   最新15分钟特征: 2024-01-01 12:15:00
```

## 特征验证

### VPIN特征

验证以下VPIN相关特征：
- `vpin` - 当前VPIN值
- `vpin_*` - VPIN衍生特征

### CVD特征

验证以下CVD相关特征：
- `cvd` - 累积成交量差
- `cvd_change_1` - 1分钟CVD变化
- `cvd_change_5` - 5分钟CVD变化
- `cvd_change_5_normalized` - 归一化CVD变化

### Trade Cluster特征

验证以下交易聚类特征：
- `trade_cluster_*` - 交易聚类相关特征

### Volume Profile特征

验证以下成交量分布特征：
- `vp_*` - 成交量分布特征
- `vpvr_*` - 成交量分布波动率特征

### VWAP特征

验证以下VWAP相关特征：
- `vwap` - 成交量加权平均价
- `vwap_*` - VWAP衍生特征

## 数据保存

测试过程中会自动保存：
- 1分钟聚合tick数据：`data/live_storage/ticks/{symbol}/{YYYY-MM-DD}.parquet`
- 15分钟特征：`data/live_storage/features_15min/{symbol}/{YYYY-MM-DD}.parquet`
- 4小时特征：`data/live_storage/features_4h/{symbol}/{YYYY-MM-DD}.parquet`

## 注意事项

### 安全

- ⚠️ **使用主网API key，请谨慎操作**
- 空策略不执行交易，只监听数据
- 建议先用测试网验证（`--testnet`）

### 网络

- 确保网络可以访问币安服务器
- 如果在中国大陆，可能需要代理

### 数据量

- 实盘数据流非常快，注意控制运行时长
- 建议先用短时长测试（如5分钟）

## 故障排除

### 问题1：API key加载失败

**错误**: `无法加载API key`

**解决方案**:
- 检查 `config/local/binance_mainnet.env` 文件是否存在
- 确认文件格式正确：`BINANCE_API_KEY=...` 和 `BINANCE_API_SECRET=...`
- 确认API key有效

### 问题2：连接失败

**错误**: `WebSocket connection error`

**解决方案**:
- 检查网络连接
- 检查防火墙设置
- 使用代理（如需要）

### 问题3：没有接收到数据

**可能原因**:
- symbol名称错误
- 市场类型错误（spot vs futures）

**解决方案**:
- 确认symbol名称正确（如 "BTCUSDT"）
- 确认使用期货市场（`-PERP.BINANCE`）

## 相关文件

- `src/live_data_stream/live_test_strategy.py` - 实盘测试策略
- `scripts/run_live_test.py` - 运行脚本
- `config/local/binance_mainnet.env` - 主网API key配置
- `config/local/binance_testnet.env` - 测试网API key配置
