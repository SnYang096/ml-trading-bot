# 📚 使用指南 - GPU训练和OOS测试

## 🚀 快速开始

### 方法1：使用默认参数（1月训练，2月测试）

```powershell
cd D:\GitHub\trading\rlbot\ml_project
.\train_and_test.ps1
```

### 方法2：自定义参数

```powershell
.\train_and_test.ps1 `
    -TrainData "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-03.zip" `
    -TestData "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-04.zip" `
    -ModelName "model_march" `
    -Timeframe "15T"
```

## 📋 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TrainData` | `BTCUSDT-aggTrades-2025-01.zip` | 训练数据路径 |
| `TestData` | `BTCUSDT-aggTrades-2025-02.zip` | 测试数据路径 |
| `ModelName` | `model_btc` | 模型名称（不含扩展名） |
| `Timeframe` | `5T` | 时间框架 (5T=5分钟, 15T=15分钟, 60T=1小时) |

## 💡 使用示例

### 示例1：训练1月数据，测试2月数据（默认）

```powershell
.\train_and_test.ps1
```

### 示例2：训练2月数据，测试3月数据

```powershell
.\train_and_test.ps1 `
    -TrainData "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-02.zip" `
    -TestData "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-03.zip" `
    -ModelName "model_feb"
```

### 示例3：使用15分钟时间框架

```powershell
.\train_and_test.ps1 `
    -ModelName "model_15m" `
    -Timeframe "15T"
```

### 示例4：训练Q1数据，测试Q2数据

```powershell
# 需要先合并多个月份的数据，或者分别训练测试
.\train_and_test.ps1 `
    -TrainData "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-01.zip" `
    -TestData "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-04.zip" `
    -ModelName "model_q1_to_q2"
```

## 🔧 分步执行

如果你想分开执行训练和测试：

### 1. 只训练模型

```powershell
$env:PYTHONPATH = "src"
python scripts/train_model_gpu.py `
    --data "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-01.zip" `
    --model-name "model_jan" `
    --timeframe "5T"
```

### 2. 只运行OOS测试

```powershell
$env:PYTHONPATH = "src"
python scripts/oos_test.py `
    --model "model_jan" `
    --data "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-02.zip" `
    --output "jan_to_feb_test"
```

## 📊 输出文件

### 训练输出

- `models/{ModelName}.txt` - 训练好的LightGBM模型
- `models/{ModelName}_metadata.json` - 模型元数据（特征、指标等）

### OOS测试输出

- `results/{ModelName}_oos/backtest_results.json` - 回测结果摘要
- `results/{ModelName}_oos/trades.csv` - 所有交易记录
- `results/{ModelName}_oos/equity_curve.csv` - 权益曲线数据

## ⚙️ 高级选项

### 禁用GPU（使用CPU）

```powershell
python scripts/train_model_gpu.py `
    --data "path/to/data.zip" `
    --model-name "model_cpu" `
    --no-gpu
```

### 查看脚本帮助

```powershell
python scripts/train_model_gpu.py --help
python scripts/oos_test.py --help
```

## 📈 时间框架选项

- `1T` - 1分钟（数据量大，训练慢）
- `5T` - 5分钟（推荐，平衡）
- `15T` - 15分钟（中等频率）
- `30T` - 30分钟
- `60T` - 1小时（低频率）
- `240T` - 4小时
- `1D` - 1天

## 🎯 最佳实践

1. **数据分割**：使用连续的月份，训练集和测试集不要重叠
2. **时间框架**：从5分钟开始，根据结果调整
3. **模型命名**：使用有意义的名称，如 `model_jan_5m`、`model_q1_15m`
4. **结果分析**：重点关注Max Drawdown和Win Rate

## 🐛 常见问题

### Q: 提示数据文件不存在？
A: 使用 `.\download_to_agg_data.ps1` 下载数据

### Q: GPU不可用？
A: 脚本会自动降级到CPU，或使用 `--no-gpu` 参数

### Q: 训练时间太长？
A: 使用更大的时间框架（如15T、60T）或减少数据量

### Q: 如何比较不同模型？
A: 查看各自的 `backtest_results.json` 文件对比指标

## 📞 需要帮助？

查看其他文档：
- `QUICK_START.md` - 快速入门
- `DATA_DOWNLOAD_README.md` - 数据下载指南
- `快速下载说明.txt` - 中文下载说明

