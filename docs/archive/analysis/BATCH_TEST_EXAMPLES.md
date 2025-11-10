# 📊 批量OOS测试使用示例

## 🚀 快速开始

### 示例1：测试2-9月（默认）

```powershell
.\batch_test.ps1
```

这会测试2025年2月到9月的所有数据。

## 📋 自定义测试

### 示例2：只测试2-5月

```powershell
.\batch_test.ps1 -Pattern "BTCUSDT-aggTrades-2025-0[2-5]\.zip"
```

### 示例3：只测试偶数月（2,4,6,8月）

```powershell
.\batch_test.ps1 -Pattern "BTCUSDT-aggTrades-2025-0[2468]\.zip"
```

### 示例4：测试特定月份（如3月和6月）

```powershell
.\batch_test.ps1 -Pattern "BTCUSDT-aggTrades-2025-0[36]\.zip"
```

### 示例5：测试所有2025年数据

```powershell
.\batch_test.ps1 -Pattern "BTCUSDT-aggTrades-2025-.*\.zip"
```

### 示例6：测试Q2（4-6月）

```powershell
.\batch_test.ps1 -Pattern "BTCUSDT-aggTrades-2025-0[456]\.zip" -Output "q2_results"
```

### 示例7：使用不同模型

```powershell
.\batch_test.ps1 -Model "model_march" -Pattern "BTCUSDT-aggTrades-2025-0[4-9]\.zip"
```

### 示例8：测试不同数据目录

```powershell
.\batch_test.ps1 -DataDir "D:\other\data\path" -Pattern "BTCUSDT.*\.zip"
```

## 🔍 正则表达式模式说明

| 模式 | 说明 | 匹配示例 |
|------|------|----------|
| `0[2-9]` | 2月到9月 | 02, 03, ..., 09 |
| `0[2-5]` | 2月到5月 | 02, 03, 04, 05 |
| `0[2468]` | 偶数月 | 02, 04, 06, 08 |
| `0[13579]` | 奇数月 | 01, 03, 05, 07, 09 |
| `1[0-2]` | 10-12月 | 10, 11, 12 |
| `.*` | 所有文件 | 任意 |
| `0[3-6]\|09` | 3-6月和9月 | 03, 04, 05, 06, 09 |

## 📊 输出结果

批量测试会生成以下文件：

```
results/batch_oos_results/
├── summary.json                          # 汇总结果
├── all_trades.csv                        # 所有交易合并
├── BTCUSDT-aggTrades-2025-02/
│   ├── results.json
│   ├── trades.csv
│   └── equity_curve.csv
├── BTCUSDT-aggTrades-2025-03/
│   ├── results.json
│   ├── trades.csv
│   └── equity_curve.csv
└── ...
```

## 🐍 使用Python直接调用

如果你更喜欢Python：

```powershell
$env:PYTHONPATH = "src"

# 测试2-9月
python scripts/oos_batch_test.py --model model_btc --pattern "BTCUSDT-aggTrades-2025-0[2-9]\.zip"

# 测试特定月份
python scripts/oos_batch_test.py --model model_btc --pattern "BTCUSDT-aggTrades-2025-0[3-6]\.zip" --output q2_test

# 自定义数据目录
python scripts/oos_batch_test.py --model model_btc --data-dir "D:\other\path" --pattern "BTC.*\.zip"
```

## 📈 高级用法

### 组合测试：训练+批量OOS

```powershell
# 1. 训练模型（1月数据）
python scripts/train_model_gpu.py `
    --data "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-01.zip" `
    --model-name "model_jan" `
    --timeframe "5T"

# 2. 批量测试（2-9月）
.\batch_test.ps1 -Model "model_jan" -Pattern "BTCUSDT-aggTrades-2025-0[2-9]\.zip"
```

### 比较不同训练数据的效果

```powershell
# 训练1月模型，测试2-5月
.\train_and_test.ps1 -TrainData "...-2025-01.zip" -TestData "...-2025-02.zip" -ModelName "model_jan"
.\batch_test.ps1 -Model "model_jan" -Pattern "0[2-5]\.zip" -Output "jan_model_q1"

# 训练2月模型，测试3-6月
.\train_and_test.ps1 -TrainData "...-2025-02.zip" -TestData "...-2025-03.zip" -ModelName "model_feb"
.\batch_test.ps1 -Model "model_feb" -Pattern "0[3-6]\.zip" -Output "feb_model_q2"

# 比较结果
# 查看 jan_model_q1/summary.json 和 feb_model_q2/summary.json
```

## 🎯 实用技巧

1. **滚动测试**: 每月用前一个月训练，测试当月
2. **季度测试**: 用Q1训练，测试Q2-Q4
3. **概念漂移检测**: 观察远期月份的表现下降
4. **最佳时间窗口**: 找出模型表现最好的测试期

## ⚠️ 注意事项

- 确保所有测试月份的数据都已下载
- 批量测试会占用较多时间，建议后台运行
- 定期检查 `results/` 目录避免磁盘空间不足
- Pattern必须是有效的正则表达式

