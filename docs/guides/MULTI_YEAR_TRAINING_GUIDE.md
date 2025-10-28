# 📚 多年多币种训练指南

## 🎯 训练方案

### 方案：2021-2023训练 → 2024-2025测试

**训练数据：**
- BTC/ETH/SOL
- 2021年1月 - 2023年12月 (36个月 × 3币种 = 108个文件)
- 采样率30%（加速训练）

**测试数据：**
- BTC/ETH/SOL
- 2024年1月 - 2025年9月 (21个月 × 3币种 = 63个文件)

## 🚀 快速开始

### 方式1：一键运行（推荐）

```powershell
cd D:\GitHub\trading\rlbot\ml_project
.\train_2021_2023_test_2024_2025.ps1
```

### 方式2：快速测试（使用少量数据）

```powershell
# 只使用前10个文件测试流程
.\train_2021_2023_test_2024_2025.ps1 -MaxFiles 10 -SampleRate 0.1
```

### 方式3：单币种训练

```powershell
# 只训练BTC
.\train_2021_2023_test_2024_2025.ps1 -Symbols "BTCUSDT" -ModelName "model_btc_2021_2023"
```

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `Symbols` | BTCUSDT,ETHUSDT,SOLUSDT | 币种列表 |
| `SampleRate` | 0.3 | 数据采样率（0.1-1.0） |
| `MaxFiles` | null | 最大文件数（用于测试） |
| `ModelName` | model_2021_2023_multi | 模型名称 |

## 📊 预计时间和资源

### 使用30%采样率（推荐）

- **训练时间**: 10-30分钟 ⏱️
- **内存需求**: 8-16 GB
- **GPU**: 推荐RTX 3080或更高
- **磁盘空间**: 训练数据~60GB，模型~100MB

### 使用100%数据（完整训练）

```powershell
.\train_2021_2023_test_2024_2025.ps1 -SampleRate 1.0
```

- **训练时间**: 30-120分钟 ⏱️⏱️⏱️
- **内存需求**: 16-32 GB
- **数据量**: ~200GB

## 🎓 使用示例

### 示例1：标准训练（30%采样，3币种）

```powershell
.\train_2021_2023_test_2024_2025.ps1
```

预期结果：
- 训练文件: 108个
- 训练时间: ~15分钟
- 测试文件: 63个
- 测试时间: ~10分钟

### 示例2：快速原型测试

```powershell
.\train_2021_2023_test_2024_2025.ps1 -MaxFiles 6 -SampleRate 0.1
```

预期结果：
- 只使用前6个文件
- 只采样10%数据
- 训练时间: ~2分钟

### 示例3：只训练和测试BTC

```powershell
.\train_2021_2023_test_2024_2025.ps1 `
    -Symbols "BTCUSDT" `
    -ModelName "model_btc_hist" `
    -SampleRate 0.5
```

### 示例4：高质量完整训练

```powershell
.\train_2021_2023_test_2024_2025.ps1 `
    -SampleRate 1.0 `
    -ModelName "model_full_2021_2023"
```

⚠️ 需要32GB内存和2小时时间

## 📈 分步执行

如果你想分开执行训练和测试：

### 1. 只训练模型

```powershell
$env:PYTHONPATH = "src"
python scripts/train_multi_year_multi_symbol.py `
    --symbols BTCUSDT ETHUSDT SOLUSDT `
    --start-year 2021 `
    --end-year 2023 `
    --model-name model_2021_2023 `
    --sample-rate 0.3
```

### 2. 批量测试2024年

```powershell
python scripts/oos_batch_test.py `
    --model model_2021_2023 `
    --pattern ".*-2024-.*\.zip" `
    --output test_2024
```

### 3. 批量测试2025年

```powershell
python scripts/oos_batch_test.py `
    --model model_2021_2023 `
    --pattern ".*-2025-.*\.zip" `
    --output test_2025
```

## 📊 结果分析

训练完成后，查看结果：

```powershell
# 查看模型元数据
Get-Content models/model_2021_2023_multi_metadata.json | ConvertFrom-Json

# 查看测试结果
Get-Content results/model_2021_2023_multi_BTCUSDT_2024_2025/summary.json | ConvertFrom-Json
```

## 🎯 为什么这个方案有意义？

1. **充足的训练数据**: 3年数据涵盖多种市场状态
2. **真实的OOS测试**: 用最新的2024-2025数据验证
3. **多币种泛化**: 学习不同资产的共同模式
4. **避免过拟合**: 大规模数据集减少过拟合风险

## ⚠️ 注意事项

1. **内存不足**：降低SampleRate（如0.1或0.2）
2. **时间太长**：使用MaxFiles限制或降低SampleRate
3. **GPU不可用**：会自动降级到CPU（但会更慢）
4. **数据缺失**：确保已下载2021-2025年的数据

## 🔍 故障排查

### 问题：内存溢出

```powershell
# 解决方案：降低采样率
.\train_2021_2023_test_2024_2025.ps1 -SampleRate 0.1
```

### 问题：训练太慢

```powershell
# 解决方案1：减少文件
.\train_2021_2023_test_2024_2025.ps1 -MaxFiles 20

# 解决方案2：单币种
.\train_2021_2023_test_2024_2025.ps1 -Symbols "BTCUSDT"
```

### 问题：GPU不工作

检查GPU：
```powershell
python check_gpu.py
```

## 📚 进阶技巧

### 滚动训练窗口

```powershell
# 2020-2022 → 2023
python scripts/train_multi_year_multi_symbol.py --start-year 2020 --end-year 2022

# 2021-2023 → 2024
python scripts/train_multi_year_multi_symbol.py --start-year 2021 --end-year 2023
```

### 季度分析

```powershell
# 按季度测试
python scripts/oos_batch_test.py --pattern ".*-2024-0[1-3].*"  # Q1
python scripts/oos_batch_test.py --pattern ".*-2024-0[4-6].*"  # Q2
```

## 🎉 期待结果

如果一切正常，你会得到：

1. ✅ 一个在3年历史数据上训练的模型
2. ✅ 在2024-2025年每个月的详细回测结果
3. ✅ 每个币种独立的性能分析
4. ✅ 可用于实盘的参考模型

祝训练顺利！🚀

