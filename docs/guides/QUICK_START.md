# 🚀 Windows GPU 快速开始指南

## ⚡ 最快上手（3分钟）

### 第一次使用

```powershell
# 1. 打开PowerShell，进入项目目录
cd D:\GitHub\trading\rlbot\ml_project

# 2. 运行一键启动脚本
.\START_HERE.ps1

# 3. 选择选项1（首次建议完整测试）
# 然后选择 y 开始训练
```

就这么简单！脚本会自动：
- ✅ 检测GPU
- ✅ 测试LightGBM
- ✅ 训练模型（5月数据）
- ✅ 运行OOS测试（6月数据）
- ✅ 生成结果文件

---

## 📋 所有可用脚本

| 脚本名 | 用途 | 运行时间 | 推荐场景 |
|-------|------|----------|---------|
| `START_HERE.ps1` | 🌟 一键启动（推荐） | 5-30分钟 | 首次使用 |
| `test_gpu.ps1` | 测试GPU | ~5秒 | 检查GPU状态 |
| `quick_gpu_train.ps1` | 快速训练+OOS | 10-30分钟 | 日常训练 |
| `run_gpu_training.ps1` | 交互菜单 | 可变 | 需要单独步骤 |

---

## 🎯 常用命令

### 场景1：首次使用

```powershell
.\START_HERE.ps1
```

### 场景2：测试GPU是否正常

```powershell
.\test_gpu.ps1
```

### 场景3：快速运行训练

```powershell
.\quick_gpu_train.ps1
```

### 场景4：只想训练，不要OOS

```powershell
.\run_gpu_training.ps1
# 选择选项 1
```

### 场景5：已有模型，只要OOS

```powershell
.\run_gpu_training.ps1
# 选择选项 2
```

---

## ⚙️ GPU配置

### 当前状态

GPU已启用 ✅

配置位置：`src/ml_trading/config/settings.py`

```python
USE_GPU = True  # 已启用
```

### 如何禁用GPU（改用CPU）

编辑 `src/ml_trading/config/settings.py`：

```python
USE_GPU = False  # 禁用GPU
```

---

## 📊 输出结果

训练完成后，结果保存在：

```
ml_project/
├── models/
│   ├── trained_model_wavelet_may_2025.pkl      # ← 训练好的模型
│   └── feature_scalers_wavelet_may_2025.pkl    # ← 特征缩放器
│
└── results/
    └── june_2025_oos/
        ├── predictions_5T.csv                   # ← 5分钟预测
        ├── predictions_15T.csv                  # ← 15分钟预测
        ├── trades_5T.csv                        # ← 5分钟交易
        └── trades_15T.csv                       # ← 15分钟交易
```

---

## 🔍 检查结果

### 方法1：PowerShell

```powershell
# 查看交易记录
Get-Content results\june_2025_oos\trades_5T.csv | Select-Object -First 10

# 查看预测
Get-Content results\june_2025_oos\predictions_5T.csv | Select-Object -First 10
```

### 方法2：Excel

直接打开CSV文件：
- `results\june_2025_oos\trades_5T.csv`
- `results\june_2025_oos\predictions_5T.csv`

### 方法3：Python

```python
import pandas as pd

# 读取交易记录
trades = pd.read_csv('results/june_2025_oos/trades_5T.csv')
print(trades.head())

# 读取预测
predictions = pd.read_csv('results/june_2025_oos/predictions_5T.csv')
print(predictions.head())
```

---

## ⚠️ 常见问题

### Q: 脚本报错怎么办？

**A**: 依次检查：

1. **确认在正确目录**
   ```powershell
   pwd  # 应该显示 D:\GitHub\trading\rlbot\ml_project
   ```

2. **检查Python环境**
   ```powershell
   python --version  # 应该是 Python 3.8+
   ```

3. **检查数据文件**
   ```powershell
   # 应该存在这些文件
   Test-Path data\raw\BTCUSDT-aggTrades-2025-05.zip
   Test-Path data\raw\BTCUSDT-aggTrades-2025-06.zip
   ```

4. **重新安装依赖**
   ```powershell
   pip install -r requirements.txt
   ```

### Q: GPU不可用，能用CPU吗？

**A**: 可以！CPU也能训练，只是慢一些。

1. 修改配置（或保持默认，脚本会自动降级到CPU）
2. 运行 `.\quick_gpu_train.ps1`
3. 等待完成（可能需要更长时间）

### Q: 训练很慢怎么办？

**A**: 

- **GPU可用但慢**：检查显存占用（nvidia-smi），可能需要降低 `max_bin`
- **CPU训练慢**：正常现象，CPU训练就是慢，考虑安装GPU支持
- **数据加载慢**：确保数据在SSD上

### Q: 如何监控训练进度？

**A**: 

训练时会实时显示：
```
Fold 1/5: Train [0:150000], Val [150000:180000]
  Accuracy: 0.6234
Fold 2/5: Train [0:180000], Val [180000:210000]
  Accuracy: 0.6187
...
```

### Q: 想用不同的数据怎么办？

**A**: 

修改脚本中的数据路径，例如 `train_model_wavelet.py` 中的：
```python
MAY_ZIP = os.path.join('data', 'raw', 'YOUR_DATA.zip')
```

---

## 💡 高级技巧

### 技巧1：后台运行

```powershell
Start-Job -ScriptBlock { cd D:\GitHub\trading\rlbot\ml_project; .\quick_gpu_train.ps1 }

# 查看状态
Get-Job

# 查看输出
Receive-Job -Id 1
```

### 技巧2：性能基准测试

```powershell
# 记录开始时间
$start = Get-Date

# 运行训练
.\quick_gpu_train.ps1

# 计算耗时
$duration = (Get-Date) - $start
Write-Host "Total time: $($duration.TotalMinutes) minutes"
```

### 技巧3：批量训练

创建一个新的脚本 `batch_train.ps1`：

```powershell
# 训练多个配置
$configs = @("config1", "config2", "config3")

foreach ($config in $configs) {
    Write-Host "Training with $config..."
    # 修改配置
    # 运行训练
    .\quick_gpu_train.ps1
}
```

---

## 🎓 学习资源

### 想深入了解？

- **GPU配置详解**：查看 `WINDOWS_GPU_README.md`
- **Makefile说明**：查看 `Makefile`（Linux命令参考）
- **代码详解**：查看 `src/ml_trading/models/lightgbm_model.py`

### 相关文档

```
ml_project/
├── QUICK_START.md              ← 你现在看的这个
├── WINDOWS_GPU_README.md       ← 详细GPU使用指南
├── START_HERE.ps1              ← 一键启动脚本
├── test_gpu.ps1                ← GPU测试
├── quick_gpu_train.ps1         ← 快速训练
└── run_gpu_training.ps1        ← 交互菜单
```

---

## ✅ 检查清单

使用前确认：

- [ ] 在项目目录 `D:\GitHub\trading\rlbot\ml_project`
- [ ] Python 已安装（3.8+）
- [ ] 依赖已安装（`pip install -r requirements.txt`）
- [ ] 数据文件存在（`data/raw/*.zip`）
- [ ] （可选）GPU 驱动已安装（nvidia-smi 可用）

第一次使用：

- [ ] 运行 `.\START_HERE.ps1`
- [ ] 选择选项1（完整测试）
- [ ] 确认GPU状态
- [ ] 开始训练
- [ ] 检查结果文件

---

## 🆘 需要帮助？

如果遇到问题：

1. 📖 查看 `WINDOWS_GPU_README.md` 的常见问题部分
2. 🔍 检查错误信息
3. 🧪 运行 `.\test_gpu.ps1` 诊断
4. 🐛 查看 Python 输出的详细错误

---

**祝你训练顺利！🚀**

有任何问题随时查阅文档或重新运行测试脚本。

