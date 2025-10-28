# Windows GPU 训练指南

## 🚀 快速开始

### 1. 测试GPU是否可用

```powershell
.\test_gpu.ps1
```

这会检查：
- ✅ LightGBM 是否安装
- ✅ NVIDIA GPU 是否检测到
- ✅ GPU 训练是否正常工作
- ✅ 当前配置状态

### 2. 运行训练和OOS测试

#### 方式A：快速运行（推荐）

```powershell
.\quick_gpu_train.ps1
```

这会自动执行：
1. 训练模型（5月数据）
2. 运行OOS测试（6月数据）
3. 显示性能统计

#### 方式B：交互菜单

```powershell
.\run_gpu_training.ps1
```

提供菜单选项：
- 1️⃣ 仅训练
- 2️⃣ 仅OOS测试
- 3️⃣ 训练 + OOS（完整流程）
- 4️⃣ 生成报告
- 5️⃣ 退出

---

## ⚙️ 配置说明

### GPU已启用

配置文件已修改为启用GPU：`src/ml_trading/config/settings.py`

```python
USE_GPU = True  # ✅ 已启用
```

### GPU参数

```python
GPU_LGBM_PARAMS = {
    "device": "gpu",         # 使用GPU
    "gpu_platform_id": 0,    # GPU平台ID
    "gpu_device_id": 0,      # GPU设备ID（第一块GPU）
    "max_bin": 255,          # 降低显存占用
}
```

### 如何禁用GPU

如果遇到问题，可以临时禁用GPU：

```python
# 修改 src/ml_trading/config/settings.py
USE_GPU = False
```

---

## 📊 脚本说明

### `test_gpu.ps1`
- **用途**：测试GPU是否可用
- **运行时间**：~5秒
- **输出**：GPU状态报告

### `quick_gpu_train.ps1`
- **用途**：快速运行完整流程（训练+OOS）
- **运行时间**：5-30分钟（取决于GPU性能）
- **输出**：
  - `models/trained_model_wavelet_may_2025.pkl`
  - `results/june_2025_oos/`

### `run_gpu_training.ps1`
- **用途**：交互式菜单，选择执行步骤
- **适合**：需要单独运行某个步骤的情况

---

## 🔧 常见问题

### Q1: GPU测试失败怎么办？

**A**: 检查以下几点：

1. **确认GPU驱动安装**
   ```powershell
   nvidia-smi
   ```
   应该显示GPU信息

2. **重新安装LightGBM**
   ```powershell
   pip uninstall lightgbm -y
   pip install lightgbm --config-settings=cmake.args="-DUSE_GPU=1"
   ```

3. **使用CPU训练（备选方案）**
   - 修改 `src/ml_trading/config/settings.py`
   - 设置 `USE_GPU = False`

### Q2: 显存不足怎么办？

**A**: 降低 `max_bin` 参数：

```python
# 在 src/ml_trading/config/settings.py 中修改
GPU_LGBM_PARAMS = {
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "max_bin": 127,  # 从255降到127
}
```

### Q3: 如何切换到第二块GPU？

**A**: 修改 `gpu_device_id`：

```python
GPU_LGBM_PARAMS = {
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 1,  # 使用第二块GPU
    "max_bin": 255,
}
```

### Q4: 训练速度没有提升？

**A**: 可能原因：

1. **数据量太小**：GPU在大数据集上才有优势
2. **CPU瓶颈**：数据加载可能是瓶颈
3. **GPU未正确启用**：检查日志是否显示 "🚀 GPU acceleration enabled"

---

## 📈 性能对比

预期加速比（取决于GPU型号）：

| 数据量 | CPU时间 | GPU时间 | 加速比 |
|-------|---------|---------|--------|
| 100K  | 2分钟   | 1分钟   | 2x     |
| 1M    | 20分钟  | 5分钟   | 4x     |
| 10M   | 3小时   | 30分钟  | 6x     |

实际性能取决于：
- GPU型号（RTX 3080 > RTX 3060 > GTX 1660）
- 特征数量
- 模型复杂度

---

## 🎯 完整工作流

```powershell
# 1. 测试GPU
.\test_gpu.ps1

# 2. 运行完整流程
.\quick_gpu_train.ps1

# 3. 查看结果
cd results\june_2025_oos
# 查看CSV文件

# 4. （可选）生成报告
.\run_gpu_training.ps1
# 选择选项 4
```

---

## 📁 输出文件

训练完成后会生成：

```
ml_project/
├── models/
│   ├── trained_model_wavelet_may_2025.pkl      # 训练好的模型
│   └── feature_scalers_wavelet_may_2025.pkl    # 特征缩放器
│
├── results/
│   └── june_2025_oos/
│       ├── predictions_5T.csv                   # 5分钟预测
│       ├── predictions_15T.csv                  # 15分钟预测
│       ├── trades_5T.csv                        # 5分钟交易记录
│       └── trades_15T.csv                       # 15分钟交易记录
│
└── reports/
    └── june_oos_report.txt                      # OOS报告
```

---

## 💡 提示

1. **首次运行**：建议先运行 `test_gpu.ps1` 确认GPU可用
2. **时间估算**：完整流程（训练+OOS）约需 10-30 分钟
3. **后台运行**：可以最小化PowerShell窗口，但不要关闭
4. **查看日志**：脚本会实时显示训练进度
5. **中断恢复**：如果中断，重新运行脚本即可（会覆盖之前的模型）

---

## 🆘 需要帮助？

如果遇到问题：

1. 查看错误信息
2. 检查 `test_gpu.ps1` 的输出
3. 尝试禁用GPU（`USE_GPU = False`）用CPU训练
4. 检查数据文件是否存在：`data/raw/BTCUSDT-aggTrades-2025-05.zip`

---

## ✨ 建议

- **首次使用**：先用小数据集测试
- **生产环境**：确认GPU稳定后再大规模训练
- **监控显存**：使用 `nvidia-smi` 监控GPU显存使用
- **定期清理**：删除不需要的临时文件节省磁盘空间

---

**祝训练顺利！🚀**

