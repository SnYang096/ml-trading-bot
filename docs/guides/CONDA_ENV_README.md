# Conda环境 `quant` 使用说明

## 📦 环境信息

- **环境名称**: quant
- **Python版本**: 3.11
- **包管理**: conda
- **位置**: `/home/yin/miniconda3/envs/quant`

## ✅ 已安装的包

### 核心依赖
- `lightgbm==4.6.0` (CPU版本)
- `numpy==2.3.3`
- `pandas==2.3.3`
- `scikit-learn==1.7.2`
- `scipy==1.16.2`

### 机器学习
- `optuna==4.5.0`

### 可视化
- `matplotlib==3.10.7`
- `seaborn==0.13.2`

### 其他
- `python-dotenv==1.1.1`
- `pywavelets==1.9.0`

## 🚀 使用方法

### 激活环境

```bash
conda activate quant
```

### 运行训练

```bash
conda activate quant
cd /home/yin/trading/rlbot/ml_project
PYTHONPATH=src python scripts/train_model_enhanced.py
```

### 运行OOS测试

```bash
conda activate quant
cd /home/yin/trading/rlbot/ml_project
make oos-months
```

### 退出环境

```bash
conda deactivate
```

## ⚠️ GPU状态

### 当前状态
- **GPU硬件**: ✅ RTX 3080 (CUDA 12.8)
- **LightGBM GPU**: ❌ 未启用

### 说明
虽然安装的lightgbm标记为cuda版本，但实际没有GPU Tree Learner功能。这是conda-forge的已知限制。

### GPU选项

#### 选项1：继续使用CPU（推荐）
- CPU版本已完全可用
- 适合中小规模数据
- 无需额外配置

```bash
# 直接使用，默认CPU
conda activate quant
make train-wavelet
```

#### 选项2：从源码编译GPU版本（高级）
如果需要GPU加速，需要从源码编译：

```bash
# 1. 安装OpenCL
sudo apt-get install ocl-icd-opencl-dev

# 2. 从源码编译
cd /tmp
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake -DUSE_GPU=1 ..
make -j4

# 3. 在quant环境中安装
conda activate quant
pip uninstall lightgbm -y
cd ../python-package
python setup.py install
```

## 📝 Makefile命令

所有Makefile命令现在会使用系统默认环境（需要手动激活quant）：

```bash
conda activate quant

# 训练
make train-wavelet
make train-enhanced

# OOS测试
make oos-june
make oos-months

# 其他
make test
make clean
```

## 🔄 环境更新

### 添加新包

```bash
conda activate quant
pip install <package-name>
# 或
conda install -c conda-forge <package-name>
```

### 导出环境

```bash
conda activate quant
conda env export > environment.yml
```

### 从环境文件重建

```bash
conda env create -f environment.yml
```

## 🗑️ 删除环境

```bash
conda deactivate
conda env remove -n quant
```

## 💡 最佳实践

### 1. 始终激活环境

```bash
# 每次开始工作时
conda activate quant
```

### 2. 更新PATH变量（可选）

在 `~/.bashrc` 添加：
```bash
# 自动激活quant环境
# conda activate quant
```

### 3. 使用脚本

创建启动脚本 `start_training.sh`:
```bash
#!/bin/bash
conda activate quant
cd /home/yin/trading/rlbot/ml_project
PYTHONPATH=src python scripts/train_model_enhanced.py
```

## 📊 性能建议

### CPU优化
虽然没有GPU，但可以优化CPU性能：

```python
# 在代码中设置
params = {
    'num_threads': 16,  # 使用更多CPU线程
    'force_col_wise': True,  # 列式训练
    'histogram_pool_size': 1024,  # 增加内存池
}
```

### 并行训练
利用多核CPU并行训练多个模型：

```bash
# 同时训练多个时间周期
make train-5T &
make train-15T &
make train-60T &
wait
```

## 🆘 故障排查

### 问题1：环境找不到

```bash
conda env list  # 查看所有环境
conda activate quant  # 重新激活
```

### 问题2：包导入失败

```bash
conda activate quant
python -c "import lightgbm; print(lightgbm.__version__)"
```

### 问题3：PYTHONPATH问题

```bash
# 确保在项目根目录
cd /home/yin/trading/rlbot/ml_project
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

## 📚 相关文档

- [pyproject.toml](pyproject.toml) - 项目依赖定义
- [Makefile](Makefile) - 构建和训练命令
- [requirements.txt](requirements.txt) - pip依赖（已弃用，使用conda）

---

**环境创建时间**: 2025-10-21
**最后更新**: 2025-10-21
**维护者**: 用户自行维护

