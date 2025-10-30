# 🐳 LightGBM GPU Docker 环境

解决 Windows WSL 中 LightGBM GPU 兼容性问题的 Docker 方案。

```sh

# HTTP_PROXY/HTTPS_PROXY/NO_PROXY（大写）是 POSIX / curl / apt 等多数 C/C++ 程序默认识别的代理变量；只要你在环境里 export，它们就会走代理。apt-get、wget、curl 都看这几个。
# http_proxy/https_proxy/no_proxy（小写）是 curl 旧版本和一些 Python/Ruby 生态习惯使用的写法；Git 也支持，看的是小写版本。
# 多数程序会同时查找，优先级通常是小写覆盖大写，所以我们在 Dockerfile 里同时设置，保证无论谁检查哪一种都能拿到值。
# 如果只给小写（http_proxy）赋值，apt-get 这类工具不会走代理；只给大写赋值，Git 可能不生效。因此在需要代理的场景，最好两个都设。

docker build -f docker/Dockerfile.gpu  --target runtime -t hansenlovefiona017/lightgbm-runtime:v0.0.2 . --build-arg HTTP_PROXY= --build-arg HTTPS_PROXY= --build-arg http_proxy=http://host.docker.internal:7899 --build-arg https_proxy=http://host.docker.internal:7899 --build-arg NO_PROXY=localhost,127.0.0.1,archive.ubuntu.com,security.ubuntu.com --build-arg no_proxy=localhost,127.0.0.1,archive.ubuntu.com,security.ubuntu.com

docker build -f ml_project/docker/Dockerfile.gpu --target builder -t lightgbm-builder .  --build-arg HTTP_PROXY= --build-arg HTTPS_PROXY= --build-arg http_proxy=http://host.docker.internal:7899 --build-arg https_proxy=http://host.docker.internal:7899 --build-arg NO_PROXY=localhost,127.0.0.1,archive.ubuntu.com,security.ubuntu.com --build-arg no_proxy=localhost,127.0.0.1,archive.ubuntu.com,security.ubuntu.com

docker run --rm -it lightgbm-builder bash -lc "ls /lightgbm/python-package/"

docker run --rm -it lightgbm-builder bash

docker run --rm -it hansenlovefiona017/lightgbm-runtime:v0.0.2 bash

docker commit -m "安装seaborn" -a "hansen" fe05bbe8ebf2 hansenlovefiona017/lightgbm-runtime:v0.0.2

docker run --rm lightgbm-runtime python3 -c "import torch; import pandas as pd; import numpy as np; print('✅ PyTorch:', torch.__version__); print('✅ Pandas:', pd.__version__); print('✅ NumPy:', np.__version__); print('✅ CUDA available:', torch.cuda.is_available())"

docker run --rm hansenlovefiona017/lightgbm-runtime:v0.0.2 pip3 show mamba-ssm

docker run --rm -it --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    hansenlovefiona017/lightgbm-runtime:v0.0.2 \
    python3 -c "import torch; print(torch.__version__); import mamba_ssm; from mamba_ssm import Mamba; print('Mamba OK')"
```
## 🎯 快速开始

```bash
# 1. 运行测试
make docker-gpu-quickstart

# 2. 查看结果
# 你应该看到 GPU 比 CPU 快 6-8 倍
```

## 📚 文档导航

| 文档 | 说明 | 适合人群 |
|-----|------|---------|
| [📖 QUICKSTART.md](./QUICKSTART.md) | 3步快速开始 | 所有人 ⭐ |
| [📘 SETUP_SUMMARY.md](./SETUP_SUMMARY.md) | 配置总结 | 已配置用户 |
| [📕 README_GPU_DOCKER.md](./README_GPU_DOCKER.md) | 完整文档 | 需要详细配置 ⭐⭐⭐ |
| [📚 INDEX.md](./INDEX.md) | 文件索引 | 开发者 |

## 📦 文件列表

- `Dockerfile.gpu` - GPU 镜像定义
- `docker-compose.gpu.yml` - Compose 配置
- `test_gpu_lightgbm.py` - GPU 测试脚本
- `run_gpu_test.ps1` - Windows 运行脚本
- `run_gpu_test.sh` - Linux 运行脚本

## 🚀 使用方法

### Windows (PowerShell)

```powershell
.\docker\run_gpu_test.ps1
```

### Linux / WSL (Bash)

```bash
bash docker/run_gpu_test.sh
```

### 使用 Makefile

```bash
make docker-gpu-test       # 运行测试
make docker-gpu-shell      # 交互模式
make docker-gpu-check      # 环境检查
```

## ✅ 解决的问题

- ✅ WSL LightGBM GPU 兼容性
- ✅ CUDA/OpenCL 库冲突
- ✅ 环境配置复杂性
- ✅ 6-8倍训练加速

## 📊 性能对比

| 环境 | 训练时间 | 状态 |
|-----|---------|------|
| WSL Native | ❌ 失败 | GPU 不兼容 |
| Docker GPU | ✅ 2秒 | 正常工作 |
| Docker CPU | 15秒 | 对比基准 |

**加速比: 7.5x** 🚀

## 🔗 相关链接

- [LightGBM GPU](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

---

**需要帮助?** 查看 [QUICKSTART.md](./QUICKSTART.md) 或 [README_GPU_DOCKER.md](./README_GPU_DOCKER.md)

