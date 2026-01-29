# 🐳 LightGBM GPU Docker 环境

解决 Windows WSL 中 LightGBM GPU 兼容性问题的 Docker 方案。

## 🚀 快速构建（推荐）

使用自动化构建脚本，自动处理 LightGBM 源码复制和清理：

```bash
# 基本构建
./docker/build-gpu.sh

# 指定镜像名称和标签（GPU 镜像）
./docker/build-gpu.sh -n hansenlovefiona017/lightgbm-runtime -t v0.0.9 --no-ssh --no-proxy

# 构建实盘镜像 (Dockerfile.live，无 GPU/LightGBM)
./docker/build-gpu.sh --live -n quant-engine-live -t v0.0.9 --no-proxy
# 若拉取 ubuntu:22.04 出现 TLS handshake timeout，请去掉 --no-proxy 并设置代理后再构建，或先配置 Docker 守护进程的代理/镜像站

HTTP_PROXY=http://127.0.0.1:7897 \
HTTPS_PROXY=http://127.0.0.1:7897 \
NO_PROXY=localhost,127.0.0.1 \
./docker/build-gpu.sh --live -n quant-engine-live -t v0.0.9
# 使用代理构建
# 注意：脚本会自动修复代理地址（127.0.0.1 -> Docker 网桥 IP，以便容器访问宿主机代理）
HTTP_PROXY=http://127.0.0.1:7897 \
HTTPS_PROXY=http://127.0.0.1:7897 \
NO_PROXY=localhost,127.0.0.1 \
./docker/build-gpu.sh -n hansenlovefiona017/lightgbm-runtime -t v0.0.9 --no-ssh

# Clash TUN 模式用户注意：
# 1. 如果配置了很多 DIRECT 规则（如 docker.io, ubuntu.com 等），可能不需要代理
# 2. 如果需要代理，确保 Clash 配置中允许局域网连接（监听 0.0.0.0:7897）
# 3. 或者直接不设置代理，让 TUN 模式处理流量

# 不使用 SSH（如果不需要更新子模块）
./docker/build-gpu.sh --no-ssh

# 不使用代理（忽略环境变量中的代理设置，适用于 Clash TUN 模式）
./docker/build-gpu.sh --no-proxy

# 同时禁用 SSH 和代理
./docker/build-gpu.sh --no-ssh --no-proxy

# Makefile 便捷目标（使用固定名称/标签）
make docker-build-gpu

# 查看帮助
./docker/build-gpu.sh --help
```

## 📝 手动构建

如果需要手动控制构建过程：

```sh
# 1. 复制 LightGBM 到构建上下文
cp -r /home/yin/trading/LightGBM ./LightGBM

# 2. 构建镜像
DOCKER_BUILDKIT=1 docker build \
  --ssh default \
  -f docker/Dockerfile.gpu \
  --target runtime \
  -t lightgbm-gpu:latest \
  . \
  --build-arg http_proxy=http://host.docker.internal:7897 \
  --build-arg https_proxy=http://host.docker.internal:7897 \
  --build-arg NO_PROXY=localhost,127.0.0.1,archive.ubuntu.com,security.ubuntu.com \
  --build-arg no_proxy=localhost,127.0.0.1,archive.ubuntu.com,security.ubuntu.com

# 3. 清理临时文件
rm -rf ./LightGBM
```

**关于代理变量：**
- `HTTP_PROXY/HTTPS_PROXY/NO_PROXY`（大写）是 POSIX / curl / apt 等多数 C/C++ 程序默认识别的代理变量
- `http_proxy/https_proxy/no_proxy`（小写）是 curl 旧版本和一些 Python/Ruby 生态习惯使用的写法；Git 也支持
- 多数程序会同时查找，优先级通常是小写覆盖大写，所以我们在 Dockerfile 里同时设置，保证无论谁检查哪一种都能拿到值
## 🎯 快速开始

```bash
# 1. 运行测试
make docker-gpu-quickstart

# 2. 查看结果
# 你应该看到 GPU 比 CPU 快 6-8 倍
```

## 📚 文档导航

| 文档                                             | 说明        | 适合人群         |
| ------------------------------------------------ | ----------- | ---------------- |
| [📖 QUICKSTART.md](./QUICKSTART.md)               | 3步快速开始 | 所有人 ⭐         |
| [📘 SETUP_SUMMARY.md](./SETUP_SUMMARY.md)         | 配置总结    | 已配置用户       |
| [📕 README_GPU_DOCKER.md](./README_GPU_DOCKER.md) | 完整文档    | 需要详细配置 ⭐⭐⭐ |
| [📚 INDEX.md](./INDEX.md)                         | 文件索引    | 开发者           |

## 📦 文件列表

- `Dockerfile.gpu` - GPU 镜像定义（Python 3.12，包含 CUDA、PyTorch、LightGBM GPU）
- `Dockerfile.live` - 实盘交易镜像（Python 3.12，CPU 版本，包含 WebSocket、HTTP、特征计算）
- `docker-compose.gpu.yml` - Compose 配置
- `test_gpu_lightgbm.py` - GPU 测试脚本
- `run_gpu_test.ps1` - Windows 运行脚本
- `run_gpu_test.sh` - Linux 运行脚本

## 🚀 使用方法

### GPU 版本（训练/模型开发）

#### Windows (PowerShell)

```powershell
.\docker\run_gpu_test.ps1
```

#### Linux / WSL (Bash)

```bash
bash docker/run_gpu_test.sh
```

#### 使用 Makefile

```bash
make docker-gpu-test       # 运行测试
make docker-gpu-shell      # 交互模式
make docker-gpu-check     # 环境检查
```

### 实盘版本（Live Trading - CPU Only）

使用同一构建脚本加 `--live` 即可构建实盘镜像（不复制 LightGBM/dtaidistance，不校验 GPU）：

```bash
./docker/build-gpu.sh --live -n quant-engine -t v0.0.9 --no-proxy
```

实盘 Dockerfile (`Dockerfile.live`) 专为生产环境设计，不包含 GPU 支持，专注于：
- WebSocket 实时数据接收
- HTTP API 调用
- 特征计算
- 实时交易策略执行

#### 构建实盘镜像

```bash
docker build -f docker/Dockerfile.live \
  -t ml-trading-bot-live:latest \
  . \
  --build-arg http_proxy=http://host.docker.internal:7897 \
  --build-arg https_proxy=http://host.docker.internal:7897 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  --build-arg no_proxy=localhost,127.0.0.1
```

#### 运行实盘容器

```bash
# 交互模式
docker run --rm -it \
  -v $(pwd):/workspace \
  ml-trading-bot-live:latest bash

# 运行实盘策略
docker run --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/config:/workspace/config \
  -v $(pwd)/data:/workspace/data \
  ml-trading-bot-live:latest \
  python3 scripts/your_live_strategy.py
```

#### 实盘镜像特性

- ✅ Python 3.12
- ✅ WebSocket 支持（websockets, ccxt）
- ✅ HTTP 客户端（requests, aiohttp, httpx）
- ✅ 特征计算库（numpy, pandas, scipy, scikit-learn, TA-Lib）
- ✅ LightGBM CPU 版本
- ✅ 数据存储（pyarrow, parquet）
- ❌ 不包含 GPU/CUDA（减小镜像体积）
- ❌ 不包含深度学习框架（PyTorch, mamba-ssm）

## ✅ 解决的问题

- ✅ WSL LightGBM GPU 兼容性
- ✅ CUDA/OpenCL 库冲突
- ✅ 环境配置复杂性
- ✅ 6-8倍训练加速

## 📊 性能对比

| 环境       | 训练时间 | 状态       |
| ---------- | -------- | ---------- |
| WSL Native | ❌ 失败   | GPU 不兼容 |
| Docker GPU | ✅ 2秒    | 正常工作   |
| Docker CPU | 15秒     | 对比基准   |

**加速比: 7.5x** 🚀

## 🔗 相关链接

- [LightGBM GPU](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

## 🔧 故障排除

### Docker 镜像拉取错误

#### 错误：`failed to resolve source metadata` 或 `no such host`

如果遇到以下错误：
```
ERROR: failed to solve: nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04: 
failed to resolve source metadata for docker.io/nvidia/cuda: 
dial tcp: lookup docker.mirrors.ustc.edu.cn on 8.8.8.8:53: no such host
```

**原因：** Docker 配置了镜像加速器但无法访问，导致拉取基础镜像失败。

**快速修复（推荐）：**

1. **临时禁用镜像加速器，使用官方源+代理：**
   ```bash
   # 备份当前配置
   sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.backup
   
   # 编辑配置，移除 registry-mirrors，添加代理配置
   sudo nano /etc/docker/daemon.json
   ```
   
   将配置改为：
   ```json
   {
     "runtimes": {
       "nvidia": {
         "args": [],
         "path": "nvidia-container-runtime"
       }
     },
     "proxies": {
       "http-proxy": "http://host.docker.internal:7897",
       "https-proxy": "http://host.docker.internal:7897",
       "no-proxy": "localhost,127.0.0.1"
     }
   }
   ```
   
   然后重启 Docker：
   ```bash
   sudo systemctl restart docker
   ```

2. **或者直接使用官方源（如果代理已配置在构建参数中）：**
   ```bash
   # 临时移除镜像加速器
   sudo sed -i '/registry-mirrors/d' /etc/docker/daemon.json
   sudo systemctl restart docker
   ```

### TLS Handshake Timeout 错误

如果遇到 `TLS handshake timeout` 错误（如 `failed to resolve source metadata for docker.io/nvidia/cuda`），可以尝试以下解决方案：

**快速诊断：**
```bash
# 运行诊断脚本
bash docker/fix_docker_timeout.sh
```

该脚本会自动检测代理设置并生成正确的构建命令。

#### 方案 1: 使用代理（推荐）

如果你有可用的代理，在构建时传递代理参数：

```bash
docker build -f docker/Dockerfile.gpu \
  --target runtime \
  -t lightgbm-runtime:v0.0.5 \
  . \
  --build-arg HTTP_PROXY=http://host.docker.internal:7897 \
  --build-arg HTTPS_PROXY=http://host.docker.internal:7897 \
  --build-arg http_proxy=http://host.docker.internal:7897 \
  --build-arg https_proxy=http://host.docker.internal:7897 \
  --build-arg NO_PROXY=localhost,127.0.0.1,archive.ubuntu.com,security.ubuntu.com \
  --build-arg no_proxy=localhost,127.0.0.1,archive.ubuntu.com,security.ubuntu.com
```

#### 方案 2: 配置 Docker Daemon 代理

编辑 `/etc/docker/daemon.json`（需要 root 权限）：

```json
{
  "proxies": {
    "http-proxy": "http://host.docker.internal:7897",
    "https-proxy": "http://host.docker.internal:7897",
    "no-proxy": "localhost,127.0.0.1"
  }
}
```

然后重启 Docker：
```bash
sudo systemctl restart docker
```

#### 方案 3: 使用 Docker 镜像加速器

编辑 `/etc/docker/daemon.json`：

```json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ]
}
```

重启 Docker 服务。

#### 方案 4: 验证 CUDA 镜像标签

检查 CUDA 12.8.1 镜像是否存在：

```bash
# 尝试手动拉取镜像
docker pull nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# 如果失败，可以尝试其他版本
docker pull nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04
```

如果标签不存在，需要更新 Dockerfile 中的 CUDA 版本。

#### 方案 5: 增加超时时间

在构建前设置环境变量：

```bash
export DOCKER_CLIENT_TIMEOUT=300
export COMPOSE_HTTP_TIMEOUT=300
```

#### 方案 6: 重试构建

网络问题可能是暂时的，可以多次重试：

```bash
# 重试 3 次
for i in {1..3}; do
  echo "Attempt $i/3..."
  docker build -f docker/Dockerfile.gpu --target runtime -t lightgbm-runtime:v0.0.5 . && break
  sleep 10
done
```

---

**需要帮助?** 查看 [QUICKSTART.md](./QUICKSTART.md) 或 [README_GPU_DOCKER.md](./README_GPU_DOCKER.md)

