# mamba-ssm GPU 问题分析

## 问题描述

在 Docker 环境中，mamba-ssm GPU 功能无法正常工作。从代码中看到：
```
⚠️  Mamba not available, will use Transformer
```

## 可能的原因

### 1. Python 版本兼容性问题 ⚠️ **最可能**

**当前配置：**
- Python 3.12 (Ubuntu 24.04 默认)
- PyTorch 2.9.0 (CUDA 12.8)

**问题：**
- mamba-ssm 及其依赖 `causal-conv1d` 可能不完全支持 Python 3.12
- 官方推荐使用 Python 3.10 或 3.11

**解决方案：**
```dockerfile
# 在 Dockerfile 中明确安装 Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

### 2. CUDA 版本兼容性问题

**当前配置：**
- CUDA 12.8.1
- PyTorch 2.9.0 (cu128)

**问题：**
- mamba-ssm 的预编译包可能不支持 CUDA 12.8
- `causal-conv1d` 需要特定 CUDA 版本支持

**解决方案：**
- 降级到 CUDA 11.8 或 12.1（更稳定的版本）
- 或者强制从源码编译

### 3. 编译环境问题

**当前配置：**
- Builder 阶段有编译工具
- Runtime 阶段没有编译工具

**问题：**
- mamba-ssm 的 wheel 可能在 builder 阶段编译失败
- 或者编译的 wheel 与 runtime 环境不兼容

**解决方案：**
```dockerfile
# 在 builder 阶段强制从源码编译
RUN echo "🔨 Pre-building mamba-ssm wheel (this will take 30-60 minutes)..." && \
    export MAMBA_FORCE_BUILD=TRUE && \
    export CAUSAL_CONV1D_FORCE_BUILD=TRUE && \
    pip3 wheel \
    --no-cache-dir \
    --timeout=3600 \
    --wheel-dir /wheelhouse \
    mamba-ssm>=1.0.0
```

### 4. GPU 架构支持问题

**问题：**
- mamba-ssm 的预编译包可能不支持某些 GPU 架构
- 需要检查 GPU 架构（如 sm_75, sm_80, sm_86, sm_89 等）

**解决方案：**
```dockerfile
# 在编译时指定 GPU 架构
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
```

### 5. Docker GPU 配置问题

**问题：**
- Docker 容器可能没有正确配置 GPU 访问
- 运行时无法检测到 GPU

**解决方案：**
- 确保使用 `--gpus all` 参数
- 确保安装了 NVIDIA Container Toolkit

## 推荐的修复方案

### 方案 1: 降级 Python 版本（推荐）

```dockerfile
# 在 builder 阶段
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

# 安装 Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
```

### 方案 2: 强制从源码编译

```dockerfile
# 在 builder 阶段
RUN echo "🔨 Pre-building mamba-ssm wheel (this will take 30-60 minutes)..." && \
    export MAMBA_FORCE_BUILD=TRUE && \
    export CAUSAL_CONV1D_FORCE_BUILD=TRUE && \
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" && \
    pip3 wheel \
    --no-cache-dir \
    --timeout=3600 \
    --wheel-dir /wheelhouse \
    mamba-ssm>=1.0.0 && \
    ls -lh /wheelhouse/*.whl
```

### 方案 3: 使用预编译的兼容版本

```dockerfile
# 如果 PyPI 有预编译的 wheel，直接安装
RUN pip3 install --no-cache-dir \
    causal-conv1d>=1.2.0 \
    mamba-ssm>=1.0.0
```

## 诊断步骤

1. **检查 Python 版本：**
   ```bash
   docker run --gpus all <image> python3 --version
   ```

2. **检查 PyTorch CUDA 支持：**
   ```bash
   docker run --gpus all <image> python3 -c "import torch; print(torch.cuda.is_available())"
   ```

3. **尝试导入 mamba-ssm：**
   ```bash
   docker run --gpus all <image> python3 -c "from mamba_ssm import Mamba; print('OK')"
   ```

4. **检查编译日志：**
   - 查看 Docker 构建日志，确认 mamba-ssm wheel 是否成功编译
   - 检查是否有 CUDA 编译错误

## 结论

**最可能的原因是 Python 3.12 兼容性问题。** 建议：
1. 降级到 Python 3.11
2. 或者强制从源码编译 mamba-ssm
3. 确保 CUDA 版本兼容

