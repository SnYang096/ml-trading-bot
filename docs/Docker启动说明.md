# Docker 启动说明

## ✅ Docker 已成功启动

Docker 已通过 `sudo service docker start` 启动。

## 启动脚本

已创建启动脚本：`scripts/start_docker.sh`

### 使用方法

```bash
# 方法 1: 直接运行脚本
bash scripts/start_docker.sh

# 方法 2: 使用 Makefile
make start-docker

# 方法 3: 手动启动（WSL2）
sudo service docker start
```

### 脚本功能

脚本会自动：
1. 检查 Docker 是否已运行
2. 如果未运行，尝试多种方法启动：
   - `service docker start` (WSL2)
   - `systemctl start docker` (systemd)
   - 直接启动 `dockerd` (如果可用)
   - 连接 Docker Desktop (WSL2)

## Makefile 自动启动

现在所有需要 Docker 的命令都会自动检查并启动 Docker：

- `make test-complex-features-comprehensive` - 自动启动 Docker
- `make test-key-features-all` - 自动启动 Docker
- `make ts-sr-reversal-model-comparison` - 自动启动 Docker

## 验证 Docker 状态

```bash
# 检查 Docker 是否运行
docker ps

# 查看 Docker 版本
docker --version

# 查看 Docker 信息
docker info
```

## 故障排除

### 问题 1: Permission denied

```bash
# 将用户添加到 docker 组
sudo usermod -aG docker $USER
# 然后重新登录或执行
newgrp docker
```

### 问题 2: WSL2 + Docker Desktop

如果使用 WSL2 和 Docker Desktop：
1. 在 Windows 上打开 Docker Desktop
2. 确保 "Use the WSL 2 based engine" 已启用
3. 在 Docker Desktop Settings > Resources > WSL Integration 中启用你的 WSL 发行版

### 问题 3: Docker daemon 未运行

```bash
# 检查 dockerd 进程
ps aux | grep dockerd

# 手动启动（如果需要）
sudo dockerd &
```

## 测试结果

✅ **复杂特征测试**: 9 passed, 477 warnings (主要是 GARCH 数据缩放警告，不影响功能)

测试已成功运行！

