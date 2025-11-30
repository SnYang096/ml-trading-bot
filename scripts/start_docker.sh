#!/bin/bash
# Docker 启动脚本（适用于 WSL2 和其他环境）

set -e

echo "🔧 检查 Docker 状态..."

# 检查 Docker 是否已安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查 Docker 是否已运行
if docker ps &> /dev/null; then
    echo "✅ Docker 已在运行"
    docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}"
    exit 0
fi

echo "⚠️  Docker 未运行，尝试启动..."

# 方法 1: 使用 service 命令（适用于 WSL2）
if command -v service &> /dev/null; then
    echo "📦 使用 service 命令启动 Docker..."
    sudo service docker start
    sleep 3
    
    if docker ps &> /dev/null; then
        echo "✅ Docker 启动成功"
        docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}"
        exit 0
    fi
fi

# 方法 2: 使用 systemctl（如果可用）
if command -v systemctl &> /dev/null && systemctl is-system-running &> /dev/null; then
    echo "📦 使用 systemctl 启动 Docker..."
    sudo systemctl start docker
    sleep 3
    
    if docker ps &> /dev/null; then
        echo "✅ Docker 启动成功"
        docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}"
        exit 0
    fi
fi

# 方法 3: 直接启动 dockerd（如果可用）
if command -v dockerd &> /dev/null; then
    echo "📦 尝试直接启动 dockerd..."
    echo "⚠️  注意：这将在后台启动 dockerd 进程"
    
    # 检查是否已有 dockerd 进程
    if pgrep -f dockerd > /dev/null; then
        echo "⚠️  dockerd 进程已存在，但 Docker 无法连接"
        echo "   可能需要检查 /var/run/docker.sock 的权限"
    else
        echo "   启动 dockerd（需要 root 权限）..."
        sudo dockerd &> /tmp/dockerd.log &
        sleep 5
        
        if docker ps &> /dev/null; then
            echo "✅ Docker 启动成功"
            docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}"
            exit 0
        else
            echo "❌ dockerd 启动失败，查看日志: /tmp/dockerd.log"
        fi
    fi
fi

# 方法 4: WSL2 中使用 Docker Desktop
if [ -f /mnt/wsl/docker-desktop/docker.sock ] || [ -f /var/run/docker-desktop/docker.sock ]; then
    echo "📦 检测到 Docker Desktop，尝试连接..."
    
    # 设置 DOCKER_HOST（如果需要）
    export DOCKER_HOST=unix:///var/run/docker-desktop/docker.sock 2>/dev/null || \
    export DOCKER_HOST=unix:///mnt/wsl/docker-desktop/docker.sock 2>/dev/null || true
    
    if docker ps &> /dev/null; then
        echo "✅ Docker Desktop 连接成功"
        docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}"
        exit 0
    fi
fi

# 如果所有方法都失败
echo ""
echo "❌ 无法启动 Docker"
echo ""
echo "可能的解决方案："
echo "1. 在 Windows 上启动 Docker Desktop（如果使用 WSL2）"
echo "2. 手动启动 Docker: sudo dockerd"
echo "3. 检查 Docker 安装: docker --version"
echo "4. 检查权限: ls -la /var/run/docker.sock"
echo ""
echo "对于 WSL2 + Docker Desktop:"
echo "  1. 在 Windows 上打开 Docker Desktop"
echo "  2. 确保 'Use the WSL 2 based engine' 已启用"
echo "  3. 在 Docker Desktop Settings > Resources > WSL Integration 中启用你的 WSL 发行版"
echo ""

exit 1

