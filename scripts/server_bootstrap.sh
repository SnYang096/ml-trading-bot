#!/bin/bash
# =============================================================
# 服务器一次性初始化脚本（Docker 模式，幂等）
# =============================================================
# 用法（从本地执行）:
#   ssh root@<SERVER_IP> 'bash -s' < scripts/server_bootstrap.sh
#
# 执行完毕后，GitHub Actions CI/CD 即可自动部署镜像
# 服务器只需要: Docker + 数据目录 + API 密钥
# =============================================================

set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-/opt/quant-engine}"

echo "============================================================"
echo "🚀 Server Bootstrap — Quant Engine (Docker Mode)"
echo "============================================================"
echo "Deploy path : $DEPLOY_PATH"
echo "Time        : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# ---- 1. Docker ----
echo "🐳 [1/4] 检查 Docker..."
if command -v docker &>/dev/null; then
    echo "   ✅ Docker $(docker --version | grep -oP '[\d.]+')"
else
    echo "   📥 安装 Docker..."
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg lsb-release > /dev/null 2>&1

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin > /dev/null 2>&1

    systemctl enable docker
    systemctl start docker
    echo "   ✅ Docker 安装完成"
fi

# ---- 2. 数据目录（持久化，不随镜像更新丢失） ----
echo "📁 [2/4] 创建数据目录..."
mkdir -p "$DEPLOY_PATH/live/highcap/data/db"
mkdir -p "$DEPLOY_PATH/live/highcap/data/ticks"
mkdir -p "$DEPLOY_PATH/live/highcap/data/features_15min"
mkdir -p "$DEPLOY_PATH/live/highcap/data/features_4h"
echo "   ✅ 数据目录就绪"

# ---- 3. systemd 服务 ----
echo "⚙️  [3/4] 配置 systemd 服务..."
cat > /etc/systemd/system/quant-engine.service << 'SYSTEMD_EOF'
[Unit]
Description=Quant Engine - Three Strategy Live Trading (Docker)
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=simple
TimeoutStartSec=120

# 清理残留容器
ExecStartPre=-/usr/bin/docker rm -f quant-engine

# 运行容器
ExecStart=/usr/bin/docker run \
    --name quant-engine \
    --network host \
    --memory=3g \
    --cpus=1.8 \
    -e MLBOT_LIVE_TRADE_SIZE=0.0 \
    -e MLBOT_ORDER_MODE=test \
    -e MLBOT_ORDER_MANAGER_ENABLED=true \
    -e PYTHONUNBUFFERED=1 \
    -v /opt/quant-engine/live/highcap/data:/app/live/highcap/data \
    -v /opt/quant-engine/live/binance_mainnet.env:/app/live/binance_mainnet.env:ro \
    quant-engine:latest

# 停止容器
ExecStop=/usr/bin/docker stop -t 30 quant-engine
ExecStopPost=-/usr/bin/docker rm -f quant-engine

# 日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=quant-engine

# 崩溃自动重启
Restart=on-failure
RestartSec=30
StartLimitIntervalSec=600
StartLimitBurst=5

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF

systemctl daemon-reload
systemctl enable quant-engine
echo "   ✅ quant-engine.service 已配置"

# ---- 4. 防火墙基本配置 ----
echo "🔒 [4/4] 安全检查..."
if command -v ufw &>/dev/null; then
    ufw allow 22/tcp comment "SSH" 2>/dev/null || true
    echo "   ✅ SSH 端口已放行"
fi

# ---- 完成 ----
echo ""
echo "============================================================"
echo "✅ Bootstrap 完成！"
echo "============================================================"
echo ""
echo "📋 你还需要做:"
echo ""
echo "  1. 配置 GitHub Secrets（共 6 个）:"
echo "     DEPLOY_HOST       = <服务器公网IP>"
echo "     DEPLOY_USER       = root"
echo "     DEPLOY_SSH_KEY    = <SSH 私钥内容>"
echo "     GHCR_TOKEN        = <GitHub PAT, 需 read:packages + write:packages>"
echo "     BINANCE_API_KEY   = <Binance API Key>"
echo "     BINANCE_API_SECRET = <Binance API Secret>"
echo ""
echo "  2. 准备 warmup 数据（首次部署镜像后，约需 5-10 分钟）:"
echo "     docker run --rm -v $DEPLOY_PATH/live/highcap/data:/app/live/highcap/data \\"
echo "       quant-engine:latest bash live/scripts/prepare_warmup_ticks.sh highcap 6"
echo ""
echo "  3. 首次手动部署（GitHub Actions 页面点 'Run workflow'）"
echo "     或 push 代码到 main 分支自动触发"
echo ""
echo "📋 常用命令:"
echo "  sudo systemctl start quant-engine    # 启动"
echo "  sudo systemctl stop quant-engine     # 停止"
echo "  sudo journalctl -u quant-engine -f   # 实时日志"
echo "  docker images quant-engine           # 查看镜像"
echo ""
