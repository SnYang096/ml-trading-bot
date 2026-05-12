#!/bin/bash
# =============================================================
# 监控栈一次性初始化脚本（Prometheus + Grafana Docker 容器，幂等）
# =============================================================
# 用法（两步）:
#   1. 同步配置文件到服务器:
#      rsync -avz -e "ssh -i ~/.ssh/id_tencent_cloud_ssh" \
#        deploy/monitoring/ ubuntu@43.135.44.160:/opt/monitoring/
#
#   2. 执行此脚本:
#      ssh -i ~/.ssh/id_tencent_cloud_ssh ubuntu@43.135.44.160 \
#        'sudo bash -s' < scripts/monitoring_bootstrap.sh
#
# 前提: server_bootstrap.sh 已执行（Docker 已安装）
# =============================================================

set -euo pipefail

MONITORING_PATH="${MONITORING_PATH:-/opt/monitoring}"

echo "============================================================"
echo "📊 Monitoring Bootstrap — Prometheus + Grafana (Docker)"
echo "============================================================"
echo "Monitoring path : $MONITORING_PATH"
echo "Time            : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# ---- 1. 检查 Docker ----
echo "🐳 [1/4] 检查 Docker..."
if ! command -v docker &>/dev/null; then
    echo "   ❌ Docker 未安装，请先运行 server_bootstrap.sh"
    exit 1
fi
echo "   ✅ Docker $(docker --version | grep -oP '[\d.]+')"

# ---- 2. 检查配置文件 ----
echo "📁 [2/4] 检查配置文件..."
REQUIRED_FILES=(
    "$MONITORING_PATH/docker-compose.monitoring.yml"
    "$MONITORING_PATH/prometheus.yml"
    "$MONITORING_PATH/grafana-provisioning/datasources/prometheus.yml"
    "$MONITORING_PATH/grafana-provisioning/dashboards/dashboard.yml"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "   ❌ 缺少: $f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "   请先同步配置文件:"
    echo "   rsync -avz -e 'ssh -i ~/.ssh/id_tencent_cloud_ssh' \\"
    echo "     deploy/monitoring/ ubuntu@<IP>:$MONITORING_PATH/"
    exit 1
fi
echo "   ✅ 配置文件就绪"

# ---- 3. 适配 prometheus.yml 为服务器环境 ----
echo "⚙️  [3/4] 适配 Prometheus 配置..."

# 服务器上 quant-engine 用 --network host，metrics 在 host:9090
# Prometheus 容器通过 host.docker.internal 访问（docker-compose 已配置 extra_hosts）
# 检查 prometheus.yml 中 target 是否正确
if grep -q "host.docker.internal:9090" "$MONITORING_PATH/prometheus.yml"; then
    echo "   ✅ Prometheus target 已指向 host.docker.internal:9090"
else
    echo "   ⚠️  更新 Prometheus target..."
    sed -i 's|targets:.*|targets: ["host.docker.internal:9090"]|' "$MONITORING_PATH/prometheus.yml"
    echo "   ✅ 已更新为 host.docker.internal:9090"
fi

# ---- 4. 启动监控容器 ----
echo "🚀 [4/4] 启动监控容器..."

cd "$MONITORING_PATH"

# 停掉旧容器（如果有）
docker compose -f docker-compose.monitoring.yml down 2>/dev/null || true

# 拉取镜像并启动
docker compose -f docker-compose.monitoring.yml pull
docker compose -f docker-compose.monitoring.yml up -d

# 等待启动
echo "   ⏳ 等待服务启动..."
sleep 10

# 健康检查
echo ""
echo "   📊 服务状态:"
if docker ps --filter name=mlbot-prometheus --format '{{.Status}}' | grep -q "Up"; then
    echo "   ✅ Prometheus — http://<SERVER_IP>:9091"
else
    echo "   ❌ Prometheus 启动失败"
    docker logs mlbot-prometheus --tail 10
fi

if docker ps --filter name=mlbot-grafana --format '{{.Status}}' | grep -q "Up"; then
    echo "   ✅ Grafana    — http://<SERVER_IP>:3000 (admin/admin)"
else
    echo "   ❌ Grafana 启动失败"
    docker logs mlbot-grafana --tail 10
fi

# ---- 完成 ----
echo ""
echo "============================================================"
echo "✅ Monitoring Bootstrap 完成！"
echo "============================================================"
echo ""
echo "📋 你还需要做:"
echo ""
echo "  1. 腾讯云安全组放行端口:"
echo "     - 9091/tcp (Prometheus) — 限制为你的 IP"
echo "     - 3000/tcp (Grafana)    — 限制为你的 IP"
echo ""
echo "  2. 首次登录 Grafana:"
echo "     - 地址: http://<SERVER_IP>:3000"
echo "     - 账号: admin / admin（首次登录会要求改密码）"
echo "     - Dashboard 已自动加载（quant.json + account_market.json + signal_pipeline.json）"
echo ""
echo "📋 常用命令:"
echo "  cd $MONITORING_PATH"
echo "  docker compose -f docker-compose.monitoring.yml logs -f    # 实时日志"
echo "  docker compose -f docker-compose.monitoring.yml restart    # 重启"
echo "  docker compose -f docker-compose.monitoring.yml down       # 停止"
echo ""
echo "📋 内存占用:"
echo "  docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}' mlbot-prometheus mlbot-grafana"
echo ""
