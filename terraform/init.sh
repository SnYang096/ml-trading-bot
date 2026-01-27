#!/bin/bash
set -e

echo "=== Deploying quant system ==="

# 0. 设置 SSH 公钥（允许 Terraform provisioner 连接）
mkdir -p /home/ubuntu/.ssh
chmod 700 /home/ubuntu/.ssh
# 公钥通过 user_data 模板变量传入
echo "${ssh_public_key}" >> /home/ubuntu/.ssh/authorized_keys
chmod 600 /home/ubuntu/.ssh/authorized_keys
chown -R ubuntu:ubuntu /home/ubuntu/.ssh

# 1. 挂载数据盘
mkfs.ext4 /dev/vdb
mkdir -p /data
mount /dev/vdb /data
echo '/dev/vdb /data ext4 defaults 0 0' >> /etc/fstab

# 2. 安装必要软件（包括 Docker）
apt-get update
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 安装 Docker
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    # 添加 Docker 官方 GPG key
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    
    # 添加 Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # 将 ubuntu 用户添加到 docker 组（允许无 sudo 运行 docker）
    usermod -aG docker ubuntu
    
    # 启动 Docker 服务
    systemctl enable docker
    systemctl start docker
    
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# 3. 复制 systemd 服务文件（从 Terraform 传入）
cp /tmp/systemd/*.service /etc/systemd/system/

# 4. 复制监控配置
mkdir -p /etc/prometheus
cp /tmp/monitoring/prometheus.yml /etc/prometheus/

mkdir -p /var/lib/grafana/dashboards
cp /tmp/monitoring/grafana-provisioning/dashboards/quant.json /var/lib/grafana/dashboards/

# 5. 复制日志配置
mkdir -p /etc/filebeat
cp /tmp/logging/filebeat.yml /etc/filebeat/
# 注意：/etc/default/filebeat 需在 Terraform 中通过 template 生成（含密钥）

# 6. 安装组件
# --- Node Exporter ---
wget -q https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-amd64.tar.gz
tar xzf node_exporter-*.tar.gz
cp node_exporter-*/node_exporter /usr/local/bin/

# --- Prometheus ---
wget -q https://github.com/prometheus/prometheus/releases/download/v2.47.1/prometheus-2.47.1.linux-amd64.tar.gz
tar xzf prometheus-*.tar.gz
cp prometheus-*/prometheus /usr/local/bin/

# --- Grafana ---
wget -q https://dl.grafana.com/oss/release/grafana_10.3.3_amd64.deb
dpkg -i grafana_10.3.3_amd64.deb

# --- Filebeat ---
wget -q https://artifacts.elastic.co/downloads/beats/filebeat/filebeat-8.12.0-amd64.deb
dpkg -i filebeat-8.12.0-amd64.deb

# 7. 创建数据目录
mkdir -p /data/prometheus
mkdir -p /data/trades  # SQLite 数据库目录（订单流数据）
mkdir -p /data/quant-engine  # 策略代码和数据目录

# 8. 构建 Docker 镜像（如果 Dockerfile 存在）
# 注意：代码应该通过 volume 挂载，镜像只包含依赖
if [ -f /opt/quant-engine/Dockerfile.live ]; then
    echo "📦 Building quant-engine Docker image..."
    cd /opt/quant-engine
    docker build -f Dockerfile.live -t quant-engine:latest .
    echo "✅ Docker image built"
else
    echo "⚠️  Dockerfile.live not found, skipping image build"
    echo "    You can build the image manually or copy it from registry"
fi

# 9. 启用所有服务
systemctl daemon-reload
systemctl enable quant-engine node_exporter prometheus grafana-server filebeat
systemctl start quant-engine
