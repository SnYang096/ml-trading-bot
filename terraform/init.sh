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

# 2. 安装必要软件
apt-get update

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

# 8. 启用所有服务
systemctl daemon-reload
systemctl enable quant-engine node_exporter prometheus grafana-server filebeat
systemctl start quant-engine
