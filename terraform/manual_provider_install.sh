#!/bin/bash
# 手动安装 Terraform TencentCloud Provider
# 用于解决网络问题导致的自动下载失败

set -e

PROVIDER_VERSION="1.80.0"
PROVIDER_NAME="tencentcloudstack/tencentcloud"
ARCH="linux_amd64"  # 根据你的系统调整：linux_amd64, darwin_amd64, windows_amd64

# 检测系统架构
if [[ "$(uname -m)" == "x86_64" ]]; then
    ARCH="linux_amd64"
elif [[ "$(uname -m)" == "arm64" ]] || [[ "$(uname -m)" == "aarch64" ]]; then
    ARCH="linux_arm64"
fi

echo "=== 手动安装 Terraform Provider ==="
echo "Provider: $PROVIDER_NAME"
echo "Version: $PROVIDER_VERSION"
echo "Architecture: $ARCH"

# 创建 provider 目录
PLUGIN_DIR="$HOME/.terraform.d/plugins/registry.terraform.io/${PROVIDER_NAME}/${PROVIDER_VERSION}/${ARCH}"
mkdir -p "$PLUGIN_DIR"

# 下载 provider
DOWNLOAD_URL="https://github.com/tencentcloudstack/terraform-provider-tencentcloud/releases/download/v${PROVIDER_VERSION}/terraform-provider-tencentcloud_${PROVIDER_VERSION}_${ARCH}.zip"
ZIP_FILE="/tmp/terraform-provider-tencentcloud_${PROVIDER_VERSION}_${ARCH}.zip"

echo "下载 provider..."
if command -v curl &> /dev/null; then
    curl -L -o "$ZIP_FILE" "$DOWNLOAD_URL" || {
        echo "下载失败，尝试使用代理..."
        curl -L -x http://127.0.0.1:7897 -o "$ZIP_FILE" "$DOWNLOAD_URL"
    }
elif command -v wget &> /dev/null; then
    wget -O "$ZIP_FILE" "$DOWNLOAD_URL" || {
        echo "下载失败，尝试使用代理..."
        wget -e http_proxy=http://127.0.0.1:7897 -O "$ZIP_FILE" "$DOWNLOAD_URL"
    }
else
    echo "错误: 需要 curl 或 wget"
    exit 1
fi

# 解压
echo "解压 provider..."
unzip -o "$ZIP_FILE" -d "$PLUGIN_DIR"

# 设置执行权限
chmod +x "$PLUGIN_DIR"/terraform-provider-tencentcloud_*

echo "✅ Provider 安装完成: $PLUGIN_DIR"
echo ""
echo "现在可以运行: terraform init"
