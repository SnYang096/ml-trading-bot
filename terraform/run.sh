#!/bin/bash
# Terraform 运行脚本
# 自动加载环境变量并运行 Terraform 命令

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 加载环境变量
echo "📦 加载环境变量..."
source "$PROJECT_ROOT/config/local/qclould.env"
source "$PROJECT_ROOT/config/local/terraform.env"

# 验证环境变量
if [ -z "$TENCENTCLOUD_SECRET_ID" ] || [ -z "$TENCENTCLOUD_SECRET_KEY" ]; then
    echo "❌ 错误: TENCENTCLOUD_SECRET_ID 或 TENCENTCLOUD_SECRET_KEY 未设置"
    echo "请确保 config/local/qclould.env 文件存在且包含正确的凭证"
    exit 1
fi

echo "✅ 环境变量已加载"
echo "   TENCENTCLOUD_SECRET_ID: ${TENCENTCLOUD_SECRET_ID:0:20}..."
echo "   TENCENTCLOUD_SECRET_KEY: ${TENCENTCLOUD_SECRET_KEY:0:20}..."

# 导出环境变量，确保子进程可以访问
export TENCENTCLOUD_SECRET_ID
export TENCENTCLOUD_SECRET_KEY

# 同时通过 Terraform 变量传递（作为备选方案）
export TF_VAR_secret_id="$TENCENTCLOUD_SECRET_ID"
export TF_VAR_secret_key="$TENCENTCLOUD_SECRET_KEY"

# 切换到 terraform 目录
cd "$SCRIPT_DIR"

# 运行 Terraform 命令
echo ""
if [ "$1" = "destroy" ]; then
    echo "⚠️  警告: 即将删除所有云资源！"
    echo "   这将删除 CVM 实例、VPC、安全组等所有资源。"
    echo "   请确保已备份重要数据！"
    echo ""
    echo "   提示: 使用 'plan -destroy' 可以预览删除计划"
    echo ""
    read -p "   确认继续？(输入 'yes' 继续): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "❌ 已取消删除操作"
        exit 0
    fi
    echo ""
fi

echo "🚀 运行 Terraform: $@"
echo ""

terraform "$@"
