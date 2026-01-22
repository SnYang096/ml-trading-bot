#!/bin/bash
# 快速修复 Git 大文件问题
# 从 Git 历史中移除 terraform provider 二进制文件

set -e

echo "⚠️  警告: 此脚本将从 Git 历史中移除大文件"
echo "   这将会重写 Git 历史，如果有其他人在使用这个仓库，需要通知他们"
echo ""
read -p "确认继续？(输入 'yes' 继续): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ 已取消"
    exit 0
fi

echo ""
echo "📦 步骤 1: 备份当前分支..."
git branch backup-before-cleanup-$(date +%Y%m%d-%H%M%S) || true

echo ""
echo "🧹 步骤 2: 从 Git 历史中移除大文件..."
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch -r terraform/terraform-provider-tencentcloud_1.82.58_linux_amd64/" \
  --prune-empty --tag-name-filter cat -- --all

echo ""
echo "🧹 步骤 3: 清理引用..."
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d 2>/dev/null || true

echo ""
echo "🗑️  步骤 4: 强制垃圾回收..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "✅ 清理完成！"
echo ""
echo "📝 下一步操作："
echo "   1. 检查清理结果: git log --oneline"
echo "   2. 强制推送到远程: git push origin --force --all"
echo "   3. 如果有标签: git push origin --force --tags"
echo ""
echo "⚠️  注意: 强制推送会重写远程历史，其他协作者需要重新克隆或执行:"
echo "   git fetch --all && git reset --hard origin/main"
