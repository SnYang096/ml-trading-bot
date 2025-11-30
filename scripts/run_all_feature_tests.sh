#!/bin/bash
# 运行所有特征测试的脚本

set -e

echo "=" | head -c 70; echo ""
echo "运行所有特征测试"
echo "=" | head -c 70; echo ""

# 检查 Docker 是否运行
if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker 未运行，请先启动 Docker"
    exit 1
fi

# 运行关键特征测试
echo ""
echo "1️⃣ 运行关键特征测试 (VPIN, WPT, Volume Profile Volatility)..."
make test-key-features-all || {
    echo "⚠️  关键特征测试有失败，继续运行其他测试..."
}

# 运行复杂特征测试
echo ""
echo "2️⃣ 运行复杂特征测试 (GARCH, EVT, Hurst, Spectrum, DTW, Extended Volatility)..."
make test-complex-features-comprehensive || {
    echo "⚠️  复杂特征测试有失败，请检查日志..."
}

echo ""
echo "=" | head -c 70; echo ""
echo "✅ 所有测试完成"
echo "=" | head -c 70; echo ""

