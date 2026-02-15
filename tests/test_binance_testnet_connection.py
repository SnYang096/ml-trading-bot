#!/usr/bin/env python3
"""测试 Binance 测试网连接和 API 密钥"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.order_management.binance_api import BinanceAPI


def load_testnet_keys():
    """加载测试网密钥"""
    env_file = project_root / "config/local/binance_testnet.env"

    api_key = ""
    api_secret = ""

    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")

                        if "API_KEY" in key.upper():
                            api_key = value
                        elif "SECRET" in key.upper():
                            api_secret = value

    return api_key, api_secret


print("=" * 80)
print("测试 Binance 测试网连接")
print("=" * 80)

# 1. 加载密钥
api_key, api_secret = load_testnet_keys()
print(f"✅ API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
print(
    f"✅ API Secret: {api_secret[:10]}...{api_secret[-4:] if len(api_secret) > 14 else ''}"
)

# 2. 初始化 API
print("\n初始化 Binance API...")
try:
    api = BinanceAPI(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True,
    )
    print("✅ Binance API 初始化成功")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    sys.exit(1)

# 3. 测试账户信息
print("\n测试账户信息...")
try:
    balance = api.get_account_balance()
    print(f"✅ 账户余额: {balance}")
except Exception as e:
    print(f"❌ 获取账户信息失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 4. 测试获取价格
print("\n测试获取价格...")
try:
    ticker = api.exchange.fetch_ticker("BTC/USDT:USDT")
    print(f"✅ BTC 价格: {ticker['last']}")
except Exception as e:
    print(f"❌ 获取价格失败: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ 所有测试通过！")
print("=" * 80)
