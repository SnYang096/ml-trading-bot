#!/usr/bin/env python3
"""
快速检查币安账户的 SL/TP 保护单状态

使用方法：
1. 设置环境变量：
   export MULTI_LEG_BINANCE_FUTURES_API_KEY="your_key"
   export MULTI_LEG_BINANCE_FUTURES_API_SECRET="your_secret"

2. 运行脚本：
   python scripts/check_sl_protection.py
"""

import os
import sys
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, "/home/yin/trading/ml_trading_bot")

from mlbot_console.services.exchange_balances import (
    _fetch_open_orders_raw,
    _fetch_futures_account_raw,
    _env_first,
    _SCOPE_META,
)


def main():
    # 获取 Multi-leg API 密钥
    meta = _SCOPE_META["multi_leg"]
    api_key = _env_first(*meta["key_envs"])
    api_secret = _env_first(*meta["secret_envs"])

    if not api_key or not api_secret:
        print("❌ 错误：API 密钥未配置")
        print("\n请设置环境变量：")
        print('  export MULTI_LEG_BINANCE_FUTURES_API_KEY="your_key"')
        print('  export MULTI_LEG_BINANCE_FUTURES_API_SECRET="your_secret"')
        sys.exit(1)

    print("=" * 80)
    print("币安账户 SL/TP 保护单检查")
    print("=" * 80)
    print()

    try:
        # 1. 获取所有持仓
        print("📊 步骤 1: 获取持仓信息...")
        account_data = _fetch_futures_account_raw(
            api_key=api_key, api_secret=api_secret
        )

        positions = []
        for pos in account_data.get("positions", []):
            amt = float(pos.get("positionAmt", 0))
            if amt != 0:
                positions.append(
                    {
                        "symbol": pos["symbol"],
                        "side": "LONG" if amt > 0 else "SHORT",
                        "quantity": abs(amt),
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "mark_price": float(pos.get("markPrice", 0)),
                        "leverage": int(pos.get("leverage", 0)),
                        "liquidation_price": float(pos.get("liquidationPrice", 0)),
                    }
                )

        print(f"   找到 {len(positions)} 个持仓\n")

        # 2. 获取所有挂单
        print("📋 步骤 2: 获取挂单信息...")
        orders = _fetch_open_orders_raw(api_key=api_key, api_secret=api_secret)
        print(f"   找到 {len(orders)} 个挂单\n")

        # 3. 按 symbol 和 positionSide 分组挂单
        orders_by_key = defaultdict(list)
        for order in orders:
            symbol = order.get("symbol", "")
            pos_side = order.get("positionSide", "BOTH")
            key = (symbol, pos_side)
            orders_by_key[key].append(order)

        # 4. 检查每个仓位的保护单
        print("=" * 80)
        print("🛡️  保护单检查结果")
        print("=" * 80)
        print()

        for pos in sorted(positions, key=lambda x: (x["symbol"], x["side"])):
            symbol = pos["symbol"]
            side = pos["side"]

            # 查找对应的挂单
            key = (symbol, side)
            related_orders = orders_by_key.get(key, [])

            # 分类挂单
            sl_orders = [
                o for o in related_orders if o.get("type") in ["STOP_MARKET", "STOP"]
            ]
            tp_orders = [
                o
                for o in related_orders
                if o.get("type") in ["TAKE_PROFIT_MARKET", "TAKE_PROFIT"]
            ]
            limit_orders = [o for o in related_orders if o.get("type") == "LIMIT"]

            has_sl = len(sl_orders) > 0
            has_tp = len(tp_orders) > 0

            # 显示结果
            status_sl = "✅ 有" if has_sl else "❌ 无"
            status_tp = "✅ 有" if has_tp else "❌ 无"

            print(
                f"{symbol} {side:5} | qty={pos['quantity']:10.2f} | entry={pos['entry_price']:.4f} | liq={pos['liquidation_price']:.4f}"
            )
            print(f"  SL 保护: {status_sl}", end="")
            if has_sl:
                for sl in sl_orders:
                    print(
                        f" (ID:{sl['orderId']}, price={sl.get('stopPrice', sl.get('price'))})",
                        end="",
                    )
            print()

            print(f"  TP 保护: {status_tp}", end="")
            if has_tp:
                for tp in tp_orders:
                    print(
                        f" (ID:{tp['orderId']}, price={tp.get('stopPrice', tp.get('price'))})",
                        end="",
                    )
            print()

            if not has_sl:
                print(f"  ⚠️  警告：{symbol} {side} 没有 SL 保护！")

            print()

        # 5. 总结
        print("=" * 80)
        print("📊 总结")
        print("=" * 80)

        total_positions = len(positions)
        positions_with_sl = sum(
            1
            for pos in positions
            if any(
                o.get("type") in ["STOP_MARKET", "STOP"]
                for o in orders_by_key.get((pos["symbol"], pos["side"]), [])
            )
        )

        print(f"总持仓数: {total_positions}")
        print(f"有 SL 保护的持仓: {positions_with_sl}/{total_positions}")
        print(
            f"无 SL 保护的持仓: {total_positions - positions_with_sl}/{total_positions}"
        )

        if positions_with_sl < total_positions:
            print()
            print("⚠️  警告：部分持仓缺少 SL 保护！")
            print("   建议：检查 chop_grid live engine 是否正常运行")
            print("   或手动在币安设置止损单")

        print()

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
