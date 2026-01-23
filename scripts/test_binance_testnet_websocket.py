#!/usr/bin/env python3
"""
测试币安测试网WebSocket连接，验证是否能接收到tick数据
"""
import asyncio
import json
import websockets
from datetime import datetime
from typing import List


async def test_binance_testnet_websocket(
    symbols: List[str], duration_seconds: int = 60
):
    """
    测试币安测试网WebSocket连接

    Args:
        symbols: 交易对符号列表，如 ['BTCUSDT', 'ETHUSDT']
        duration_seconds: 测试时长（秒）
    """
    # 币安测试网公共数据流端点
    # 格式: wss://stream.binancefuture.com/stream?streams=btcusdt@trade/ethusdt@trade
    streams = [f"{symbol.lower()}@trade" for symbol in symbols]
    stream_path = "/".join(streams)
    url = f"wss://stream.binancefuture.com/stream?streams={stream_path}"

    print(f"🔌 连接到币安测试网WebSocket...")
    print(f"   URL: {url}")
    print(f"   Symbols: {symbols}")
    print(f"   测试时长: {duration_seconds} 秒")
    print()

    tick_counts = {symbol: 0 for symbol in symbols}
    start_time = datetime.now()

    try:
        async with websockets.connect(url) as websocket:
            print(f"✅ WebSocket连接已建立")
            print(f"⏳ 等待接收数据...")
            print()

            # 设置超时
            end_time = start_time.timestamp() + duration_seconds

            while datetime.now().timestamp() < end_time:
                try:
                    # 设置接收超时（1秒）
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)

                    # 币安数据格式: {"stream": "btcusdt@trade", "data": {...}}
                    if "stream" in data and "data" in data:
                        stream = data["stream"]
                        tick_data = data["data"]

                        # 提取symbol（从stream名称中提取，如 btcusdt@trade -> BTCUSDT）
                        stream_symbol = stream.split("@")[0].upper()

                        # 找到对应的symbol（处理大小写）
                        symbol = None
                        for s in symbols:
                            if s.upper() == stream_symbol:
                                symbol = s
                                break

                        if symbol:
                            tick_counts[symbol] += 1

                            # 显示前10条tick的详细信息
                            if tick_counts[symbol] <= 10:
                                price = tick_data.get("p", "N/A")
                                quantity = tick_data.get("q", "N/A")
                                time_ms = tick_data.get("T", "N/A")
                                print(
                                    f"📊 {symbol}: 收到第 {tick_counts[symbol]} 条tick"
                                )
                                print(
                                    f"   价格: {price}, 数量: {quantity}, 时间: {time_ms}"
                                )

                            # 每100条tick输出一次统计
                            if tick_counts[symbol] % 100 == 0:
                                print(
                                    f"📊 {symbol}: 已接收 {tick_counts[symbol]} 条tick"
                                )

                    # 检查是否是ping消息
                    elif isinstance(data, dict) and "ping" in data:
                        # 发送pong响应
                        pong = {"pong": data["ping"]}
                        await websocket.send(json.dumps(pong))
                        print("🏓 收到ping，已发送pong")

                except asyncio.TimeoutError:
                    # 超时，继续等待
                    elapsed = datetime.now().timestamp() - start_time.timestamp()
                    if int(elapsed) % 10 == 0 and elapsed > 0:
                        print(
                            f"⏳ 等待中... ({int(elapsed)}秒，已接收: {dict(tick_counts)})"
                        )
                    continue
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON解析错误: {e}")
                    print(f"   原始消息: {message[:200]}")
                except Exception as e:
                    print(f"⚠️  处理消息时出错: {e}")
                    import traceback

                    traceback.print_exc()

    except websockets.exceptions.InvalidURI as e:
        print(f"❌ 无效的WebSocket URL: {e}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ WebSocket连接已关闭: {e}")
    except Exception as e:
        print(f"❌ 连接错误: {e}")
        import traceback

        traceback.print_exc()

    # 输出最终统计
    print()
    print("=" * 80)
    print("📊 测试结果统计")
    print("=" * 80)
    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"⏱️  测试时长: {elapsed_time:.1f} 秒")
    print()

    for symbol in symbols:
        count = tick_counts[symbol]
        rate = count / elapsed_time if elapsed_time > 0 else 0
        print(f"🔹 {symbol}:")
        print(f"   接收tick数: {count}")
        print(f"   平均速率: {rate:.2f} ticks/秒")
        print()

    total_ticks = sum(tick_counts.values())
    print(f"📈 总计: {total_ticks} 条tick")

    if total_ticks == 0:
        print()
        print("⚠️  警告: 未接收到任何tick数据")
        print("   可能的原因:")
        print("   1. 测试网数据流不活跃（测试网可能没有实时交易数据）")
        print("   2. WebSocket端点不正确")
        print("   3. 网络连接问题")
    else:
        print()
        print("✅ 成功接收到tick数据，测试网数据流正常")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试币安测试网WebSocket连接")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="交易对符号列表（默认: BTCUSDT ETHUSDT）",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="测试时长（秒，默认: 60）",
    )

    args = parser.parse_args()

    await test_binance_testnet_websocket(args.symbols, args.duration)


if __name__ == "__main__":
    asyncio.run(main())
