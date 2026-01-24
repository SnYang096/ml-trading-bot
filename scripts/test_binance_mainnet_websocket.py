#!/usr/bin/env python3
"""
测试币安主网WebSocket连接，验证是否能接收到tick数据
支持通过SOCKS5代理连接
"""
import asyncio
import json
import websockets
from datetime import datetime
from typing import List, Optional
import os
import threading

try:
    from python_socks.async_.asyncio import Proxy

    PYTHON_SOCKS_AVAILABLE = True
except ImportError:
    PYTHON_SOCKS_AVAILABLE = False
    print("⚠️ python-socks库未安装，SOCKS5代理功能不可用")
    print("   安装: pip install python-socks[asyncio]")


def get_windows_host_ip() -> str:
    """
    获取Windows主机IP地址

    在WSL2中，Windows主机IP通常是默认网关的IP地址

    Returns:
        Windows主机IP地址，如果获取失败则返回127.0.0.1
    """
    import subprocess

    # 方法1: 从路由表获取默认网关
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            # 从路由表提取默认网关IP
            parts = result.stdout.strip().split()
            if "via" in parts:
                idx = parts.index("via")
                if idx + 1 < len(parts):
                    gateway_ip = parts[idx + 1]
                    # 验证是否是私有IP地址（Windows主机通常是私有IP）
                    if gateway_ip and (
                        gateway_ip.startswith("192.168.")
                        or gateway_ip.startswith("172.")
                        or gateway_ip.startswith("10.")
                    ):
                        print(f"📡 从路由表检测到Windows主机IP: {gateway_ip}")
                        return gateway_ip
    except Exception as e:
        print(f"⚠️ 无法从路由表获取IP: {e}")

    # 方法2: 从/etc/resolv.conf获取（如果配置了Windows主机作为DNS）
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("nameserver"):
                    ip = line.split()[1]
                    # 只接受私有IP地址（排除公共DNS如8.8.8.8）
                    if ip and (
                        ip.startswith("192.168.")
                        or ip.startswith("172.")
                        or ip.startswith("10.")
                    ):
                        print(f"📡 从resolv.conf检测到Windows主机IP: {ip}")
                        return ip
    except Exception as e:
        print(f"⚠️ 无法读取/etc/resolv.conf: {e}")

    # 如果无法获取，返回默认值
    print("⚠️ 无法自动检测Windows主机IP，使用默认值127.0.0.1")
    print("   提示：可以使用--proxy-host参数手动指定代理主机IP")
    return "127.0.0.1"  # 默认值


def create_proxy(proxy_type: str, proxy_host: str, proxy_port: int) -> Optional[Proxy]:
    """
    创建代理对象

    Args:
        proxy_type: 代理类型 ('socks5' 或 'http')
        proxy_host: 代理主机地址
        proxy_port: 代理端口

    Returns:
        Proxy对象，如果类型不支持则返回None
    """
    if not PYTHON_SOCKS_AVAILABLE:
        print("⚠️ python-socks库未安装，无法使用代理")
        return None

    try:
        if proxy_type.lower() == "socks5":
            return Proxy.from_url(f"socks5://{proxy_host}:{proxy_port}")
        elif proxy_type.lower() == "http":
            return Proxy.from_url(f"http://{proxy_host}:{proxy_port}")
        else:
            print(f"⚠️ 不支持的代理类型: {proxy_type}")
            return None
    except Exception as e:
        print(f"⚠️ 创建代理对象失败: {e}")
        return None


async def test_binance_mainnet_websocket(
    symbols: List[str],
    duration_seconds: int = 60,
    use_proxy: bool = False,
    proxy_type: str = "socks5",
    proxy_host: Optional[str] = None,
    proxy_port: int = 7897,
):
    """
    测试币安主网WebSocket连接

    注意：SOCKS5代理仅用于本地测试，服务端不需要使用。
    可以通过环境变量 USE_SOCKS5_PROXY=true 来启用代理。

    Args:
        symbols: 交易对符号列表，如 ['BTCUSDT', 'ETHUSDT']
        duration_seconds: 测试时长（秒）
        use_proxy: 是否使用代理（命令行参数优先级高于环境变量）
        proxy_type: 代理类型 ('socks5' 或 'http')
        proxy_host: 代理主机地址，如果为None则从/etc/resolv.conf获取
        proxy_port: 代理端口
    """
    import os

    # 检查环境变量 USE_SOCKS5_PROXY（仅用于本地测试）
    # 如果命令行参数未指定，则检查环境变量
    env_use_proxy = os.environ.get("USE_SOCKS5_PROXY", "false").lower() == "true"
    if not use_proxy and env_use_proxy:
        use_proxy = True
        print(
            "📡 检测到环境变量 USE_SOCKS5_PROXY=true，启用SOCKS5代理（仅用于本地测试）"
        )

    # 如果使用代理，获取代理主机IP
    proxy = None
    if use_proxy:
        if proxy_host is None:
            # 优先使用环境变量中的代理地址（通常是127.0.0.1）
            env_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
            if env_proxy and "://" in env_proxy:
                # 从环境变量提取代理地址
                try:
                    from urllib.parse import urlparse

                    parsed = urlparse(env_proxy)
                    if parsed.hostname:
                        proxy_host = parsed.hostname
                        if parsed.port:
                            proxy_port = parsed.port
                        print(f"📡 从环境变量获取代理地址: {proxy_host}:{proxy_port}")
                    else:
                        proxy_host = get_windows_host_ip()
                except Exception:
                    proxy_host = get_windows_host_ip()
            else:
                proxy_host = get_windows_host_ip()
        proxy = create_proxy(proxy_type, proxy_host, proxy_port)
        if proxy:
            print(f"🔌 使用{proxy_type.upper()}代理: {proxy_host}:{proxy_port}")
        else:
            print("⚠️ 代理创建失败，将尝试直接连接")
            use_proxy = False

    # 处理环境变量代理设置
    import os

    original_proxy = os.environ.get("HTTP_PROXY")
    original_https_proxy = os.environ.get("HTTPS_PROXY")
    # 如果使用代码中的代理，临时取消环境变量代理以避免冲突
    if use_proxy and original_proxy:
        print(f"⚠️  检测到环境变量代理: HTTP_PROXY={original_proxy}")
        print(f"   临时取消环境变量代理，使用代码中指定的代理...")
        del os.environ["HTTP_PROXY"]
        if "HTTPS_PROXY" in os.environ:
            del os.environ["HTTPS_PROXY"]
    # 币安主网公共数据流端点
    # 格式1（组合流）: wss://fstream.binance.com/stream?streams=btcusdt@trade/ethusdt@trade
    # 格式2（单个流）: wss://fstream.binance.com/ws/btcusdt@trade
    # 使用组合流格式（更可靠，与测试网一致）
    streams = [f"{symbol.lower()}@trade" for symbol in symbols]
    stream_path = "/".join(streams)
    url = f"wss://fstream.binance.com/stream?streams={stream_path}"

    print(f"🔌 连接到币安主网WebSocket...")
    print(f"   URL: {url}")
    print(f"   Symbols: {symbols}")
    print(f"   测试时长: {duration_seconds} 秒")
    print()

    tick_counts = {symbol: 0 for symbol in symbols}
    start_time = datetime.now()

    try:
        # 配置WebSocket连接参数
        connect_kwargs = {
            "ping_interval": 20,
            "ping_timeout": 10,
            "close_timeout": 10,
            "max_size": 2**23,  # 8MB
            "max_queue": 32,
            "open_timeout": 60,  # 增加握手超时时间到60秒
        }

        # 如果使用代理，需要使用websocket-client库（同步）
        # websockets库（异步）不支持代理，但我们可以创建一个包装函数
        if use_proxy and proxy:
            print(f"📡 通过{proxy_type.upper()}代理连接")
            print(f"   代理: {proxy_host}:{proxy_port}")
            print("⚠️ websockets库（异步）不支持直接使用代理")
            print("   将尝试使用websocket-client库（同步）进行代理连接测试")
            print("   注意：这将使用同步方式，可能影响性能")

            # 使用websocket-client库进行代理连接
            try:
                import websocket
                import socks
                import socket

                # 设置SOCKS5代理
                if proxy_type.lower() == "socks5":
                    socks.set_default_proxy(socks.SOCKS5, proxy_host, proxy_port)
                    socket.socket = socks.socksocket
                    print(f"✅ 已设置SOCKS5代理: {proxy_host}:{proxy_port}")
                elif proxy_type.lower() == "http":
                    # HTTP代理需要不同的处理
                    print("⚠️ HTTP代理需要特殊处理，暂时只支持SOCKS5")
                    use_proxy = False
                else:
                    print(f"⚠️ 不支持的代理类型: {proxy_type}")
                    use_proxy = False

                if use_proxy:
                    # 使用websocket-client库（同步）进行连接
                    print("📡 使用websocket-client库通过代理连接...")
                    # 注意：websocket-client是同步的，我们需要在异步环境中运行
                    # 使用线程在后台运行
                    tick_counts_proxy = {symbol: 0 for symbol in symbols}
                    start_time_proxy = datetime.now()
                    stop_event = threading.Event()

                    def sync_websocket_test():
                        """同步WebSocket测试函数"""

                        def on_message(ws, msg):
                            try:
                                data = json.loads(msg)
                                # 币安数据格式: {"stream": "btcusdt@trade", "data": {...}}
                                if "stream" in data and "data" in data:
                                    stream = data["stream"]
                                    tick_data = data["data"]
                                    stream_symbol = stream.split("@")[0].upper()

                                    for s in symbols:
                                        if s.upper() == stream_symbol:
                                            tick_counts_proxy[s] += 1
                                            if tick_counts_proxy[s] <= 10:
                                                price = tick_data.get("p", "N/A")
                                                quantity = tick_data.get("q", "N/A")
                                                print(
                                                    f"📊 {s}: 收到第 {tick_counts_proxy[s]} 条tick - 价格: {price}, 数量: {quantity}"
                                                )
                                            elif tick_counts_proxy[s] % 100 == 0:
                                                print(
                                                    f"📊 {s}: 已接收 {tick_counts_proxy[s]} 条tick"
                                                )
                            except Exception as e:
                                print(f"⚠️ 处理消息时出错: {e}")

                        def on_error(ws, error):
                            print(f"❌ WebSocket错误: {error}")

                        def on_close(ws, close_status_code, close_msg):
                            print("🔌 WebSocket连接关闭")

                        def on_open(ws):
                            print("✅ WebSocket连接已建立（通过代理）")
                            print("⏳ 等待接收数据...")
                            print()

                        ws = websocket.WebSocketApp(
                            url,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close,
                            on_open=on_open,
                        )
                        ws.run_forever()

                    # 在后台线程运行同步WebSocket
                    thread = threading.Thread(target=sync_websocket_test, daemon=True)
                    thread.start()

                    # 等待指定时间
                    await asyncio.sleep(duration_seconds)

                    # 停止WebSocket（通过关闭连接）
                    stop_event.set()

                    # 恢复原始socket
                    socket.socket = socket._socket

                    # 输出统计
                    print()
                    print("=" * 80)
                    print("📊 代理连接测试结果统计")
                    print("=" * 80)
                    elapsed_time = (datetime.now() - start_time_proxy).total_seconds()
                    print(f"⏱️  测试时长: {elapsed_time:.1f} 秒")
                    print()

                    for symbol in symbols:
                        count = tick_counts_proxy[symbol]
                        rate = count / elapsed_time if elapsed_time > 0 else 0
                        print(f"🔹 {symbol}:")
                        print(f"   接收tick数: {count}")
                        print(f"   平均速率: {rate:.2f} ticks/秒")
                        print()

                    total_ticks = sum(tick_counts_proxy.values())
                    print(f"📈 总计: {total_ticks} 条tick")

                    if total_ticks > 0:
                        print()
                        print("✅ 通过代理成功接收到tick数据！")
                    else:
                        print()
                        print("⚠️ 未接收到任何tick数据")

                    return  # 同步方式已处理，直接返回
            except ImportError:
                print("⚠️ websocket-client库未安装")
                print("   安装: pip install websocket-client")
                use_proxy = False
            except Exception as e:
                print(f"⚠️ 代理连接配置失败: {e}")
                import traceback

                traceback.print_exc()
                use_proxy = False

        async with websockets.connect(url, **connect_kwargs) as websocket:
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
    finally:
        # 恢复代理环境变量
        import os

        if original_proxy:
            os.environ["HTTP_PROXY"] = original_proxy
            if original_https_proxy:
                os.environ["HTTPS_PROXY"] = original_https_proxy

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
        print("   1. 主网数据流不活跃（不太可能）")
        print("   2. WebSocket端点不正确")
        print("   3. 网络连接问题")
    else:
        print()
        print("✅ 成功接收到tick数据，主网数据流正常")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试币安主网WebSocket连接")
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
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        help="使用代理连接（默认: False，也可通过环境变量 USE_SOCKS5_PROXY=true 启用）",
    )
    parser.add_argument(
        "--proxy-type",
        type=str,
        default="socks5",
        choices=["socks5", "http"],
        help="代理类型（默认: socks5）",
    )
    parser.add_argument(
        "--proxy-host",
        type=str,
        default=None,
        help="代理主机地址（默认: 从/etc/resolv.conf获取Windows主机IP）",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=7897,
        help="代理端口（默认: 7897）",
    )

    args = parser.parse_args()

    await test_binance_mainnet_websocket(
        args.symbols,
        args.duration,
        use_proxy=args.use_proxy,
        proxy_type=args.proxy_type,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
    )


if __name__ == "__main__":
    asyncio.run(main())
