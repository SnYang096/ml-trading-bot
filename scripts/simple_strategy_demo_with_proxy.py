#!/usr/bin/env python3
"""
简单策略Demo - 使用SOCKS5代理连接主网

功能：
1. 通过SOCKS5代理连接Binance主网WebSocket
2. 接收实时ticks数据
3. 保存数据到本地存储
4. 测试补数据功能（能补多久的数据）
"""

import asyncio
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from python_socks.async_.asyncio import Proxy

    PYTHON_SOCKS_AVAILABLE = True
except ImportError:
    PYTHON_SOCKS_AVAILABLE = False
    print("⚠️ python-socks库未安装，SOCKS5代理功能不可用")
    print("   安装: pip install python-socks[asyncio]")

try:
    import websocket

    WEBSOCKET_CLIENT_AVAILABLE = True
except ImportError:
    WEBSOCKET_CLIENT_AVAILABLE = False
    print("⚠️ websocket-client库未安装")
    print("   安装: pip install websocket-client")

from src.live_data_stream.feature_storage import StorageManager
from src.live_data_stream.gap_filler import GapFiller
from src.live_data_stream.data_gap_filler import DataGapFiller

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_windows_host_ip() -> str:
    """获取Windows主机IP地址"""
    import subprocess

    # 从路由表获取默认网关
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if "via" in parts:
                idx = parts.index("via")
                if idx + 1 < len(parts):
                    gateway_ip = parts[idx + 1]
                    if gateway_ip and (
                        gateway_ip.startswith("192.168.")
                        or gateway_ip.startswith("172.")
                        or gateway_ip.startswith("10.")
                    ):
                        return gateway_ip
    except Exception:
        pass

    # 从/etc/resolv.conf获取
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("nameserver"):
                    ip = line.split()[1]
                    if ip and (
                        ip.startswith("192.168.")
                        or ip.startswith("172.")
                        or ip.startswith("10.")
                    ):
                        return ip
    except Exception:
        pass

    # 默认使用127.0.0.1（本地代理）
    return "127.0.0.1"


class SimpleStrategyDemo:
    """简单策略Demo"""

    def __init__(
        self,
        symbols: List[str] = ["BTCUSDT"],
        proxy_host: Optional[str] = None,
        proxy_port: int = 7897,
        storage_path: str = "data/simple_strategy_demo",
    ):
        self.symbols = symbols
        self.proxy_host = proxy_host or get_windows_host_ip()
        self.proxy_port = proxy_port
        self.storage_path = storage_path

        # 统计数据
        self.received_ticks: Dict[str, int] = {symbol: 0 for symbol in symbols}
        self.last_tick_time: Dict[str, float] = {}
        self.start_time = time.time()

        # 存储管理器
        self.storage_manager = StorageManager(base_path=storage_path)
        logger.info(f"✅ 存储管理器已初始化: {storage_path}")

        # GapFiller（用于补数据）
        self.gap_filler = None
        try:
            import ccxt

            exchange = ccxt.binance(
                {
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                }
            )
            self.gap_filler = GapFiller(
                storage_manager=self.storage_manager,
                exchange=exchange,
            )
            logger.info("✅ GapFiller已初始化（可用于补数据）")
        except Exception as e:
            logger.warning(f"⚠️ GapFiller初始化失败: {e}")

    def setup_proxy_socket(self):
        """设置SOCKS5代理socket"""
        try:
            import websocket
            import socks
            import socket as std_socket

            # 保存原始socket
            self._original_socket = std_socket.socket

            # 设置SOCKS5代理
            socks.set_default_proxy(socks.SOCKS5, self.proxy_host, self.proxy_port)
            std_socket.socket = socks.socksocket
            logger.info(f"✅ 已设置SOCKS5代理: {self.proxy_host}:{self.proxy_port}")
            return True
        except ImportError:
            logger.error("❌ websocket-client或python-socks库未安装")
            logger.error("   安装: pip install websocket-client python-socks")
            return False
        except Exception as e:
            logger.error(f"❌ 设置代理失败: {e}")
            return False

    def restore_socket(self):
        """恢复原始socket"""
        try:
            import socket as std_socket

            if hasattr(self, "_original_socket"):
                std_socket.socket = self._original_socket
        except:
            pass

    def process_tick(self, data: Dict[str, Any], symbol: str):
        """处理tick数据"""
        try:
            # 解析tick数据
            price = float(data.get("p", 0))
            qty = float(data.get("q", 0))
            timestamp_ms = int(data.get("T", data.get("E", time.time() * 1000)))
            timestamp = pd.Timestamp.fromtimestamp(timestamp_ms / 1000, tz="UTC")

            # 更新统计
            self.received_ticks[symbol] += 1
            self.last_tick_time[symbol] = time.time()

            # 每100条tick打印一次
            if self.received_ticks[symbol] % 100 == 0:
                logger.info(
                    f"📊 {symbol}: 已接收 {self.received_ticks[symbol]} 条tick | "
                    f"价格: {price:.2f} | 数量: {qty:.4f}"
                )

            # 保存到存储（聚合为1分钟数据）
            tick_df = pd.DataFrame(
                [
                    {
                        "timestamp": timestamp,
                        "price": price,
                        "volume": qty,
                        "turnover": price * qty,
                        "side": 1 if not data.get("m", False) else -1,
                    }
                ]
            )

            self.storage_manager.save_1min_ticks(
                symbol=symbol,
                bars=tick_df,
                timestamp=timestamp,
                include_incomplete=True,
            )

        except Exception as e:
            logger.error(f"处理tick数据失败: {e}")

    async def check_and_fill_gaps(self):
        """检查并补数据"""
        if not self.gap_filler:
            return

        logger.info("🔍 检查数据缺失...")

        for symbol in self.symbols:
            try:
                # 获取最近的数据
                end_time = pd.Timestamp.now(tz="UTC")
                start_time = end_time - timedelta(hours=24)  # 检查最近24小时

                # 加载已有数据
                start_date = start_time.strftime("%Y-%m-%d")
                end_date = end_time.strftime("%Y-%m-%d")

                existing_data = self.storage_manager.tick_1min.load_range(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )

                if existing_data is None or len(existing_data) == 0:
                    logger.info(f"   {symbol}: 无历史数据，跳过补数据")
                    continue

                # 检查缺失
                existing_data = existing_data.sort_values("timestamp")
                if len(existing_data) < 2:
                    continue

                # 计算时间间隔
                time_diffs = existing_data["timestamp"].diff()
                expected_interval = pd.Timedelta("1min")

                # 找出超过2分钟的间隔（可能缺失数据）
                large_gaps = time_diffs[time_diffs > expected_interval * 2]

                if len(large_gaps) > 0:
                    logger.warning(f"   {symbol}: 检测到 {len(large_gaps)} 个数据缺失")

                    # 尝试补数据
                    for gap_idx in large_gaps.index:
                        gap_start = existing_data.loc[gap_idx - 1, "timestamp"]
                        gap_end = existing_data.loc[gap_idx, "timestamp"]
                        gap_duration = gap_end - gap_start

                        logger.info(
                            f"   {symbol}: 补数据 {gap_start} -> {gap_end} "
                            f"(缺失 {gap_duration.total_seconds() / 60:.1f} 分钟)"
                        )

                        # 使用GapFiller补数据
                        try:
                            fill_data = self.gap_filler.fill_gap(
                                symbol=symbol,
                                start_time=gap_start,
                                end_time=gap_end,
                                source="binance",  # 直接从币安API获取
                            )

                            if len(fill_data) > 0:
                                logger.info(
                                    f"   ✅ {symbol}: 成功补全 {len(fill_data)} 条数据"
                                )
                                # 保存补全的数据
                                self.storage_manager.save_1min_ticks(
                                    symbol=symbol,
                                    bars=fill_data,
                                    include_incomplete=False,
                                )
                            else:
                                logger.warning(f"   ⚠️ {symbol}: 补数据返回空结果")

                        except Exception as e:
                            logger.error(f"   ❌ {symbol}: 补数据失败: {e}")

                else:
                    logger.info(f"   {symbol}: 数据完整，无缺失")

            except Exception as e:
                logger.error(f"检查补数据时出错 ({symbol}): {e}")

    async def run(self, duration: int = 3600):
        """运行策略"""
        logger.info("=" * 60)
        logger.info("简单策略Demo启动")
        logger.info("=" * 60)
        logger.info(f"交易对: {self.symbols}")
        logger.info(f"运行时长: {duration}秒")
        logger.info(f"代理: {self.proxy_host}:{self.proxy_port}")
        logger.info("=" * 60)

        # 设置代理
        if not self.setup_proxy_socket():
            logger.error("❌ 代理设置失败，退出")
            return

        # 构建WebSocket URL
        streams = "/".join(f"{sym.lower()}@trade" for sym in self.symbols)
        url = f"wss://fstream.binance.com/stream?streams={streams}"

        logger.info(f"📡 WebSocket URL: {url}")

        stop_event = asyncio.Event()
        last_gap_check = time.time()
        ws_app = None

        try:
            import websocket
            import threading

            def on_message(ws, msg):
                """处理接收到的消息"""
                try:
                    data = json.loads(msg)

                    # 处理Binance数据格式: {"stream": "btcusdt@trade", "data": {...}}
                    if isinstance(data, dict) and "stream" in data and "data" in data:
                        stream = data["stream"]
                        payload = data["data"]
                        symbol = stream.split("@")[0].upper()

                        if symbol in self.symbols and payload.get("e") == "trade":
                            # 直接处理tick（同步）
                            self.process_tick(payload, symbol)

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败: {e}")
                except Exception as e:
                    logger.error(f"处理消息失败: {e}")

            def on_error(ws, error):
                logger.error(f"❌ WebSocket错误: {error}")

            def on_close(ws, close_status_code, close_msg):
                logger.warning("⚠️ WebSocket连接关闭")

            def on_open(ws):
                logger.info("✅ WebSocket连接已建立（通过SOCKS5代理）")

            # 创建WebSocket应用
            ws_app = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )

            # 在后台线程运行WebSocket
            def run_websocket():
                ws_app.run_forever()

            thread = threading.Thread(target=run_websocket, daemon=True)
            thread.start()

            # 等待连接建立
            await asyncio.sleep(2)

            # 主循环
            while (
                not stop_event.is_set() and (time.time() - self.start_time) < duration
            ):
                await asyncio.sleep(1)

                # 定期检查补数据（每5分钟）
                if time.time() - last_gap_check >= 300:
                    await self.check_and_fill_gaps()
                    last_gap_check = time.time()

                # 打印状态（每30秒）
                if int(time.time() - self.start_time) % 30 == 0:
                    total_ticks = sum(self.received_ticks.values())
                    logger.info(
                        f"⏱️  运行时间: {int(time.time() - self.start_time)}s | "
                        f"总接收tick: {total_ticks}"
                    )

        except KeyboardInterrupt:
            logger.info("用户中断")
        except Exception as e:
            logger.error(f"运行出错: {e}", exc_info=True)
        finally:
            stop_event.set()

            # 关闭WebSocket
            if ws_app:
                ws_app.close()

            # 恢复原始socket
            self.restore_socket()

            # 最终检查补数据
            await self.check_and_fill_gaps()

            # 打印统计
            self.print_stats()

    def print_stats(self):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("测试结果总结")
        logger.info("=" * 60)

        total_ticks = sum(self.received_ticks.values())
        logger.info(f"📊 总接收tick数: {total_ticks}")

        for symbol in self.symbols:
            ticks = self.received_ticks[symbol]
            runtime = time.time() - self.start_time
            rate = ticks / runtime if runtime > 0 else 0
            logger.info(
                f"   {symbol}: {ticks} 条tick | " f"平均速率: {rate:.2f} tick/s"
            )

        logger.info("=" * 60)


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="简单策略Demo（使用SOCKS5代理）")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT",
        help="交易对列表，逗号分隔（默认: BTCUSDT）",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="运行时长（秒，默认: 3600）",
    )
    parser.add_argument(
        "--proxy-host",
        type=str,
        default=None,
        help="代理主机地址（默认: 自动检测Windows主机IP）",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=7897,
        help="代理端口（默认: 7897）",
    )

    args = parser.parse_args()

    symbols = args.symbols.split(",")

    demo = SimpleStrategyDemo(
        symbols=symbols,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
    )

    await demo.run(duration=args.duration)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序出错: {e}", exc_info=True)
        sys.exit(1)
