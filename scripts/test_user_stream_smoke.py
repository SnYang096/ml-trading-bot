"""
测试网冒烟测试：检查账户余额和 User Data Stream
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.order_management.storage import Storage
from src.order_management.binance_api import BinanceAPI
from src.order_management.order_manager import OrderManager
from src.order_management.binance_user_stream import BinanceUserStream
from src.order_management.demo_strategy import load_api_keys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def smoke_test(testnet: bool = True, duration: int = 60):
    """冒烟测试：检查账户余额和 User Data Stream"""
    logger.info("=" * 80)
    logger.info("🚀 启动测试网冒烟测试")
    logger.info("=" * 80)

    # 加载API密钥
    api_key, api_secret = load_api_keys(testnet)
    logger.info(f"✅ API密钥已加载（测试网: {testnet}）")

    # 初始化系统
    db_path = project_root / "data" / "user_stream_smoke_test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = Storage(str(db_path))

    binance_api = BinanceAPI(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    order_manager = OrderManager(storage, binance_api)

    # 检查账户余额
    try:
        account_info = binance_api.get_account_info()
        logger.info("=" * 80)
        logger.info("💰 账户余额信息")
        logger.info("=" * 80)
        logger.info(f"总余额: {account_info.get('total_balance', 0):.4f} USDT")
        logger.info(f"可用余额: {account_info.get('free_balance', 0):.4f} USDT")
        logger.info(f"已用余额: {account_info.get('used_balance', 0):.4f} USDT")
    except Exception as e:
        logger.warning(f"⚠️ 获取账户余额失败（继续测试 User Data Stream）: {e}")
        logger.info("继续测试 User Data Stream...")

    # 初始化 User Data Stream
    execution_reports_received = []

    def on_execution_report(report: dict) -> None:
        """处理订单执行回报"""
        execution_reports_received.append(report)
        logger.info(
            f"📨 收到订单回报 #{len(execution_reports_received)}: "
            f"order_id={report.get('order_id')}, "
            f"status={report.get('status')}, "
            f"symbol={report.get('symbol')}"
        )
        try:
            order_manager.handle_execution_report(report)
        except Exception as e:
            logger.error(f"处理订单回报失败: {e}", exc_info=True)

    user_stream = BinanceUserStream(
        binance_api=binance_api,
        on_execution_report=on_execution_report,
        keepalive_interval=30 * 60,
    )

    try:
        # 启动 User Data Stream
        logger.info("=" * 80)
        logger.info("🔌 启动 User Data Stream")
        logger.info("=" * 80)
        await user_stream.start()
        logger.info(f"✅ User Data Stream已启动，等待 {duration} 秒...")

        # 等待指定时间
        await asyncio.sleep(duration)

        # 统计信息
        logger.info("=" * 80)
        logger.info("📊 测试统计")
        logger.info("=" * 80)
        logger.info(f"收到订单回报数量: {len(execution_reports_received)}")
        if execution_reports_received:
            logger.info("最近的订单回报:")
            for report in execution_reports_received[-5:]:  # 显示最后5个
                logger.info(f"  - {report}")

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
    finally:
        # 停止 User Data Stream
        await user_stream.stop()
        logger.info("✅ User Data Stream已停止")
        logger.info("=" * 80)
        logger.info("✅ 冒烟测试完成")
        logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试网冒烟测试")
    parser.add_argument(
        "--testnet", action="store_true", default=True, help="使用测试网（默认: True）"
    )
    parser.add_argument(
        "--mainnet", action="store_true", help="使用主网（覆盖 --testnet）"
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="测试持续时间（秒，默认: 60）"
    )

    args = parser.parse_args()

    # 如果指定了 --mainnet，则使用主网
    testnet = not args.mainnet if args.mainnet else args.testnet

    asyncio.run(smoke_test(testnet=testnet, duration=args.duration))
