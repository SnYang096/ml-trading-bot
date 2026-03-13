#!/usr/bin/env python3
"""
监控系统演示脚本
启动一个模拟的交易机器人，暴露 Prometheus 指标用于演示
"""

import time
import threading
import random
import logging
import signal
import sys
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from src.time_series_model.live.metrics_exporter import (
        start_metrics_server,
        METRICS,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client 未安装，将使用模拟模式")
    PROMETHEUS_AVAILABLE = False


class MockTradingBot:
    """模拟交易机器人，用于演示监控系统"""

    def __init__(self):
        self.running = True
        self.bars_processed = 0
        self.positions = 0
        self.pnl = 0.0
        self.drawdown = 0.0
        self.orders_by_strategy = {"bpc": 0, "me-long": 0, "fer": 0}

    def simulate_trading_activity(self):
        """模拟交易活动，更新指标"""
        while self.running:
            try:
                # 模拟处理 bars
                bars_this_cycle = random.randint(1, 5)
                METRICS.bars_processed.inc(bars_this_cycle)
                self.bars_processed += bars_this_cycle

                # 模拟信号通过漏斗
                for strategy in ["bpc", "me-long", "fer"]:
                    if random.random() > 0.3:  # 70% 概率有信号
                        # 信号通过各阶段
                        METRICS.funnel_stage.labels(
                            stage="direction", strategy=strategy
                        ).inc()

                        if random.random() > 0.2:  # 80% 概率通过 gate
                            METRICS.funnel_stage.labels(
                                stage="gate", strategy=strategy
                            ).inc()

                            if random.random() > 0.3:  # 70% 概率通过 evidence
                                METRICS.funnel_stage.labels(
                                    stage="evidence", strategy=strategy
                                ).inc()
                                METRICS.signals_total.labels(strategy=strategy).inc()

                                if random.random() > 0.4:  # 60% 概率下单
                                    METRICS.funnel_stage.labels(
                                        stage="order", strategy=strategy
                                    ).inc()
                                    METRICS.orders_total.labels(strategy=strategy).inc()
                                    self.orders_by_strategy[strategy] += 1

                                    # 模拟 PnL 变化
                                    pnl_change = random.uniform(
                                        -0.002, 0.005
                                    )  # -0.2% to +0.5%
                                    self.pnl += pnl_change
                                    METRICS.pnl_realized_total.set(self.pnl)

                                    # 模拟仓位变化
                                    if random.random() > 0.8:  # 20% 概率开仓
                                        self.positions = min(2, self.positions + 1)
                                    elif random.random() > 0.9:  # 10% 概率平仓
                                        self.positions = max(0, self.positions - 1)
                                    METRICS.positions_active.set(self.positions)

                # 模拟系统健康指标
                METRICS.update_system_health()

                # 模拟其他指标
                METRICS.kill_switch_halted.set(0)  # 0 = running, 1 = halted
                METRICS.ws_connected.labels(symbol="BTCUSDT").set(1)
                METRICS.ws_connected.labels(symbol="ETHUSDT").set(1)

                # 计算 gate reject rate (模拟)
                if self.bars_processed > 0:
                    reject_rate = random.uniform(0.6, 0.85)  # 60-85% 拦截率
                    METRICS.gate_reject_rate.set(reject_rate)

                # 模拟亏损指标
                daily_loss = max(0, random.uniform(-0.01, 0.005))  # 模拟日亏损
                METRICS.loss.labels(period="daily").set(daily_loss)

                logger.info(
                    f"📊 模拟数据: Bars={self.bars_processed}, PnL={self.pnl:.3f}, Positions={self.positions}, Orders=BPC:{self.orders_by_strategy['bpc']} ME:{self.orders_by_strategy['me']} FER:{self.orders_by_strategy['fer']}"
                )

                time.sleep(5)  # 每5秒更新一次

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"模拟交易活动出错: {e}")
                time.sleep(5)


def signal_handler(sig, frame):
    logger.info("👋 收到退出信号，正在停止...")
    if hasattr(MockTradingBot, "running"):
        MockTradingBot.running = False
    sys.exit(0)


def main():
    logger.info("🚀 启动监控系统演示...")

    if not PROMETHEUS_AVAILABLE:
        logger.error(
            "❌ prometheus_client 未安装，请先运行: pip install prometheus_client"
        )
        return

    # 启动 metrics server
    logger.info("🔌 启动 Prometheus metrics server...")
    success = start_metrics_server(port=9090)
    if not success:
        logger.error("❌ 无法启动 metrics server")
        return

    logger.info("✅ Metrics server 启动成功: http://localhost:9090/metrics")
    logger.info("💡 现在你可以:")
    logger.info("   1. 访问 http://localhost:9090/metrics 查看原始指标")
    logger.info("   2. 启动 Prometheus + Grafana (见下面说明)")

    # 启动模拟交易活动
    bot = MockTradingBot()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("🎮 开始模拟交易活动...")
    try:
        bot.simulate_trading_activity()
    except KeyboardInterrupt:
        logger.info("👋 用户中断，正在停止...")
    finally:
        bot.running = False


if __name__ == "__main__":
    main()
