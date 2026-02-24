"""Prometheus 指标暴露模块

启动一个 HTTP server 在指定端口 (默认 9090)，暴露 /metrics 端点供 Prometheus 抓取。
所有指标以 mlbot_ 为前缀。

使用:
    from src.time_series_model.live.metrics_exporter import (
        start_metrics_server, METRICS,
    )
    start_metrics_server(port=9090)

    # 更新指标
    METRICS.bars_processed.inc(6)
    METRICS.funnel_stage.labels(stage="gate", strategy="me").inc()
    METRICS.positions_active.set(1)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── 延迟导入: prometheus_client 可选依赖 ──────────────────────

_PROM_AVAILABLE = False
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Info,
        start_http_server,
    )

    _PROM_AVAILABLE = True
except ImportError:
    pass


# ── 指标定义 ──────────────────────────────────────────────────


class _NoopMetric:
    """prometheus_client 未安装时的空操作替身"""

    def inc(self, *a, **kw):
        pass

    def dec(self, *a, **kw):
        pass

    def set(self, *a, **kw):
        pass

    def observe(self, *a, **kw):
        pass

    def labels(self, *a, **kw):
        return self

    def info(self, *a, **kw):
        pass


_NOOP = _NoopMetric()


class Metrics:
    """全部 Prometheus 指标的集中定义"""

    def __init__(self) -> None:
        if not _PROM_AVAILABLE:
            # 所有属性指向空操作
            self.bars_processed = _NOOP
            self.funnel_stage = _NOOP
            self.signals_total = _NOOP
            self.orders_total = _NOOP
            self.positions_active = _NOOP
            self.pnl_realized_total = _NOOP
            self.drawdown = _NOOP
            self.loss = _NOOP
            self.kill_switch_halted = _NOOP
            self.last_bar_age = _NOOP
            self.ws_connected = _NOOP
            self.cpu_percent = _NOOP
            self.memory_mb = _NOOP
            self.uptime_seconds = _NOOP
            self.gate_reject_rate = _NOOP
            self.bot_info = _NOOP
            return

        # ── Counters (累计值，只增不减) ──

        self.bars_processed = Counter(
            "mlbot_bars_processed_total",
            "Total bars processed across all symbols",
        )

        self.funnel_stage = Counter(
            "mlbot_funnel_total",
            "Signal funnel stage pass count",
            ["stage", "strategy"],
        )

        self.signals_total = Counter(
            "mlbot_signals_total",
            "Signals generated (evidence passed)",
            ["strategy"],
        )

        self.orders_total = Counter(
            "mlbot_orders_total",
            "Orders placed",
            ["strategy"],
        )

        # ── Gauges (当前值，可升可降) ──

        self.positions_active = Gauge(
            "mlbot_positions_active",
            "Currently active positions",
        )

        self.pnl_realized_total = Gauge(
            "mlbot_pnl_realized_total",
            "Cumulative realized PnL (percent of equity)",
        )

        self.drawdown = Gauge(
            "mlbot_drawdown",
            "Current drawdown (0.0 to 1.0)",
        )

        self.loss = Gauge(
            "mlbot_loss",
            "Loss by period (fraction of equity)",
            ["period"],  # daily / weekly / monthly
        )

        self.kill_switch_halted = Gauge(
            "mlbot_kill_switch_halted",
            "Kill switch state: 0=running, 1=halted",
        )

        self.last_bar_age = Gauge(
            "mlbot_last_bar_age_seconds",
            "Seconds since last bar was processed",
            ["symbol"],
        )

        self.ws_connected = Gauge(
            "mlbot_ws_connected",
            "WebSocket connection state: 1=connected, 0=disconnected",
            ["symbol"],
        )

        self.cpu_percent = Gauge(
            "mlbot_cpu_percent",
            "CPU usage percent",
        )

        self.memory_mb = Gauge(
            "mlbot_memory_mb",
            "Memory usage in MB",
        )

        self.uptime_seconds = Gauge(
            "mlbot_uptime_seconds",
            "Process uptime in seconds",
        )

        self.gate_reject_rate = Gauge(
            "mlbot_gate_reject_rate",
            "Gate rejection rate (0.0 to 1.0)",
        )

        # ── Info ──

        self.bot_info = Info(
            "mlbot",
            "Trading bot metadata",
        )

    def update_system_health(self) -> None:
        """读取 psutil 更新 CPU / 内存 / uptime"""
        try:
            import psutil

            self.cpu_percent.set(psutil.cpu_percent(interval=0))
            mem = psutil.virtual_memory()
            self.memory_mb.set(round(mem.used / 1024 / 1024, 1))
        except Exception:
            pass

    def update_from_flush(
        self,
        bars: int,
        direction: int,
        gate: int,
        entry_filter: int,
        evidence: int,
        pcm_selected: int,
        orders: int,
        by_strategy: dict,
        positions_count: int,
    ) -> None:
        """StatsCollector.flush() 调用此方法同步更新 Prometheus 指标"""
        self.bars_processed.inc(bars)
        self.positions_active.set(positions_count)

        # Gate reject rate
        if bars > 0:
            self.gate_reject_rate.set(1.0 - gate / bars if bars else 0)

        # 按策略更新漏斗
        for strategy, stats in by_strategy.items():
            s = str(strategy)
            if stats.get("direction", 0):
                self.funnel_stage.labels(stage="direction", strategy=s).inc(
                    stats["direction"]
                )
            if stats.get("gate_passed", 0):
                self.funnel_stage.labels(stage="gate", strategy=s).inc(
                    stats["gate_passed"]
                )
            if stats.get("entry_filter_passed", 0):
                self.funnel_stage.labels(stage="entry_filter", strategy=s).inc(
                    stats["entry_filter_passed"]
                )
            if stats.get("signals", 0):
                self.funnel_stage.labels(stage="evidence", strategy=s).inc(
                    stats["signals"]
                )
                self.signals_total.labels(strategy=s).inc(stats["signals"])
            if stats.get("pcm_selected", 0):
                self.funnel_stage.labels(stage="pcm", strategy=s).inc(
                    stats["pcm_selected"]
                )
            if stats.get("orders", 0):
                self.funnel_stage.labels(stage="order", strategy=s).inc(stats["orders"])
                self.orders_total.labels(strategy=s).inc(stats["orders"])


# ── 全局单例 ──────────────────────────────────────────────────

METRICS = Metrics()

# 启动时间 (用于计算 uptime)
_START_TIME: Optional[float] = None


def start_metrics_server(port: int = 9090) -> bool:
    """启动 Prometheus HTTP metrics server

    Returns:
        True if started, False if prometheus_client not available
    """
    global _START_TIME

    if not _PROM_AVAILABLE:
        logger.warning(
            "prometheus_client 未安装，跳过 metrics server。"
            "安装: pip install prometheus_client"
        )
        return False

    _START_TIME = time.time()

    # 设置 bot info
    METRICS.bot_info.info(
        {
            "version": "1.0",
            "strategies": "bpc,me,fer",
        }
    )

    # 注册 uptime callback
    if hasattr(METRICS.uptime_seconds, "set_function"):
        METRICS.uptime_seconds.set_function(
            lambda: time.time() - _START_TIME if _START_TIME else 0
        )

    try:
        start_http_server(port)
        logger.info(
            "✅ Prometheus metrics server 启动: http://0.0.0.0:%d/metrics", port
        )
        return True
    except OSError as e:
        logger.error("❌ Prometheus metrics server 启动失败 (端口 %d): %s", port, e)
        return False
