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
import os
import time
from typing import List, Optional

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
            self.gate_rejected_total = _NOOP
            self.gate_reject_reasons_total = _NOOP
            self.direction_total = _NOOP
            self.memory_rss_mb = _NOOP
            self.funding_rate = _NOOP
            self.mark_price = _NOOP
            self.open_interest_usd = _NOOP
            self.account_balance = _NOOP
            self.account_margin_ratio = _NOOP
            self.unrealized_pnl_total = _NOOP
            self.system_mode = _NOOP
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

        # ── Per-strategy gate rejection ──

        self.gate_rejected_total = Counter(
            "mlbot_gate_rejected_total",
            "Gate rejections by strategy",
            ["strategy"],
        )

        self.gate_reject_reasons_total = Counter(
            "mlbot_gate_reject_reasons_total",
            "Gate rejection reasons by strategy",
            ["strategy", "reason"],
        )

        self.direction_total = Counter(
            "mlbot_direction_total",
            "Direction assignments by strategy and side",
            ["strategy", "side"],  # side: long / short
        )

        self.memory_rss_mb = Gauge(
            "mlbot_memory_rss_mb",
            "Process RSS memory in MB",
        )

        # ── Market Data (公开 API) ──

        self.funding_rate = Gauge(
            "mlbot_funding_rate",
            "Current funding rate per symbol",
            ["symbol"],
        )

        self.mark_price = Gauge(
            "mlbot_mark_price",
            "Current mark price per symbol",
            ["symbol"],
        )

        self.open_interest_usd = Gauge(
            "mlbot_open_interest_usd",
            "Open interest in USD per symbol",
            ["symbol"],
        )

        # ── Account Data (需要 API key) ──

        self.account_balance = Gauge(
            "mlbot_account_balance",
            "Account balance in USDT",
            ["type"],  # total / available / margin
        )

        self.account_margin_ratio = Gauge(
            "mlbot_account_margin_ratio",
            "Maintenance margin ratio (0-1, >1 = liquidation)",
        )

        self.unrealized_pnl_total = Gauge(
            "mlbot_unrealized_pnl_total",
            "Total unrealized PnL in USDT",
        )

        # ── System Mode ──

        self.system_mode = Gauge(
            "mlbot_system_mode",
            "System operating mode: 0=OFFLINE, 1=DEGRADED, 2=NORMAL",
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
            # 进程级 RSS
            proc = psutil.Process(os.getpid())
            self.memory_rss_mb.set(round(proc.memory_info().rss / 1024 / 1024, 1))
        except Exception:
            pass

    # ── Market & Account Data ───────────────────────────────────

    def update_market_data(self, symbols: List[str]) -> None:
        """从 Binance 公开 REST API 获取资金费率 / 标记价格 / OI

        不需要 API Key，调用频率建议 ≤ 1次/30s。
        """
        try:
            import requests
        except ImportError:
            return

        session = self._get_http_session()
        base = "https://fapi.binance.com"
        sym_set = set(symbols)

        # ── premiumIndex (批量: 所有 symbol 一次请求) ──
        try:
            resp = session.get(f"{base}/fapi/v1/premiumIndex", timeout=10)
            if resp.ok:
                mark_prices = {}  # 临时缓存用于计算 OI USD
                for item in resp.json():
                    sym = item.get("symbol", "")
                    if sym not in sym_set:
                        continue
                    fr = float(item.get("lastFundingRate", 0))
                    mp = float(item.get("markPrice", 0))
                    self.funding_rate.labels(symbol=sym).set(fr)
                    self.mark_price.labels(symbol=sym).set(mp)
                    mark_prices[sym] = mp
        except Exception as exc:
            logger.debug("premiumIndex 获取失败: %s", exc)
            mark_prices = {}

        # ── openInterest (每个 symbol 单独请求) ──
        for sym in symbols:
            try:
                resp = session.get(
                    f"{base}/fapi/v1/openInterest",
                    params={"symbol": sym},
                    timeout=5,
                )
                if resp.ok:
                    oi_contracts = float(resp.json().get("openInterest", 0))
                    mp = mark_prices.get(sym, 0)
                    oi_usd = oi_contracts * mp if mp > 0 else oi_contracts
                    self.open_interest_usd.labels(symbol=sym).set(oi_usd)
            except Exception:
                pass

    def update_account_data(self) -> None:
        """从 Binance Futures 私有 API 获取账户余额/保证金/未实现盈亏

        需要 BINANCE_API_KEY + BINANCE_API_SECRET 环境变量。
        如果未配置则静默跳过。
        """
        api_key = os.getenv("BINANCE_API_KEY") or os.getenv(
            "BINANCE_FUTURES_API_KEY", ""
        )
        api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
            "BINANCE_FUTURES_API_SECRET", ""
        )
        if not api_key or not api_secret:
            logger.warning(
                "account data 跳过: BINANCE_API_KEY / BINANCE_API_SECRET 未配置"
            )
            return

        try:
            import hashlib
            import hmac
            import requests  # noqa: F811
        except ImportError:
            return

        session = self._get_http_session()
        base = "https://fapi.binance.com"

        # 获取服务器时间以修正本地时钟偏移
        try:
            srv_resp = session.get(f"{base}/fapi/v1/time", timeout=5)
            if srv_resp.ok:
                server_ts = int(srv_resp.json().get("serverTime", 0))
            else:
                server_ts = int(time.time() * 1000)
        except Exception:
            server_ts = int(time.time() * 1000)

        query = f"timestamp={server_ts}"
        sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

        try:
            resp = session.get(
                f"{base}/fapi/v2/account?{query}&signature={sig}",
                headers={"X-MBX-APIKEY": api_key},
                timeout=10,
            )
            if not resp.ok:
                logger.warning("account API %d: %s", resp.status_code, resp.text[:200])
                return

            data = resp.json()
            self.account_balance.labels(type="total").set(
                float(data.get("totalWalletBalance", 0))
            )
            self.account_balance.labels(type="available").set(
                float(data.get("availableBalance", 0))
            )
            self.account_balance.labels(type="margin").set(
                float(data.get("totalMarginBalance", 0))
            )

            margin_bal = float(data.get("totalMarginBalance", 0))
            maint_margin = float(data.get("totalMaintMargin", 0))
            if margin_bal > 0:
                self.account_margin_ratio.set(round(maint_margin / margin_bal, 6))

            self.unrealized_pnl_total.set(float(data.get("totalUnrealizedProfit", 0)))
        except Exception as exc:
            logger.warning("account data 获取失败: %s", exc)

    @staticmethod
    def _get_http_session():
        """返回带代理配置的 requests Session (如果需要)"""
        import requests as _req

        session = _req.Session()
        if os.getenv("USE_SOCKS5_PROXY", "").lower() in ("1", "true", "yes"):
            host = os.getenv("SOCKS5_HOST", "127.0.0.1")
            port = os.getenv("SOCKS5_PORT", "7897")
            proxy = f"socks5h://{host}:{port}"
            session.proxies = {"http": proxy, "https": proxy}
        return session

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

        # Gate reject rate (基于 direction 而非 bars)
        total_dir = direction
        if total_dir > 0:
            self.gate_reject_rate.set(1.0 - gate / total_dir)

        # 按策略更新漏斗
        for strategy, stats in by_strategy.items():
            s = str(strategy)
            if stats.get("direction", 0):
                self.funnel_stage.labels(stage="direction", strategy=s).inc(
                    stats["direction"]
                )
            # 方向分布 (long/short)
            if stats.get("long", 0):
                self.direction_total.labels(strategy=s, side="long").inc(stats["long"])
            if stats.get("short", 0):
                self.direction_total.labels(strategy=s, side="short").inc(
                    stats["short"]
                )
            if stats.get("gate_passed", 0):
                self.funnel_stage.labels(stage="gate", strategy=s).inc(
                    stats["gate_passed"]
                )
            # Per-strategy gate rejection
            if stats.get("gate_rejected", 0):
                self.gate_rejected_total.labels(strategy=s).inc(stats["gate_rejected"])
            # Gate rejection reasons
            gate_reasons = stats.get("gate_reject_reasons")
            if isinstance(gate_reasons, dict):
                for reason, count in gate_reasons.items():
                    if isinstance(count, (int, float)) and count > 0:
                        self.gate_reject_reasons_total.labels(
                            strategy=s, reason=str(reason)
                        ).inc(count)
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
