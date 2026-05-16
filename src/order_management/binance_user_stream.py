"""
Binance User Data Stream (Futures)
负责listenKey维护与订单成交事件处理
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional, Dict, Any, List

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:  # pragma: no cover
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore

from .binance_api import BinanceAPI

logger = logging.getLogger(__name__)


class BinanceUserStream:
    """Binance User Data Stream客户端（期货）"""

    def __init__(
        self,
        binance_api: BinanceAPI,
        on_execution_report: Callable[[Dict[str, Any]], None],
        on_account_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        keepalive_interval: int = 30 * 60,
    ) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets 模块未安装，请安装: pip install websockets")

        self.binance_api = binance_api
        self.on_execution_report = on_execution_report
        self.on_account_update = on_account_update
        self.keepalive_interval = keepalive_interval
        self._listen_key: Optional[str] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """启动User Data Stream"""
        if self._running:
            return
        self._running = True

        self._listen_key = self.binance_api.get_listen_key()
        self._ws_task = asyncio.create_task(self._listen_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        logger.info("✅ User Data Stream已启动")

    async def stop(self) -> None:
        """停止User Data Stream"""
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._listen_key:
            try:
                self.binance_api.close_listen_key(self._listen_key)
            except Exception as e:
                logger.warning(f"关闭listenKey失败: {e}")
            self._listen_key = None
        logger.info("User Data Stream已停止")

    async def _keepalive_loop(self) -> None:
        while self._running and self._listen_key:
            try:
                await asyncio.sleep(self.keepalive_interval)
                self.binance_api.keepalive_listen_key(self._listen_key)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"listenKey续期失败: {e}")

    async def _listen_loop(self) -> None:
        while self._running and self._listen_key:
            url = self.binance_api.get_user_stream_url(self._listen_key)
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:  # type: ignore[attr-defined]
                    logger.info("User Data Stream连接成功")
                    async for msg in ws:
                        if not self._running:
                            break
                        self._handle_message(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"User Data Stream连接异常: {e}")
                await asyncio.sleep(2)

    def _handle_message(self, msg: str) -> None:
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            logger.warning("User Data Stream消息解析失败")
            return

        event_type = data.get("e") or data.get("eventType")
        if event_type in ("executionReport", "ORDER_TRADE_UPDATE"):
            report = self._normalize_execution_report(data)
            if report:
                try:
                    self.on_execution_report(report)
                except Exception as e:
                    logger.error(f"处理executionReport回调失败: {e}")
        elif event_type in ("ACCOUNT_UPDATE", "outboundAccountPosition"):
            account = self._normalize_account_update(data)
            if account and self.on_account_update is not None:
                try:
                    self.on_account_update(account)
                except Exception as e:
                    logger.error(f"处理accountUpdate回调失败: {e}")

    def _normalize_execution_report(
        self, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """统一执行回报字段（兼容期货与现货格式）"""
        if data.get("e") == "ORDER_TRADE_UPDATE":
            payload = data.get("o", {})
        else:
            payload = data

        if not payload:
            return None

        def _ms_to_s(ts: Optional[int]) -> Optional[int]:
            if ts is None:
                return None
            try:
                ts_int = int(ts)
            except (TypeError, ValueError):
                return None
            return ts_int // 1000 if ts_int > 10**12 else ts_int

        return {
            "order_id": str(payload.get("i") or payload.get("orderId") or ""),
            "client_order_id": payload.get("c") or payload.get("clientOrderId"),
            "symbol": payload.get("s") or payload.get("symbol"),
            "side": payload.get("S") or payload.get("side"),
            "order_type": payload.get("o") or payload.get("orderType"),
            "status": payload.get("X") or payload.get("orderStatus"),
            "execution_type": payload.get("x") or payload.get("executionType"),
            "last_filled_qty": float(payload.get("l") or 0),
            "filled_qty": float(payload.get("z") or 0),
            "last_filled_price": float(payload.get("L") or 0),
            "avg_price": float(payload.get("ap") or payload.get("avgPrice") or 0),
            "commission": float(payload.get("n") or payload.get("commission") or 0),
            "commission_asset": payload.get("N") or payload.get("commissionAsset"),
            "reject_reason": payload.get("r") or payload.get("rejectReason"),
            "realized_pnl": float(
                payload.get("rp") or payload.get("realizedPnl") or 0
            ),
            "is_maker": bool(payload.get("m", False)),
            "event_time": _ms_to_s(data.get("E")),
            "trade_time": _ms_to_s(payload.get("T") or payload.get("tradeTime")),
        }

    def _normalize_account_update(
        self, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """统一账户推送字段（兼容 Futures ACCOUNT_UPDATE / Spot outboundAccountPosition）"""
        event_type = data.get("e") or data.get("eventType")
        if event_type == "ACCOUNT_UPDATE":
            payload = data.get("a", {})
        else:
            payload = data
        if not isinstance(payload, dict):
            return None

        def _ms_to_s(ts: Optional[int]) -> Optional[int]:
            if ts is None:
                return None
            try:
                ts_int = int(ts)
            except (TypeError, ValueError):
                return None
            return ts_int // 1000 if ts_int > 10**12 else ts_int

        def _to_float(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        balances: List[Dict[str, Any]] = []
        for b in payload.get("B", []) or []:
            asset = str(b.get("a") or b.get("asset") or "").upper().strip()
            balances.append(
                {
                    "asset": asset,
                    "wallet_balance": _to_float(
                        b.get("wb") if "wb" in b else b.get("walletBalance")
                    ),
                    "cross_wallet_balance": _to_float(
                        b.get("cw") if "cw" in b else b.get("crossWalletBalance")
                    ),
                    "balance_change": _to_float(
                        b.get("bc") if "bc" in b else b.get("balanceChange")
                    ),
                }
            )

        positions: List[Dict[str, Any]] = []
        for p in payload.get("P", []) or []:
            positions.append(
                {
                    "symbol": str(p.get("s") or p.get("symbol") or "").upper().strip(),
                    "position_amount": _to_float(
                        p.get("pa") if "pa" in p else p.get("positionAmt")
                    ),
                    "entry_price": _to_float(
                        p.get("ep") if "ep" in p else p.get("entryPrice")
                    ),
                    "unrealized_pnl": _to_float(
                        p.get("up") if "up" in p else p.get("unrealizedProfit")
                    ),
                    "position_side": str(p.get("ps") or p.get("positionSide") or "")
                    .upper()
                    .strip(),
                }
            )

        usdt_balance = next(
            (
                b
                for b in balances
                if str(b.get("asset", "")).upper() in {"USDT", "USDC"}
            ),
            None,
        )
        wallet_balance = (
            float(usdt_balance.get("wallet_balance", 0.0)) if usdt_balance else 0.0
        )
        available_balance = (
            float(usdt_balance.get("cross_wallet_balance", 0.0))
            if usdt_balance
            else 0.0
        )
        unrealized_pnl_total = float(
            sum(float(p.get("unrealized_pnl", 0.0) or 0.0) for p in positions)
        )

        return {
            "event_type": str(event_type or ""),
            "event_time": _ms_to_s(data.get("E")),
            "transaction_time": _ms_to_s(data.get("T")),
            "reason": payload.get("m"),
            "balances": balances,
            "positions": positions,
            "wallet_balance": wallet_balance,
            "available_balance": available_balance,
            "unrealized_pnl_total": unrealized_pnl_total,
        }
