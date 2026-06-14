"""
Binance API封装
使用ccxt实现REST API调用
支持代理配置（用于主网连接）
"""

import logging
import os
import threading
import time
import hmac
import hashlib
from urllib.parse import urlencode
from typing import Callable, Optional, Dict, Any, List, TypeVar
from datetime import datetime
import ccxt
import requests

from .models import Order, OrderSide, OrderType, OrderStatus

logger = logging.getLogger(__name__)
T = TypeVar("T")
_BINANCE_READ_LOCK = threading.Lock()
_BINANCE_READ_LAST_AT = 0.0


# Binance error codes that mean "this order/algo does not (or no longer) exists".
_ORDER_MISS_CODES = (-2011, -2013, -2039)


def _binance_http_error_payload(exc: BaseException) -> Optional[Dict[str, Any]]:
    """Return {code, msg} parsed from a requests.HTTPError body, if available."""
    if not isinstance(exc, requests.HTTPError) or exc.response is None:
        return None
    try:
        body = (exc.response.text or "").strip()
    except Exception:
        body = ""
    if not body:
        return {"status": exc.response.status_code, "code": None, "msg": ""}
    try:
        data = exc.response.json()
    except Exception:
        return {"status": exc.response.status_code, "code": None, "msg": body}
    if not isinstance(data, dict):
        return {"status": exc.response.status_code, "code": None, "msg": body}
    return {
        "status": exc.response.status_code,
        "code": data.get("code"),
        "msg": str(data.get("msg", "")),
    }


def _is_order_lookup_miss(exc: BaseException) -> bool:
    """True when an order/algo lookup target does not exist on the exchange.

    Conservative on HTTP 400: only treated as a miss when the body carries a
    known not-found code/message, or when it is a bare 400 with no structured
    error payload (Binance's algo GET returns this for unknown clientAlgoId).
    A 400 with any *other* explicit error code (e.g. -1021/-1022 clock/signature)
    is NOT a miss and propagates, so systemic failures stay loud.
    """
    payload = _binance_http_error_payload(exc)
    if payload is not None:
        status = payload["status"]
        code = payload["code"]
        msg = str(payload["msg"]).lower()
        if status == 404:
            return True
        if status == 400:
            if code in _ORDER_MISS_CODES:
                return True
            if any(
                token in msg
                for token in ("not found", "unknown order", "does not exist")
            ):
                return True
            # Bare 400 with no structured error => unknown order (algo GET).
            return code is None and not msg
        return False
    name = type(exc).__name__.lower()
    if "ordernotfound" in name or "order not found" in name:
        return True
    text = str(exc)
    msg = text.lower()
    return (
        "not found" in msg
        or any(str(c) in text for c in _ORDER_MISS_CODES)
        or "unknown order" in msg
        or "order does not exist" in msg
    )


def _is_cancel_target_gone(exc: BaseException) -> bool:
    """True when cancel target is already filled/canceled (Binance -2011, etc.)."""
    return _is_order_lookup_miss(exc)


def _is_binance_rate_limit_detail(text: str) -> bool:
    """Recognize Binance / ccxt wording for REST rate limits (-1003, HTTP 429)."""
    chunk = str(text or "")
    lowered = chunk.lower()
    return (
        "-1003" in chunk
        or "429" in chunk
        or "too many requests" in lowered
        or "ddosprotection" in lowered.replace(" ", "")
    )


def _binance_read_retry_max() -> int:
    raw = os.getenv("MLBOT_BINANCE_REST_RETRY_MAX", "8")
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return 8


def _binance_read_retry_sleep_seconds(attempt: int) -> float:
    try:
        base = float(os.getenv("MLBOT_BINANCE_REST_RETRY_BASE_SECONDS", "3.0"))
    except ValueError:
        base = 3.0
    try:
        cap = float(os.getenv("MLBOT_BINANCE_REST_RETRY_MAX_SECONDS", "90.0"))
    except ValueError:
        cap = 90.0
    return min(base * (2**attempt), cap)


def _binance_read_min_interval_seconds() -> float:
    raw = os.getenv("MLBOT_BINANCE_REST_READ_MIN_INTERVAL_SECONDS", "0.25")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.25


def _throttle_binance_read() -> None:
    """Process-wide pacing for Binance read-only REST calls."""
    global _BINANCE_READ_LAST_AT
    min_interval = _binance_read_min_interval_seconds()
    if min_interval <= 0:
        return
    with _BINANCE_READ_LOCK:
        now = time.monotonic()
        sleep_for = (_BINANCE_READ_LAST_AT + min_interval) - now
        if sleep_for > 0:
            time.sleep(sleep_for)
            now = time.monotonic()
        _BINANCE_READ_LAST_AT = now


def _call_binance_read(op_name: str, fn: Callable[[], T]) -> T:
    """Throttle and retry read-only Binance/ccxt REST calls under IP limits."""
    retries = _binance_read_retry_max()
    last_exc: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            _throttle_binance_read()
            return fn()
        except BaseException as exc:
            last_exc = exc
            if (
                attempt + 1 >= retries
                or not _is_binance_rate_limit_detail(str(exc))
            ):
                raise
            delay = _binance_read_retry_sleep_seconds(attempt)
            logger.warning(
                "Binance REST read %s rate limited (%s); retry %s/%s in %.1fs",
                op_name,
                exc,
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None  # pragma: no cover - loop returns or raises
    raise last_exc


def _ccxt_linear_usdt_perp_symbol(symbol: Optional[str]) -> Optional[str]:
    """Map ``BTCUSDT`` -> ``BTC/USDT:USDT`` for ccxt Binance USDT-M perps."""
    if not symbol:
        return None
    s = str(symbol).strip().upper()
    if "/" in s or ":" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/USDT:USDT"
    return s


class BinanceAPI:
    """Binance API封装（使用ccxt）"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        sandbox: bool = False,
        use_proxy: Optional[bool] = None,
        proxy_type: str = "socks5",
        proxy_host: Optional[str] = None,
        proxy_port: int = 7897,
    ):
        """
        初始化Binance API客户端

        Args:
            api_key: API密钥
            api_secret: API密钥
            testnet: 是否使用测试网
            sandbox: 是否使用沙箱环境（与testnet相同）
            use_proxy: 是否使用代理（None 表示从环境变量 USE_SOCKS5_PROXY / HTTP_PROXY 推断）
            proxy_type: 代理类型 ('socks5' 或 'http')
            proxy_host: 代理主机地址（None表示从环境变量或默认值获取）
            proxy_port: 代理端口
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet or sandbox

        # 代理配置（主网和测试网都可以使用）
        if use_proxy is None:
            # 从环境变量读取
            use_proxy = os.environ.get("USE_SOCKS5_PROXY", "false").lower() == "true"
            # HTTP(S) 代理：主网 / 测试网都应支持（本机 Clash 等常见为 7890 mixed）
            if not use_proxy:
                has_http_proxy = bool(
                    os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
                )
                if has_http_proxy:
                    use_proxy = True
                    proxy_type = "http"  # 环境变量中的代理通常是 HTTP 代理

        self.use_proxy = use_proxy  # 允许测试网也使用代理
        self.proxy_type = proxy_type
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port

        # 创建ccxt交易所实例
        exchange_options = {
            "defaultType": "future",  # 使用合约交易
            "enableRateLimit": True,
            "timeout": 30000,  # 增加超时时间到30秒
            "options": {
                "defaultType": "future",
                "fetchCurrencies": False,  # 禁用货币配置获取（避免sapi端点超时）
            },
        }

        # 测试网特殊配置：禁用所有sapi端点调用
        if self.testnet:
            exchange_options["options"]["fetchMarkets"] = {
                "type": "future",  # 只获取合约市场，不调用sapi
            }

        # 配置代理（主网和测试网都可以使用）
        if self.use_proxy:
            # 获取代理地址
            if self.proxy_host is None:
                # 从环境变量获取
                env_proxy = os.environ.get("HTTP_PROXY") or os.environ.get(
                    "HTTPS_PROXY"
                )
                if env_proxy and "://" in env_proxy:
                    try:
                        from urllib.parse import urlparse

                        parsed = urlparse(env_proxy)
                        if parsed.hostname:
                            self.proxy_host = parsed.hostname
                            if parsed.port:
                                self.proxy_port = parsed.port
                    except Exception:
                        pass

                # 如果还没有，使用默认值
                if self.proxy_host is None:
                    self.proxy_host = "127.0.0.1"

            # 配置代理URL
            if self.proxy_type.lower() == "socks5":
                proxy_url = f"socks5://{self.proxy_host}:{self.proxy_port}"
            elif self.proxy_type.lower() == "http":
                proxy_url = f"http://{self.proxy_host}:{self.proxy_port}"
            else:
                logger.warning(f"不支持的代理类型: {self.proxy_type}，将不使用代理")
                self.use_proxy = False

            if self.use_proxy:
                exchange_options["proxies"] = {
                    "http": proxy_url,
                    "https": proxy_url,
                }
                logger.info(f"✅ 已配置代理: {proxy_url}")

        if self.testnet:
            # 测试网配置
            exchange_options["urls"] = {
                "api": {
                    "public": "https://testnet.binancefuture.com",
                    "private": "https://testnet.binancefuture.com",
                }
            }

        self.exchange = ccxt.binance(
            {"apiKey": api_key, "secret": api_secret, **exchange_options}
        )
        # ``MultiLegLiveOrchestrator`` syncs all open orders with ``symbol=None``;
        # ccxt-binance otherwise raises ExchangeError until this flag is set.
        self.exchange.options["warnOnFetchOpenOrdersWithoutSymbol"] = False

        # Monkey patch nonce 方法以自动修正时间偏移
        original_nonce = self.exchange.nonce

        def patched_nonce():
            """Binance 的 nonce 就是 timestamp，自动应用偏移修正"""
            timestamp = original_nonce()
            # 如果有时间偏移，自动修正
            if hasattr(self, "time_offset"):
                timestamp += self.time_offset
            return int(timestamp)

        self.exchange.nonce = patched_nonce

        # 如果是测试网，确保URLs正确设置（在创建后手动设置）
        if self.testnet:
            testnet_base = "https://testnet.binancefuture.com"
            # 设置所有API端点URL
            self.exchange.urls["api"] = {
                "public": testnet_base,
                "private": testnet_base,
                "fapiPublic": testnet_base,  # 合约公开端点
                "fapiPrivate": testnet_base,  # 合约私有端点
                "fapiPrivateV3": testnet_base,  # 合约私有端点V3
            }
            # 确保baseURLs也正确
            if hasattr(self.exchange, "base_urls"):
                self.exchange.base_urls["api"] = {
                    "public": testnet_base,
                    "private": testnet_base,
                    "fapiPublic": testnet_base,
                    "fapiPrivate": testnet_base,
                    "fapiPrivateV3": testnet_base,
                }
            # 禁用load_markets时的货币配置获取（测试网不支持sapi端点）
            self.exchange.options["fetchCurrencies"] = False

            # Monkey patch sign方法，允许测试网使用fapiPrivate端点
            original_sign = self.exchange.sign

            def patched_sign(
                path, api="public", method="GET", params=None, headers=None, body=None
            ):
                """测试网版本的sign方法，允许fapiPrivate和fapiPrivateV3端点"""
                # 如果是fapiPrivate相关端点，确保URL正确
                if api in ["fapiPrivate", "fapiPrivateV3"]:
                    # 确保URLs中有对应的端点
                    if api not in self.exchange.urls.get("api", {}):
                        self.exchange.urls["api"][api] = testnet_base
                return original_sign(path, api, method, params, headers, body)

            self.exchange.sign = patched_sign

            # 禁用margin相关调用（测试网不支持sapi端点）
            # 通过monkey patch禁用sapiGetMarginAllPairs调用
            original_fetch_markets = self.exchange.fetch_markets

            def patched_fetch_markets(params=None):
                """测试网版本的fetch_markets，禁用sapi调用"""
                # 直接调用测试网API获取exchangeInfo
                try:
                    import requests

                    url = "https://testnet.binancefuture.com/fapi/v1/exchangeInfo"
                    # 如果使用代理，配置代理
                    proxies = None
                    if self.use_proxy:
                        proxy_url = (
                            f"{self.proxy_type}://{self.proxy_host}:{self.proxy_port}"
                        )
                        proxies = {
                            "http": proxy_url,
                            "https": proxy_url,
                        }

                    resp = requests.get(url, timeout=30, proxies=proxies)
                    if resp.status_code == 200:
                        response = resp.json()
                        # 解析markets - ccxt的parse_markets期望接收整个response字典
                        # 需要将response包装成ccxt期望的格式
                        markets = []
                        for symbol_info in response.get("symbols", []):
                            # 只处理永续合约
                            if symbol_info.get("contractType") == "PERPETUAL":
                                market = self.exchange.parse_market(symbol_info)
                                if market:
                                    markets.append(market)
                        return markets
                    else:
                        raise Exception(f"API请求失败: {resp.status_code}, {resp.text}")
                except Exception as e:
                    logger.error(f"测试网fetch_markets失败: {e}")
                    raise

            # 替换fetch_markets方法
            self.exchange.fetch_markets = patched_fetch_markets

        # 预加载markets（避免每次下单都加载，提高性能）
        try:
            logger.info("正在加载markets信息...")
            self.exchange.load_markets()
            logger.info("✅ Markets信息已加载")
        except Exception as e:
            logger.warning(f"⚠️ 预加载markets失败，将在下单时自动加载: {e}")

        # 检测服务器时间偏移
        self._check_time_sync()

        # 检测账户是否开启 Hedge Mode（双向持仓）
        self.hedge_mode, self.hedge_mode_probe_error = self._detect_hedge_mode()

        # 如果是测试网，确保URLs正确设置（在创建后手动设置）
        if self.testnet:
            self.exchange.urls["api"] = {
                "public": "https://testnet.binancefuture.com",
                "private": "https://testnet.binancefuture.com",
            }
            # 确保baseURLs也正确
            if hasattr(self.exchange, "base_urls"):
                self.exchange.base_urls["api"] = {
                    "public": "https://testnet.binancefuture.com",
                    "private": "https://testnet.binancefuture.com",
                }

    def _detect_hedge_mode(self) -> tuple[bool, Optional[str]]:
        """检测 Binance 账户 Hedge Mode。

        Returns:
            (dual_side_enabled, probe_error). ``probe_error`` 非 None 表示
            REST 探测失败（鉴权/IP/权限等），不应与「确认为 One-way」混淆。
        """
        try:
            url = f"{self._get_futures_base_url()}/fapi/v1/positionSide/dual"
            headers = {"X-MBX-APIKEY": self.api_key}
            # 需要签名
            import time
            import hmac
            import hashlib

            ts = int(time.time() * 1000) + getattr(self, "time_offset", 0)
            query = f"timestamp={ts}"
            sig = hmac.new(
                self.api_secret.encode(), query.encode(), hashlib.sha256
            ).hexdigest()
            query += f"&signature={sig}"
            resp = requests.get(
                f"{url}?{query}",
                headers=headers,
                timeout=10,
                proxies=self._get_requests_proxies(),
            )
            try:
                data = resp.json()
            except ValueError:
                data = {}

            # Binance 可能在 HTTP 401/418 等状态体中返回 JSON {code,msg}
            if isinstance(data, dict):
                bc = data.get("code")
                if bc is not None and bc != 200:
                    try:
                        if int(bc) < 0:
                            return False, f"Binance error {bc}: {data.get('msg')}"
                    except (TypeError, ValueError):
                        pass

            # 典型成功体: {"dualSidePosition": true/false}
            if isinstance(data, dict) and "dualSidePosition" in data:
                is_hedge = bool(data.get("dualSidePosition", False))
                if is_hedge:
                    logger.info(
                        "✅ 检测到账户开启了 Hedge Mode（双向持仓），将自动注入 positionSide 参数"
                    )
                else:
                    logger.info("✅ 账户为 One-way Mode（单向持仓）")
                return is_hedge, None

            # 错误响应（常为 HTTP200 + Binance JSON code，也可能是 4xx）
            err_snip = ""
            if isinstance(data, dict) and data.get("code") is not None:
                err_snip = f" Binance code={data.get('code')} msg={data.get('msg')}"
            if resp.status_code >= 400:
                body = ""
                try:
                    body = (resp.text or "")[:400]
                except Exception:
                    pass
                return False, f"HTTP {resp.status_code}{err_snip} body={body!r}"

            if isinstance(data, dict):
                return False, f"unexpected hedge probe JSON:{err_snip} keys={list(data.keys())}"

            return False, f"unexpected hedge probe response ({resp.status_code}): {data!r}"[:400]
        except Exception as e:
            logger.warning("⚠️ 无法检测持仓模式（网络/签名等）: %s", e)
            return False, f"{type(e).__name__}: {e}"

    def refresh_hedge_mode(self) -> None:
        """重新拉取 Hedge / One-way 状态（例如在 POST 切换持仓模式之后）。"""
        self.hedge_mode, self.hedge_mode_probe_error = self._detect_hedge_mode()

    def _get_futures_base_url(self) -> str:
        """获取期货REST基础URL"""
        if self.testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"

    def _get_futures_ws_base(self) -> str:
        """获取期货User Data Stream WS基础URL"""
        if self.testnet:
            return "wss://stream.binancefuture.com/ws"
        return "wss://fstream.binance.com/ws"

    def _get_requests_proxies(self) -> Optional[Dict[str, str]]:
        """构建 requests 代理配置"""
        if not self.use_proxy:
            return None
        proxy_url = f"{self.proxy_type}://{self.proxy_host}:{self.proxy_port}"
        return {"http": proxy_url, "https": proxy_url}

    def _futures_market_id(self, symbol: Optional[str]) -> Optional[str]:
        """Binance 合约 REST 用的原生 symbol，如 ``BTCUSDT``。"""
        if not symbol:
            return None
        s = str(symbol).strip().upper()
        try:
            self.exchange.load_markets()
            ccxt_sym = _ccxt_linear_usdt_perp_symbol(s) or s
            if ccxt_sym in self.exchange.markets:
                return str(self.exchange.markets[ccxt_sym]["id"])
            if s in self.exchange.markets:
                return str(self.exchange.markets[s]["id"])
        except Exception:
            pass
        if "/" in s or ":" in s:
            return None
        return s if s.endswith("USDT") else None

    def _fapi_signed_get(self, path: str, params: Dict[str, Any]) -> Any:
        """
        对 ``/fapi/v1/...`` 做签名 GET。

        在部分环境下 ccxt 对 testnet 的 ``fapiPrivate*`` 会拿到空 body（解析成 ``""``），
        与 Binance 实际返回的 JSON 不一致；关键读接口用本方法走 ``requests`` 更可靠。
        """
        p = dict(params)
        if "timestamp" not in p:
            p["timestamp"] = int(time.time() * 1000) + getattr(
                self, "time_offset", 0
            )
        query = urlencode(sorted((k, v) for k, v in p.items() if v is not None))
        sig = hmac.new(
            self.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        url = f"{self._get_futures_base_url()}{path}?{query}&signature={sig}"
        resp = _call_binance_read(
            f"GET {path}",
            lambda: requests.get(
                url,
                headers={"X-MBX-APIKEY": self.api_key},
                timeout=30,
                proxies=self._get_requests_proxies(),
            ),
        )
        resp.raise_for_status()
        text = (resp.text or "").strip()
        if not text:
            return []
        data = resp.json()
        if isinstance(data, dict) and "code" in data and "msg" in data:
            raise RuntimeError(
                f"Binance futures API error {data.get('code')}: {data.get('msg')}"
            )
        return data

    def _fapi_signed_post(self, path: str, params: Dict[str, Any]) -> Any:
        """签名 POST ``/fapi/v1/...``（application/x-www-form-urlencoded body）。"""
        p_flat: Dict[str, str] = {}
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, bool):
                p_flat[k] = "true" if v else "false"
            else:
                p_flat[k] = str(v)
        if "timestamp" not in p_flat:
            p_flat["timestamp"] = str(
                int(time.time() * 1000) + getattr(self, "time_offset", 0)
            )
        query = urlencode(sorted(p_flat.items()))
        sig = hmac.new(
            self.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        body = query + "&signature=" + sig
        url = f"{self._get_futures_base_url()}{path}"
        resp = requests.post(
            url,
            headers={
                "X-MBX-APIKEY": self.api_key,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=body,
            timeout=30,
            proxies=self._get_requests_proxies(),
        )
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            raise RuntimeError(
                f"fapi POST {path}: non-JSON response {resp.status_code}: "
                f"{(resp.text or '')[:240]!r}"
            )
        if isinstance(data, dict):
            bc = data.get("code")
            if bc is not None:
                try:
                    code_i = int(bc)
                    if code_i == 200:
                        return data
                    if code_i < 0:
                        raise RuntimeError(
                            f"Binance futures API error {bc}: {data.get('msg')}"
                        )
                except (TypeError, ValueError):
                    pass
        if resp.status_code >= 400:
            raise RuntimeError(
                f"fapi POST {path} HTTP {resp.status_code}: {(resp.text or '')[:400]!r}"
            )
        return data

    def _fapi_signed_delete(self, path: str, params: Dict[str, Any]) -> Any:
        """签名 DELETE ``/fapi/v1/...``."""
        p = dict(params)
        if "timestamp" not in p:
            p["timestamp"] = int(time.time() * 1000) + getattr(
                self, "time_offset", 0
            )
        query = urlencode(sorted((k, v) for k, v in p.items() if v is not None))
        sig = hmac.new(
            self.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        url = f"{self._get_futures_base_url()}{path}?{query}&signature={sig}"
        resp = _call_binance_read(
            f"DELETE {path}",
            lambda: requests.delete(
                url,
                headers={"X-MBX-APIKEY": self.api_key},
                timeout=30,
                proxies=self._get_requests_proxies(),
            ),
        )
        resp.raise_for_status()
        text = (resp.text or "").strip()
        if not text:
            return {}
        data = resp.json()
        if isinstance(data, dict) and "code" in data and "msg" in data:
            try:
                if int(data.get("code")) < 0:
                    raise RuntimeError(
                        f"Binance futures API error {data.get('code')}: {data.get('msg')}"
                    )
            except (TypeError, ValueError):
                pass
        return data

    def set_dual_side_position(self, enabled: bool) -> Dict[str, Any]:
        """切换 USD-M 持仓模式：``True``=Hedge（双向），``False``=One-way。

        调用 ``POST /fapi/v1/positionSide/dual``。若有未平仓位或未结订单，
        Binance 会拒绝切换。

        Raises:
            RuntimeError: Binance 业务错误或 HTTP 失败。
        """
        result = self._fapi_signed_post(
            "/fapi/v1/positionSide/dual",
            {"dualSidePosition": "true" if enabled else "false"},
        )
        if not isinstance(result, dict):
            return {}
        return result

    def _open_order_from_binance_rest(self, o: Dict[str, Any]) -> Dict[str, Any]:
        """将 ``/fapi/v1/openOrders`` 单行转为与 ccxt 分支一致的 dict。"""
        price_raw = o.get("price")
        try:
            price_f = float(price_raw) if price_raw is not None else 0.0
        except (TypeError, ValueError):
            price_f = 0.0
        price = price_f if price_f else None
        orig = float(o.get("origQty") or 0)
        filled = float(o.get("executedQty") or 0)
        rem = max(orig - filled, 0.0)
        ts = o.get("time") or o.get("updateTime")
        ts_ms = int(ts) if ts is not None else None
        return {
            "order_id": str(o.get("orderId", "")),
            "client_order_id": o.get("clientOrderId"),
            "symbol": o.get("symbol"),
            "side": (o.get("side") or "").lower(),
            "type": (o.get("type") or "").lower(),
            "status": (o.get("status") or "").lower(),
            "quantity": orig,
            "price": price,
            "filled": filled,
            "remaining": rem,
            "created_at": self._normalize_timestamp(ts_ms),
            "info": o,
        }

    def _get_open_orders_fapi_rest(self, symbol: Optional[str]) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        mid = self._futures_market_id(symbol)
        if mid:
            params["symbol"] = mid
        raw = self._fapi_signed_get("/fapi/v1/openOrders", params)
        if not isinstance(raw, list):
            raise ValueError(f"openOrders expected list, got {type(raw)}: {raw!r}")
        return [
            self._open_order_from_binance_rest(x)
            for x in raw
            if isinstance(x, dict)
        ]

    def _open_algo_order_from_binance_rest(self, o: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize ``/fapi/v1/openAlgoOrders`` row for SL cleanup (-4130)."""
        algo_id = o.get("algoId") or o.get("orderId")
        return {
            "order_id": str(algo_id or ""),
            "client_order_id": o.get("clientAlgoId") or o.get("clientOrderId"),
            "symbol": o.get("symbol"),
            "side": (o.get("side") or "").lower(),
            "type": (o.get("orderType") or o.get("type") or "algo").lower(),
            "status": (o.get("algoStatus") or o.get("status") or "open").lower(),
            "quantity": float(o.get("quantity") or o.get("origQty") or 0),
            "stop_price": o.get("triggerPrice") or o.get("stopPrice"),
            "price": None,
            "filled": 0.0,
            "remaining": float(o.get("quantity") or o.get("origQty") or 0),
            "created_at": self._normalize_timestamp(
                o.get("createTime") or o.get("updateTime")
            ),
            "info": o,
            "_is_algo_order": True,
        }

    def _get_open_algo_orders_fapi_rest(
        self, symbol: Optional[str]
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        mid = self._futures_market_id(symbol)
        if mid:
            params["symbol"] = mid
        try:
            raw = self._fapi_signed_get("/fapi/v1/openAlgoOrders", params)
        except Exception as exc:
            logger.debug("openAlgoOrders unavailable: %s", exc)
            return []
        if not isinstance(raw, list):
            return []
        return [
            self._open_algo_order_from_binance_rest(x)
            for x in raw
            if isinstance(x, dict)
        ]

    def get_open_orders_for_sl_cleanup(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Open orders via fapi REST + algo channel (closePosition lives in either)."""
        orders = self._get_open_orders_fapi_rest(symbol)
        seen = {str(o.get("order_id") or "") for o in orders if o.get("order_id")}
        for row in self._get_open_algo_orders_fapi_rest(symbol):
            oid = str(row.get("order_id") or "")
            if oid and oid not in seen:
                orders.append(row)
                seen.add(oid)
        return orders

    def cancel_algo_order(self, algo_id: str, symbol: str) -> bool:
        """Cancel one open algo/conditional order by ``algoId``."""
        mid = self._futures_market_id(symbol)
        if not mid or not str(algo_id or "").strip():
            return False
        self._fapi_signed_delete(
            "/fapi/v1/algoOrder",
            {"symbol": mid, "algoId": str(algo_id).strip()},
        )
        return True

    def _check_time_sync(self) -> None:
        """
        检查本地时间与 Binance 服务器时间的偏移

        Binance 要求时间误差 < 1000ms，否则订单会被拒绝
        """
        try:
            url = f"{self._get_futures_base_url()}/fapi/v1/time"
            resp = _call_binance_read(
                "GET /fapi/v1/time",
                lambda: requests.get(
                    url, timeout=5, proxies=self._get_requests_proxies()
                ),
            )
            resp.raise_for_status()
            server_time = resp.json()["serverTime"]
            local_time = int(datetime.now().timestamp() * 1000)
            time_diff = server_time - local_time  # 服务器时间 - 本地时间

            # 保存时间偏移，用于后续签名时自动修正
            self.time_offset = time_diff

            if abs(time_diff) > 1000:
                logger.warning(f"⚠️ 时间偏移过大: {abs(time_diff)}ms > 1000ms")
                logger.warning(f"⚠️ 将自动修正时间偏移: {time_diff}ms")
            else:
                logger.info(f"✅ 时间同步正常: 偏移 {time_diff}ms")
        except Exception as e:
            logger.warning(f"⚠️ 无法检查时间同步: {e}")
            self.time_offset = 0

    def get_listen_key(self) -> str:
        """创建User Data Stream listenKey（合约）"""
        url = f"{self._get_futures_base_url()}/fapi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self.api_key}
        resp = requests.post(
            url, headers=headers, timeout=10, proxies=self._get_requests_proxies()
        )
        resp.raise_for_status()
        return resp.json()["listenKey"]

    def keepalive_listen_key(self, listen_key: str) -> None:
        """续期User Data Stream listenKey"""
        url = f"{self._get_futures_base_url()}/fapi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self.api_key}
        params = {"listenKey": listen_key}
        resp = requests.put(
            url,
            headers=headers,
            params=params,
            timeout=10,
            proxies=self._get_requests_proxies(),
        )
        resp.raise_for_status()

    def close_listen_key(self, listen_key: str) -> None:
        """关闭User Data Stream listenKey"""
        url = f"{self._get_futures_base_url()}/fapi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self.api_key}
        params = {"listenKey": listen_key}
        resp = requests.delete(
            url,
            headers=headers,
            params=params,
            timeout=10,
            proxies=self._get_requests_proxies(),
        )
        resp.raise_for_status()

    def get_user_stream_url(self, listen_key: str) -> str:
        """获取User Data Stream WS URL"""
        return f"{self._get_futures_ws_base()}/{listen_key}"

    # ========== 账户信息 ==========

    def get_account_balance(self) -> Dict[str, Any]:
        """
        获取账户余额

        Returns:
            账户余额信息
        """
        try:
            balance = _call_binance_read(
                "fetch_balance", lambda: self.exchange.fetch_balance()
            )
            return balance
        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
            raise

    def get_account_info(self) -> Dict[str, Any]:
        """
        获取账户信息

        Returns:
            账户信息
        """
        try:
            # ccxt的fetch_balance已经包含账户信息
            balance = self.get_account_balance()
            return {
                "total_balance": balance.get("USDT", {}).get("total", 0),
                "free_balance": balance.get("USDT", {}).get("free", 0),
                "used_balance": balance.get("USDT", {}).get("used", 0),
                "info": balance.get("info", {}),
            }
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            raise

    def get_margin_info(self) -> Dict[str, Any]:
        """
        获取保证金信息

        Returns:
            保证金信息
        """
        try:
            balance = self.get_account_balance()
            return {
                "total_margin": balance.get("USDT", {}).get("total", 0),
                "available_margin": balance.get("USDT", {}).get("free", 0),
                "used_margin": balance.get("USDT", {}).get("used", 0),
                "margin_ratio": 0.0,  # 需要从info中计算
            }
        except Exception as e:
            logger.error(f"获取保证金信息失败: {e}")
            raise

    # ========== 仓位查询 ==========

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取仓位信息

        Args:
            symbol: 交易对符号，None表示获取所有仓位

        Returns:
            仓位列表
        """
        try:
            ccxt_sym = _ccxt_linear_usdt_perp_symbol(symbol)
            positions = _call_binance_read(
                "fetch_positions",
                lambda: self.exchange.fetch_positions(
                    symbols=[ccxt_sym] if ccxt_sym else None
                ),
            )
            result = []
            for pos in positions:
                if pos["contracts"] != 0:  # 只返回有仓位的
                    result.append(
                        {
                            "symbol": pos["symbol"],
                            "side": pos["side"],
                            "size": pos["contracts"],
                            "entry_price": pos["entryPrice"],
                            "mark_price": pos["markPrice"],
                            "unrealized_pnl": pos["unrealizedPnl"],
                            "percentage": pos["percentage"],
                            "leverage": pos.get("leverage", 1),
                            "notional": pos.get("notional", 0),
                            "margin_mode": pos.get("marginMode", "isolated"),
                            "liquidation_price": pos.get("liquidationPrice"),
                        }
                    )
            return result
        except Exception as e:
            logger.error(f"获取仓位信息失败: {e}")
            raise

    @staticmethod
    def _normalize_timestamp(ts: Optional[int]) -> Optional[int]:
        """将毫秒时间戳统一为秒级（int）"""
        if ts is None:
            return None
        try:
            ts_int = int(ts)
        except (TypeError, ValueError):
            return None
        # 13位为毫秒时间戳
        if ts_int > 10**12:
            return ts_int // 1000
        return ts_int

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取指定交易对的仓位

        Args:
            symbol: 交易对符号

        Returns:
            仓位信息，如果没有仓位返回None
        """
        positions = self.get_positions(symbol)
        return positions[0] if positions else None

    # ========== 订单操作 ==========

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        close_position: bool = False,
        client_order_id: Optional[str] = None,
        position_side: Optional[str] = None,
        working_type: Optional[str] = None,
        price_protect: Optional[bool] = None,
        post_only: bool = False,
        time_in_force: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        下单

        Args:
            symbol: 交易对符号
            side: 订单方向
            order_type: 订单类型
            quantity: 数量
            price: 价格（限价单需要）
            stop_price: 止损价格（止损单需要）
            reduce_only: 是否只减仓
            close_position: 是否平仓
            position_side: Hedge Mode 持仓方向（LONG/SHORT），None 表示自动推断
            working_type: 条件单触发价格类型（如 MARK_PRICE 或 CONTRACT_PRICE）
            price_protect: Binance Futures priceProtect 参数
            post_only: 是否作为 post-only 限价单提交
            time_in_force: Binance Futures timeInForce（post-only 使用 GTX）

        Returns:
            订单信息
        """
        try:
            ccxt_sym = _ccxt_linear_usdt_perp_symbol(symbol) or symbol
            # 转换订单类型
            ccxt_side = "buy" if side == OrderSide.BUY else "sell"
            ccxt_type = self._convert_order_type(order_type)

            params: Dict[str, Any] = {}
            explicit_position_side = (
                str(position_side).strip().upper() if position_side else ""
            )
            if explicit_position_side and explicit_position_side not in {
                "LONG",
                "SHORT",
            }:
                raise ValueError(f"position_side must be LONG/SHORT: {position_side}")
            if self.hedge_mode:
                # P0 defense-in-depth: MARKET + closePosition=True triggers -4136 on Binance.
                # STOP_MARKET / TAKE_PROFIT_MARKET conditional orders may use closePosition
                # with positionSide; MARKET (immediate-fill) orders must use qty+reduceOnly.
                _immediate_fill_types = {OrderType.MARKET, OrderType.LIMIT}
                if close_position and order_type in _immediate_fill_types:
                    raise ValueError(
                        f"Hedge Mode 下 {order_type.value} 平仓不能使用 closePosition=True "
                        f"(Binance -4136); 应使用 positionSide + qty 或 reduceOnly + qty"
                    )
                # Hedge Mode 下必须指定 positionSide，且不能使用 reduceOnly
                # 开仓: BUY→LONG, SELL→SHORT
                # 平仓(reduce_only): BUY→SHORT, SELL→LONG
                if explicit_position_side:
                    params["positionSide"] = explicit_position_side
                elif reduce_only or close_position:
                    # 平仓：方向与仓位相反
                    params["positionSide"] = "SHORT" if ccxt_side == "buy" else "LONG"
                else:
                    # 开仓：方向与 side 一致
                    params["positionSide"] = "LONG" if ccxt_side == "buy" else "SHORT"
                # Hedge Mode 下 conditional closePosition 有效 (STOP_MARKET / TP_MARKET);
                # reduceOnly 无效（已用 positionSide 代替）
                if close_position:
                    params["closePosition"] = True
            else:
                if reduce_only:
                    params["reduceOnly"] = True
                if close_position:
                    params["closePosition"] = True
            if client_order_id:
                # Binance Futures支持newClientOrderId
                params["newClientOrderId"] = client_order_id
                params["clientOrderId"] = client_order_id
            if working_type:
                wt = str(working_type).strip().upper()
                if wt not in {"MARK_PRICE", "CONTRACT_PRICE"}:
                    raise ValueError(
                        f"working_type must be MARK_PRICE/CONTRACT_PRICE: {working_type}"
                    )
                params["workingType"] = wt
            if price_protect is not None:
                params["priceProtect"] = "TRUE" if price_protect else "FALSE"
            tif = str(time_in_force or "").strip().upper()
            if post_only:
                tif = tif or "GTX"
            if tif:
                if tif not in {"GTC", "IOC", "FOK", "GTX"}:
                    raise ValueError(f"time_in_force must be GTC/IOC/FOK/GTX: {tif}")
                params["timeInForce"] = tif

            if order_type == OrderType.MARKET:
                order = self.exchange.create_market_order(
                    ccxt_sym, ccxt_side, quantity, params=params
                )
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("限价单需要指定价格")
                order = self.exchange.create_limit_order(
                    ccxt_sym, ccxt_side, quantity, price, params=params
                )
            elif order_type in [OrderType.STOP, OrderType.STOP_MARKET]:
                if stop_price is None:
                    raise ValueError("止损单需要指定止损价格")
                params["stopPrice"] = stop_price
                if order_type == OrderType.STOP_MARKET:
                    params["type"] = "STOP_MARKET"
                amount = None if close_position else quantity
                order = self.exchange.create_order(
                    ccxt_sym, ccxt_type, ccxt_side, amount, price, params=params
                )
            elif order_type in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_MARKET]:
                if stop_price is None:
                    raise ValueError("止盈单需要指定止盈价格")
                params["stopPrice"] = stop_price
                if order_type == OrderType.TAKE_PROFIT_MARKET:
                    params["type"] = "TAKE_PROFIT_MARKET"
                amount = None if close_position else quantity
                order = self.exchange.create_order(
                    ccxt_sym, ccxt_type, ccxt_side, amount, price, params=params
                )
            else:
                raise ValueError(f"不支持的订单类型: {order_type}")

            return {
                "order_id": order["id"],
                "client_order_id": order.get("clientOrderId")
                or order.get("clientOrderId".lower())
                or order.get("client_order_id")
                or (order.get("info") or {}).get("clientOrderId")
                or client_order_id,
                "symbol": order["symbol"],
                "side": order["side"],
                "type": order["type"],
                "status": order["status"],
                "quantity": order["amount"],
                "price": order.get("price"),
                "filled": order.get("filled", 0),
                "remaining": order.get("remaining", 0),
                # ccxt Unified: cumulative average fill price once available
                "average_price": order.get("average"),
                "timestamp": order.get("timestamp"),
                "info": order.get("info", {}),
            }
        except Exception as e:
            logger.error(f"下单失败: {e}")
            raise

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        撤单

        Args:
            order_id: 订单ID
            symbol: 交易对符号

        Returns:
            是否成功
        """
        try:
            ccxt_sym = _ccxt_linear_usdt_perp_symbol(symbol) or symbol
            result = self.exchange.cancel_order(order_id, ccxt_sym)
            return result.get("status") == "canceled"
        except Exception as e:
            if _is_cancel_target_gone(e):
                logger.info("撤单: 订单已不存在 (%s)", e)
            else:
                logger.error(f"撤单失败: {e}")
            raise

    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        撤销所有订单

        Args:
            symbol: 交易对符号，None表示撤销所有交易对的订单

        Returns:
            撤销的订单列表
        """
        try:
            if symbol:
                ccxt_sym = _ccxt_linear_usdt_perp_symbol(symbol) or symbol
                result = self.exchange.cancel_all_orders(ccxt_sym)
            else:
                # 需要获取所有交易对的订单
                open_orders = self.get_open_orders()
                result = []
                for order in open_orders:
                    try:
                        canceled = self.cancel_order(order["order_id"], order["symbol"])
                        if canceled:
                            result.append(order)
                    except Exception as e:
                        logger.warning(
                            f"撤销订单 {order.get('order_id', 'unknown')} 失败: {e}"
                        )
            return result
        except Exception as e:
            logger.error(f"撤销所有订单失败: {e}")
            raise

    def _normalize_fetched_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        info = order.get("info", {}) or {}
        return {
            "order_id": order["id"],
            "client_order_id": order.get("clientOrderId")
            or info.get("clientOrderId"),
            "symbol": order["symbol"],
            "side": order["side"],
            "type": order["type"],
            "status": order["status"],
            "quantity": order.get("amount"),
            "price": order.get("price"),
            "filled": order.get("filled", 0),
            "remaining": order.get("remaining", 0),
            "average_price": order.get("average"),
            "created_at": self._normalize_timestamp(order.get("timestamp")),
            "timestamp": self._normalize_timestamp(order.get("timestamp")),
            "update_time": self._normalize_timestamp(info.get("updateTime")),
            "reject_reason": info.get("rejectReason") or info.get("r"),
            "error_message": info.get("msg"),
            "info": info,
        }

    def get_order(
        self,
        order_id: str,
        symbol: str,
        *,
        client_order_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        查询订单

        Args:
            order_id: 订单ID
            symbol: 交易对符号
            client_order_id: 可选；exchange id 查不到时用 origClientOrderId 再查一次

        Returns:
            订单信息
        """
        snap = self._fetch_order_by_exchange_id(order_id, symbol)
        if snap is not None:
            return snap
        cid = str(client_order_id or "").strip()
        if cid:
            return self.get_order_by_client_id(cid, symbol)
        return None

    def _fetch_order_by_exchange_id(
        self, order_id: str, symbol: str
    ) -> Optional[Dict[str, Any]]:
        try:
            ccxt_sym = _ccxt_linear_usdt_perp_symbol(symbol) or symbol
            order = self.exchange.fetch_order(order_id, ccxt_sym)
            return self._normalize_fetched_order(order)
        except Exception as e:
            err = str(e).lower()
            if "not found" in err or "-2013" in str(e):
                return None
            logger.error(f"查询订单失败: {e}")
            raise

    def get_algo_order_by_client_id(
        self, client_order_id: str, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Query open/historical algo (conditional) order by ``clientAlgoId``."""
        cid = str(client_order_id or "").strip()
        mid = self._futures_market_id(symbol)
        if not cid or not mid:
            return None
        try:
            raw = self._fapi_signed_get(
                "/fapi/v1/algoOrder",
                {"symbol": mid, "clientAlgoId": cid},
            )
        except Exception as e:
            if _is_order_lookup_miss(e):
                return None
            logger.error("查询 algo 订单失败 client_order_id=%s: %s", cid, e)
            raise
        if not isinstance(raw, dict) or not raw:
            return None
        return self._open_algo_order_from_binance_rest(raw)

    def get_order_by_client_id(
        self, client_order_id: str, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Query futures order by origClientOrderId (regular + algo conditional)."""
        cid = str(client_order_id or "").strip()
        if not cid:
            return None
        try:
            ccxt_sym = _ccxt_linear_usdt_perp_symbol(symbol) or symbol
            market = self.exchange.market(ccxt_sym)
            binance_sym = market.get("id") or str(symbol).replace("/", "")
            raw = self.exchange.fapiPrivateGetOrder(
                {"symbol": binance_sym, "origClientOrderId": cid}
            )
            if raw:
                order = self.exchange.parse_order(raw, market)
                return self._normalize_fetched_order(order)
        except Exception as e:
            err = str(e).lower()
            if "not found" not in err and "-2013" not in str(e):
                logger.error("查询订单失败 client_order_id=%s: %s", cid, e)
                raise
        return self.get_algo_order_by_client_id(cid, symbol)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取未完成订单

        Args:
            symbol: 交易对符号，None表示获取所有交易对的订单

        Returns:
            订单列表
        """
        try:
            ccxt_sym = _ccxt_linear_usdt_perp_symbol(symbol)
            # testnet + ccxt：部分私有 fapi 响应在库内变成空字符串，parse_orders 会崩
            if self.testnet:
                return self._get_open_orders_fapi_rest(symbol)
            orders = _call_binance_read(
                "fetch_open_orders",
                lambda: self.exchange.fetch_open_orders(ccxt_sym),
            )
            result = []
            for order in orders:
                info = order.get("info") or {}
                result.append(
                    {
                        "order_id": order["id"],
                        "client_order_id": order.get("clientOrderId")
                        or info.get("clientOrderId"),
                        "symbol": order["symbol"],
                        "side": order["side"],
                        "type": order["type"],
                        "status": order["status"],
                        "quantity": order["amount"],
                        "price": order.get("price"),
                        "filled": order.get("filled", 0),
                        "remaining": order.get("remaining", 0),
                        "position_side": info.get("positionSide"),
                        "reduce_only": str(info.get("reduceOnly") or "").lower()
                        == "true",
                        "created_at": self._normalize_timestamp(order.get("timestamp")),
                        "info": info,
                    }
                )
            return result
        except Exception as e:
            logger.error(f"获取未完成订单失败: {e}")
            raise

    def _convert_order_type(self, order_type: OrderType) -> str:
        """转换订单类型为ccxt格式"""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_MARKET: "stop_market",
            OrderType.TAKE_PROFIT: "take_profit",
            OrderType.TAKE_PROFIT_MARKET: "take_profit_market",
        }
        return mapping.get(order_type, "market")

    # ========== 交易对信息 ==========

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取交易对信息

        Args:
            symbol: 交易对符号

        Returns:
            交易对信息
        """
        try:
            markets = self.exchange.load_markets()
            ccxt_sym = _ccxt_linear_usdt_perp_symbol(symbol)
            market_key = None
            for key in (symbol, ccxt_sym):
                if key and key in markets:
                    market_key = key
                    break
            if market_key:
                market = markets[market_key]
                tick_size = None
                step_size = None
                filters = market.get("info", {}).get("filters", [])
                for f in filters:
                    if f.get("filterType") == "PRICE_FILTER":
                        tick_size = f.get("tickSize")
                    elif f.get("filterType") in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                        step_size = f.get("stepSize")
                return {
                    "symbol": market_key,
                    "base": market["base"],
                    "quote": market["quote"],
                    "precision": {
                        "amount": market["precision"]["amount"],
                        "price": market["precision"]["price"],
                    },
                    "limits": {
                        "amount": market["limits"]["amount"],
                        "price": market["limits"]["price"],
                        "cost": market["limits"]["cost"],
                    },
                    "filters": {
                        "tick_size": float(tick_size) if tick_size else None,
                        "step_size": float(step_size) if step_size else None,
                    },
                    "active": market.get("active", True),
                    "contract": market.get("contract", False),
                    "info": market.get("info", {}),
                }
            return None
        except Exception as e:
            logger.error(f"获取交易对信息失败: {e}")
            return None

    def get_leverage(self, symbol: str) -> Optional[int]:
        """
        获取杠杆倍数

        Args:
            symbol: 交易对符号

        Returns:
            杠杆倍数
        """
        try:
            # ccxt可能不支持直接获取杠杆，需要从仓位信息中获取
            position = self.get_position(symbol)
            if position:
                return position.get("leverage", 1)
            return None
        except Exception as e:
            logger.error(f"获取杠杆倍数失败: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        设置杠杆倍数

        Args:
            symbol: 交易对符号
            leverage: 杠杆倍数

        Returns:
            是否成功
        """
        try:
            ccxt_sym = _ccxt_linear_usdt_perp_symbol(symbol) or symbol
            self.exchange.set_leverage(leverage, ccxt_sym)
            return True
        except Exception as e:
            logger.error(f"设置杠杆倍数失败: {e}")
            return False

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """
        获取当前市场价格

        Args:
            symbol: 交易对符号

        Returns:
            当前价格（last price），如果获取失败返回None
        """
        try:
            import requests

            # 直接调用Binance API获取价格（避免ccxt的load_markets问题）
            if self.testnet:
                url = f"https://testnet.binancefuture.com/fapi/v1/ticker/price?symbol={symbol}"
            else:
                url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return float(data.get("price", 0))
            else:
                logger.error(f"API请求失败: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            logger.error(f"获取市场价格失败: {e}")
            return None
