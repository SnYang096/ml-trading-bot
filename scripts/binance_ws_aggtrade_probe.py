"""Probe Binance USDⓈ-M Futures @aggTrade WebSocket (and optional vs @trade bandwidth).

Docs: Aggregate Trade Streams push fills aggregated every 100ms (same price & taker side).
URL: wss://fstream.binance.com/ws/<symbol>@aggTrade
Combined: wss://fstream.binance.com/stream?streams=btcusdt@aggTrade/ethusdt@aggTrade
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets", file=sys.stderr)
    raise


def _unwrap_payload(raw: str) -> Dict[str, Any]:
    outer = json.loads(raw)
    inner = outer.get("data", outer)
    if not isinstance(inner, dict):
        raise ValueError(f"unexpected payload shape: {type(inner)}")
    return inner


async def recv_first_aggtrade(url: str, wait: float) -> None:
    print(f"=== First message ({url}) ===")
    try:
        async with websockets.connect(
            url, ping_interval=15, open_timeout=30, close_timeout=5
        ) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=wait)
        print("RAW:", raw[:240] + ("..." if len(raw) > 240 else ""))
        try:
            data = _unwrap_payload(raw)
            print(
                "EVENT:",
                data.get("e"),
                "SYMBOL:",
                data.get("s"),
                "p:",
                data.get("p"),
                "q:",
                data.get("q"),
                "m:",
                data.get("m"),
            )
            if data.get("e") != "aggTrade":
                print("WARN: expected event type aggTrade")
        except Exception as e:
            print("PARSE ERROR:", type(e).__name__, e)
    except TimeoutError:
        print(
            "TIMEOUT: cannot open WebSocket or no first message — check network/firewall "
            "or run on a host that can reach fstream.binance.com"
        )
        raise


async def bandwidth_sample(url: str, label: str, seconds: float) -> None:
    """Rough inbound bytes/sec over raw WS frames (UTF-8 length if str)."""
    print(f"=== Bandwidth ~{seconds}s ({label}) ===")
    print("URL:", url)
    total_bytes = 0
    n = 0
    t0 = time.monotonic()
    async with websockets.connect(
        url, ping_interval=15, open_timeout=30, close_timeout=5
    ) as ws:
        while time.monotonic() - t0 < seconds:
            remaining = seconds - (time.monotonic() - t0)
            if remaining <= 0:
                break
            raw = await asyncio.wait_for(ws.recv(), timeout=min(60.0, remaining + 1.0))
            total_bytes += len(raw) if isinstance(raw, bytes) else len(raw.encode("utf-8"))
            n += 1
    elapsed = time.monotonic() - t0
    if elapsed <= 0:
        elapsed = seconds
    print(f"  messages={n} bytes={total_bytes} avg={total_bytes / elapsed:.0f} B/s")
    print()


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Uppercase futures symbol (default BTCUSDT)",
    )
    p.add_argument("--wait", type=float, default=30.0, help="Seconds to wait for first msg")
    p.add_argument(
        "--bandwidth",
        action="store_true",
        help="Also sample ~8s inbound bytes/sec for aggTrade vs trade (same symbol)",
    )
    p.add_argument(
        "--bw-seconds",
        type=float,
        default=8.0,
        help="Duration for --bandwidth samples",
    )
    args = p.parse_args()
    sym = args.symbol.strip().lower()

    single_agg = f"wss://fstream.binance.com/ws/{sym}@aggTrade"
    combined_agg = (
        "wss://fstream.binance.com/stream?streams="
        "btcusdt@aggTrade/ethusdt@aggTrade"
    )

    await recv_first_aggtrade(single_agg, args.wait)
    print()
    await recv_first_aggtrade(combined_agg, args.wait)

    if args.bandwidth:
        trade_url = f"wss://fstream.binance.com/ws/{sym}@trade"
        await bandwidth_sample(single_agg, f"USDM {sym}@aggTrade", args.bw_seconds)
        await bandwidth_sample(trade_url, f"USDM {sym}@trade", args.bw_seconds)
        print("Note: aggTrade emits fewer rows than raw trade; lower B/s is expected in quiet periods.")


if __name__ == "__main__":
    asyncio.run(main())
