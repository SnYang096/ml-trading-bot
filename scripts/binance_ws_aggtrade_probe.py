"""Probe Binance USD-M futures aggTrade via python-binance.

This intentionally uses ``ThreadedWebsocketManager.start_aggtrade_futures_socket``,
matching the live listener. It does not open raw ``@trade`` or hand-rolled
websocket subscriptions.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict

from binance import ThreadedWebsocketManager


def _payload(message: Dict[str, Any]) -> Dict[str, Any]:
    payload = message.get("data", message)
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--max-messages", type=int, default=5)
    args = p.parse_args()

    received = 0

    def handle(message: Dict[str, Any]) -> None:
        nonlocal received
        payload = _payload(message)
        if payload.get("e") != "aggTrade":
            print("non-aggTrade:", str(message)[:180])
            return
        received += 1
        if received <= args.max_messages:
            print(
                "aggTrade",
                payload.get("s"),
                "p=",
                payload.get("p"),
                "q=",
                payload.get("q"),
                "T=",
                payload.get("T"),
            )
            sys.stdout.flush()

    twm = ThreadedWebsocketManager()
    twm.start()
    twm.start_aggtrade_futures_socket(callback=handle, symbol=args.symbol.upper())

    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline and received < args.max_messages:
            time.sleep(0.1)
    finally:
        twm.stop()

    if received <= 0:
        raise SystemExit(f"FAIL: no aggTrade message in {args.seconds}s")
    print(f"OK: received {received} aggTrade messages")


if __name__ == "__main__":
    main()
