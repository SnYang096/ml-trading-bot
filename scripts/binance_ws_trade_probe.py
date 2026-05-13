"""Remote probe: Binance USDM + spot @trade streams (no aggTrade subscription)."""

import asyncio
import json

import websockets


async def probe(label: str, url: str, wait: float = 30) -> None:
    print(f"=== {label} ===")
    print("URL:", url)
    try:
        async with websockets.connect(url, ping_interval=15, open_timeout=10) as ws:
            try:
                msg = await asyncio.wait_for(ws.recv(), wait)
                print("MSG:", msg[:180])
                payload = json.loads(msg)
                data = payload.get("data", payload)
                if isinstance(data, dict):
                    print("EVENT:", data.get("e"), "SYMBOL:", data.get("s"))
            except asyncio.TimeoutError:
                print(f"NO MSG within {wait}s, close_code=", ws.close_code)
    except Exception as e:
        print("ERROR:", type(e).__name__, e)
    print()


async def main() -> None:
    await probe("USDM trade", "wss://fstream.binance.com/ws/btcusdt@trade")
    await probe("Spot trade", "wss://stream.binance.com:9443/ws/btcusdt@trade")


if __name__ == "__main__":
    asyncio.run(main())
