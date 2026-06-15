#!/usr/bin/env python3
"""Quick Telegram bot connectivity test (getMe + optional sendMessage)."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.monitoring.telegram import send_telegram_message


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Test Telegram bot token and send a ping")
    p.add_argument(
        "--token",
        default="",
        help="Bot token (else env GRAFANA_ALERT_TELEGRAM_BOT_TOKEN)",
    )
    p.add_argument(
        "--chat-id",
        default="",
        help="Chat/channel id (else env GRAFANA_ALERT_TELEGRAM_CHAT_ID)",
    )
    p.add_argument("--message", default="mlbot Telegram test OK")
    p.add_argument("--send", action="store_true", help="Send test message after getMe")
    args = p.parse_args()

    import os

    token = (args.token or os.getenv("GRAFANA_ALERT_TELEGRAM_BOT_TOKEN", "")).strip()
    if not token:
        print(
            "error: pass --token or set GRAFANA_ALERT_TELEGRAM_BOT_TOKEN",
            file=sys.stderr,
        )
        return 1

    me = _get_json(f"https://api.telegram.org/bot{token}/getMe")
    print(json.dumps(me, indent=2, ensure_ascii=False))
    if not me.get("ok"):
        return 1

    updates = _get_json(f"https://api.telegram.org/bot{token}/getUpdates")
    if updates.get("result"):
        print("\nrecent chats from getUpdates:")
        for row in updates["result"][-5:]:
            msg = row.get("message") or row.get("channel_post") or {}
            chat = msg.get("chat") or {}
            print(
                f"  chat_id={chat.get('id')} title={chat.get('title') or chat.get('username')}"
            )

    if args.send:
        chat = (args.chat_id or os.getenv("GRAFANA_ALERT_TELEGRAM_CHAT_ID", "")).strip()
        if not chat:
            print(
                "error: pass --chat-id or set GRAFANA_ALERT_TELEGRAM_CHAT_ID",
                file=sys.stderr,
            )
            return 1
        ok = send_telegram_message(
            str(args.message),
            token=token,
            chat_id=chat,
            skip_cooldown=True,
        )
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
