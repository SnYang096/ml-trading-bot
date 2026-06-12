"""Shared symbol filter helpers for business console services."""

from __future__ import annotations

from typing import Optional

ALL_SYMBOLS = frozenset({"", "*", "ALL", "__ALL__"})


def is_all_symbols(symbol: Optional[str]) -> bool:
    """True when the UI/API wildcard means no per-symbol filter."""
    return str(symbol or "").strip().upper() in ALL_SYMBOLS
