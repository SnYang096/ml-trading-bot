"""Symbol-owner helpers shared by live daemon and timeline backtest."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

Action = Dict[str, Any]


def runtime_holds_symbol_engine(engine: Any) -> bool:
    """True when engine occupies the symbol slot (not just transient inventory)."""
    holds = getattr(engine, "holds_real_grid_slot", None)
    if callable(holds):
        try:
            if bool(holds()):
                return True
        except Exception:
            pass
    try:
        if bool(list(engine.local_position_snapshots())):
            return True
    except Exception:
        pass
    return False


def refresh_symbol_owner(
    runtimes: Iterable[Any],
    symbol_owner: Dict[str, str],
    sym: str,
) -> None:
    """Rebuild owner from engine slots; chop runtimes must precede trend in *runtimes*."""
    sym_u = str(sym or "").upper().strip()
    owner = ""
    for rt in runtimes:
        rt_sym = str(getattr(rt, "symbol", "") or "").upper().strip()
        if rt_sym != sym_u:
            continue
        if runtime_holds_symbol_engine(getattr(rt, "engine", rt)):
            owner = str(getattr(rt, "name", "") or "")
            break
    if owner:
        symbol_owner[sym_u] = owner
    else:
        symbol_owner.pop(sym_u, None)


def filter_places_for_owner(
    actions: Iterable[Action],
    *,
    owner: str,
    runtime_name: str,
) -> Tuple[List[Action], int]:
    """Drop ``place`` actions when another strategy owns the symbol."""
    if not owner or owner == str(runtime_name or ""):
        return [dict(a) for a in actions], 0
    kept: List[Action] = []
    dropped = 0
    for action in actions:
        kind = str((action or {}).get("action", "") or "").lower()
        if kind == "place":
            dropped += 1
            continue
        kept.append(dict(action))
    return kept, dropped
