"""Cadence timer calendar → next run / valid-until dates for CMS cards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.monitoring.staleness import parse_run_ts

# Mirror etc/systemd/mlbot-monitor-*.timer (UTC). Override via schedules.yaml `timers`.
DEFAULT_TIMERS: Dict[str, str] = {
    "daily": "*-*-* 06:30:00",
    "weekly": "Sun *-*-* 08:00:00",
    "weekly_c": "Sun *-*-* 08:30:00",
    "monthly": "*-*-01 09:00:00",
    "monthly_c": "*-*-01 09:30:00",
    "quarterly": "*-01,04,07,10-01 10:00:00",
    "yearly": "*-01-01 11:00:00",
}

_WEEKDAY = {
    "Mon": 0,
    "Tue": 1,
    "Wed": 2,
    "Thu": 3,
    "Fri": 4,
    "Sat": 5,
    "Sun": 6,
}


def resolve_timer_calendar(
    cadence: str,
    schedules_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    cfg = schedules_cfg or {}
    timers = cfg.get("timers") or {}
    if isinstance(timers, dict):
        raw = timers.get(cadence)
        if raw:
            if isinstance(raw, dict):
                cal = raw.get("on_calendar")
                if cal:
                    return str(cal).strip()
            else:
                return str(raw).strip()
    return DEFAULT_TIMERS.get(cadence)


def valid_until_at(
    run_ts: Optional[str],
    *,
    max_age_hours: float,
) -> Optional[str]:
    """ISO timestamp when the latest run stops counting as fresh."""
    dt = parse_run_ts(str(run_ts or ""))
    if dt is None:
        return None
    until = dt + timedelta(hours=float(max_age_hours))
    return until.astimezone(timezone.utc).isoformat()


def last_run_at_iso(run_ts: Optional[str]) -> Optional[str]:
    dt = parse_run_ts(str(run_ts or ""))
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _parse_time(time_spec: str) -> tuple[int, int, int]:
    parts = time_spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid time spec: {time_spec!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _at(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def next_on_calendar(
    on_calendar: str,
    *,
    after: Optional[datetime] = None,
) -> datetime:
    """Next systemd OnCalendar match strictly after ``after`` (UTC)."""
    after = after or datetime.now(timezone.utc)
    if after.tzinfo is None:
        after = after.replace(tzinfo=timezone.utc)
    else:
        after = after.astimezone(timezone.utc)

    tokens = on_calendar.strip().split()
    if len(tokens) < 2:
        raise ValueError(f"unsupported on_calendar: {on_calendar!r}")

    time_spec = tokens[-1]
    hour, minute, second = _parse_time(time_spec)
    date_tokens = tokens[:-1]

    # Weekly: Sun *-*-*
    if len(date_tokens) == 2 and date_tokens[0] in _WEEKDAY:
        target_dow = _WEEKDAY[date_tokens[0]]
        days_ahead = (target_dow - after.weekday()) % 7
        cand = _at(after.year, after.month, after.day, hour, minute, second) + timedelta(
            days=days_ahead
        )
        if cand <= after:
            cand += timedelta(days=7)
        return cand

    date_spec = " ".join(date_tokens)

    # Daily: *-*-*
    if date_spec == "*-*-*":
        cand = _at(after.year, after.month, after.day, hour, minute, second)
        if cand <= after:
            cand += timedelta(days=1)
        return cand

    # Monthly: *-*-01
    if date_spec == "*-*-01":
        cand = _at(after.year, after.month, 1, hour, minute, second)
        if cand <= after:
            if after.month == 12:
                cand = _at(after.year + 1, 1, 1, hour, minute, second)
            else:
                cand = _at(after.year, after.month + 1, 1, hour, minute, second)
        return cand

    # Yearly: *-01-01
    if date_spec == "*-01-01":
        cand = _at(after.year, 1, 1, hour, minute, second)
        if cand <= after:
            cand = _at(after.year + 1, 1, 1, hour, minute, second)
        return cand

    # Quarterly: *-01,04,07,10-01
    if date_spec.startswith("*-") and "," in date_spec and date_spec.endswith("-01"):
        months_part = date_spec[2:].rsplit("-", 1)[0]
        months = [int(m) for m in months_part.split(",") if m.strip()]
        best: Optional[datetime] = None
        for y in (after.year, after.year + 1):
            for m in months:
                cand = _at(y, m, 1, hour, minute, second)
                if cand > after and (best is None or cand < best):
                    best = cand
        if best is not None:
            return best

    raise ValueError(f"unsupported on_calendar: {on_calendar!r}")


def attach_schedule_dates(
    card: Dict[str, Any],
    *,
    schedules_cfg: Dict[str, Any],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Add last_run_at, valid_until_at, next_run_at to a cadence card."""
    now = now or datetime.now(timezone.utc)
    cadence = str(card.get("cadence") or "")
    max_age = float(card.get("max_age_hours") or 0)
    run_ts = card.get("run_ts")

    card["last_run_at"] = last_run_at_iso(run_ts)
    card["valid_until_at"] = (
        valid_until_at(run_ts, max_age_hours=max_age) if run_ts else None
    )

    cal = resolve_timer_calendar(cadence, schedules_cfg)
    if cal:
        try:
            card["next_run_at"] = next_on_calendar(cal, after=now).isoformat()
            card["timer_calendar"] = cal
        except ValueError:
            card["next_run_at"] = None
    else:
        card["next_run_at"] = None

    return card
