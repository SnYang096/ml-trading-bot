"""Constitution kill-switch for multi-leg live (C layer).

Tracks peak / period-start equity from exchange balance snapshots and blocks
new risk-increasing actions when daily/weekly/monthly loss or max drawdown
limits are breached. Aligns with ``evaluate_safety_state`` (B layer) and
``backtest_multileg_timeline`` halt semantics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.live_data_stream.constitution_config import (
    load_constitution_dict,
    multi_leg_section,
    resolve_multileg_sim_limits,
)
from src.time_series_model.core.constitution.constitution_executor import (
    load_constitution_config,
)
from src.time_series_model.core.constitution.safety_runtime import (
    SafetyRuntimeState,
    evaluate_safety_state,
)

logger = logging.getLogger(__name__)

_RISK_INCREASING_ACTIONS = frozenset({"place", "place_protection"})


def _loss_fraction(anchor: float, equity: float) -> float:
    if anchor <= 0:
        return 0.0
    return max(0.0, (float(anchor) - float(equity)) / float(anchor))


def _iso_week_key(now: datetime) -> str:
    dt = now.astimezone(timezone.utc)
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _month_key(now: datetime) -> str:
    dt = now.astimezone(timezone.utc)
    return f"{dt.year}-{dt.month:02d}"


@dataclass
class MultiLegKillSwitchConfig:
    enabled: bool = True
    daily_loss_limit: float = 0.06
    weekly_loss_limit: float = 0.08
    monthly_loss_limit: float = 0.12
    max_dd: float = 0.20
    max_turnover_mean: float = 0.35
    max_cost_mean: float = 0.002
    cooldown_minutes: int = 720
    daily_reset_timezone: str = "UTC"
    kill_on_any_hard_violation: bool = True

    @classmethod
    def from_constitution_yaml(cls, constitution_yaml: str | Path) -> MultiLegKillSwitchConfig:
        raw = load_constitution_dict(constitution_yaml)
        cfg = load_constitution_config(constitution_yaml)
        sim = resolve_multileg_sim_limits(
            {
                "kill_switch": raw.get("kill_switch") or {},
                "multi_leg": multi_leg_section(raw),
            }
        )
        max_dd = sim.get("max_drawdown_pct")
        if max_dd is None:
            max_dd = cfg.max_dd
        return cls(
            enabled=bool(cfg.kill_enabled),
            daily_loss_limit=float(cfg.daily_loss_limit),
            weekly_loss_limit=float(cfg.weekly_loss_limit),
            monthly_loss_limit=float(cfg.monthly_loss_limit),
            max_dd=float(max_dd),
            max_turnover_mean=float(cfg.max_turnover_mean),
            max_cost_mean=float(cfg.max_cost_mean),
            cooldown_minutes=int(cfg.cooldown_minutes),
            daily_reset_timezone=str(cfg.daily_reset_timezone),
            kill_on_any_hard_violation=bool(cfg.kill_on_any_hard_violation),
        )


@dataclass
class MultiLegKillSwitchTracker:
    """Shared account-level halt tracker for all multi-leg orchestrators."""

    config: MultiLegKillSwitchConfig
    state_path: Path
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    week_start_equity: float = 0.0
    month_start_equity: float = 0.0
    current_day: str = ""
    current_week: str = ""
    current_month: str = ""
    last_equity: float = 0.0
    safety: SafetyRuntimeState = field(default_factory=SafetyRuntimeState)
    drawdown_override: Optional[float] = None
    on_halt_change: Optional[Any] = None   # callback(was_halted: bool, is_halted: bool, reasons: list[str])
    _updated_this_batch: bool = field(default=False, repr=False)

    def load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "multi-leg kill-switch: failed to load %s", self.state_path, exc_info=True
            )
            return
        self.peak_equity = float(raw.get("peak_equity") or 0.0)
        self.day_start_equity = float(raw.get("day_start_equity") or 0.0)
        self.week_start_equity = float(raw.get("week_start_equity") or 0.0)
        self.month_start_equity = float(raw.get("month_start_equity") or 0.0)
        self.current_day = str(raw.get("current_day") or "")
        self.current_week = str(raw.get("current_week") or "")
        self.current_month = str(raw.get("current_month") or "")
        self.last_equity = float(raw.get("last_equity") or 0.0)
        self.safety = SafetyRuntimeState.from_dict(raw.get("safety"))

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "peak_equity": self.peak_equity,
            "day_start_equity": self.day_start_equity,
            "week_start_equity": self.week_start_equity,
            "month_start_equity": self.month_start_equity,
            "current_day": self.current_day,
            "current_week": self.current_week,
            "current_month": self.current_month,
            "last_equity": self.last_equity,
            "safety": self.safety.as_dict(),
        }
        self.state_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def begin_batch(self) -> None:
        self._updated_this_batch = False

    def update_from_equity(
        self, equity: float, *, now: Optional[datetime] = None
    ) -> None:
        if self._updated_this_batch:
            return
        self._updated_this_batch = True
        old_halted = self.safety.halted
        if not self.config.enabled:
            return

        now = now or datetime.now(timezone.utc)
        eq = float(equity)
        if eq <= 0:
            return

        day_key = now.astimezone(timezone.utc).date().isoformat()
        week_key = _iso_week_key(now)
        month_key = _month_key(now)

        if self.peak_equity <= 0:
            self.peak_equity = eq
            self.day_start_equity = eq
            self.week_start_equity = eq
            self.month_start_equity = eq
            self.current_day = day_key
            self.current_week = week_key
            self.current_month = month_key
        else:
            if day_key != self.current_day:
                self.day_start_equity = eq
                self.current_day = day_key
            if week_key != self.current_week:
                self.week_start_equity = eq
                self.current_week = week_key
            if month_key != self.current_month:
                self.month_start_equity = eq
                self.current_month = month_key
            self.peak_equity = max(self.peak_equity, eq)

        self.last_equity = eq
        drawdown = _loss_fraction(self.peak_equity, eq)
        daily_loss = _loss_fraction(self.day_start_equity, eq)
        weekly_loss = _loss_fraction(self.week_start_equity, eq)
        monthly_loss = _loss_fraction(self.month_start_equity, eq)

        if self.drawdown_override is not None:
            drawdown = float(self.drawdown_override)

        limits = {
            "daily_loss_limit": float(self.config.daily_loss_limit),
            "weekly_loss_limit": float(self.config.weekly_loss_limit),
            "monthly_loss_limit": float(self.config.monthly_loss_limit),
            "max_dd": float(self.config.max_dd),
            "max_turnover_mean": float(self.config.max_turnover_mean),
            "max_cost_mean": float(self.config.max_cost_mean),
        }
        decision = evaluate_safety_state(
            state=self.safety,
            now=now,
            cooldown_minutes=int(self.config.cooldown_minutes),
            daily_reset_tz=str(self.config.daily_reset_timezone),
            daily_loss=daily_loss,
            weekly_loss=weekly_loss,
            monthly_loss=monthly_loss,
            drawdown=drawdown,
            hard_violation=False,
            data_bad=False,
            daily_cost_mean=None,
            daily_turnover_mean=None,
            limits=limits,
        )
        self.safety = decision.state
        # ── halt-state transition callback (e.g. Telegram) ──
        is_halted = bool(self.safety.halted)
        if old_halted != is_halted and self.on_halt_change is not None:
            try:
                self.on_halt_change(old_halted, is_halted, list(decision.reasons or []))
            except Exception:
                logger.warning(
                    "multi-leg kill-switch: on_halt_change callback failed",
                    exc_info=True,
                )
        if not decision.ok:
            logger.info(
                "multi-leg kill-switch halt: reasons=%s equity=%.2f drawdown=%.4f "
                "daily_loss=%.4f",
                decision.reasons,
                eq,
                drawdown,
                daily_loss,
            )
        self.save()

    @property
    def drawdown_pct(self) -> Optional[float]:
        if self.peak_equity <= 0 or self.last_equity <= 0:
            return None
        if self.drawdown_override is not None:
            return float(self.drawdown_override)
        return _loss_fraction(self.peak_equity, self.last_equity)

    def is_halted(self) -> bool:
        if not self.config.enabled:
            return False
        return bool(self.safety.halted)

    def halt_reasons(self) -> list[str]:
        return list(self.safety.halt_reason or [])

    def blocks_action(self, kind: str) -> Optional[str]:
        """Return rejection reason when action must be blocked, else None."""
        if not self.config.enabled:
            return None
        action_kind = str(kind or "").lower()
        if action_kind not in _RISK_INCREASING_ACTIONS:
            return None
        if not self.is_halted():
            return None
        reasons = self.halt_reasons()
        detail = ",".join(reasons) if reasons else "halted"
        return f"kill_switch:{detail}"
