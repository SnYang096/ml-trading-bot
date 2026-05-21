"""Path and tuning for the read-only business console."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


def _repo_root() -> Path:
    """Best-effort repo root; production paths come from MLBOT_CONSOLE_* env."""
    p = Path(__file__).resolve()
    if len(p.parents) > 4:
        cand = p.parents[4]
        if (cand / "src").is_dir() or (cand / "live").is_dir():
            return cand
    if len(p.parents) > 2:
        return p.parents[2]
    return p.parent


@dataclass(frozen=True)
class ConsoleSettings:
    repo_root: Path
    feature_bus_root: Path
    live_data_root: Path
    engine_data_root: Path
    live_root: Path
    constitution_yaml: Path
    universe_yaml: Path
    trend_order_db: Path
    live_monitor_db: Path
    spot_order_db: Path
    spot_ledger_db: Path
    multi_leg_db: Path
    max_ohlcv_days: int
    live_storage_bars_root: Path
    stitch_live_storage: bool
    macro_spot_kline_root: Path
    daily_ohlcv_start: date
    max_daily_ohlcv_days: int
    map_poll_seconds: float
    grafana_url: str
    rolling_backtest_url: str
    basic_auth_user: str | None
    basic_auth_password: str | None

    @classmethod
    def from_env(cls) -> "ConsoleSettings":
        root = _repo_root()
        live_root = Path(
            os.getenv("MLBOT_CONSOLE_LIVE_ROOT", str(root / "live" / "highcap"))
        )
        live_data = Path(
            os.getenv("MLBOT_CONSOLE_LIVE_DATA_ROOT", str(live_root / "data"))
        )
        engine_data = Path(
            os.getenv("MLBOT_CONSOLE_ENGINE_DATA_ROOT", str(root / "data"))
        )
        const = os.getenv(
            "MLBOT_CONSTITUTION_YAML",
            str(live_root / "config" / "constitution" / "constitution.yaml"),
        )
        return cls(
            repo_root=root,
            feature_bus_root=Path(
                os.getenv(
                    "MLBOT_CONSOLE_FEATURE_BUS_ROOT",
                    str(root / "live" / "shared_feature_bus"),
                )
            ),
            live_data_root=live_data,
            engine_data_root=engine_data,
            live_root=live_root,
            constitution_yaml=Path(const),
            universe_yaml=Path(
                os.getenv(
                    "MLBOT_CONSOLE_UNIVERSE_YAML",
                    str(live_root / "universe.yaml"),
                )
            ),
            trend_order_db=Path(
                os.getenv(
                    "MLBOT_ORDER_MANAGEMENT_DB_PATH",
                    str(engine_data / "order_management.db"),
                )
            ),
            live_monitor_db=Path(
                os.getenv(
                    "MLBOT_CONSOLE_LIVE_MONITOR_DB",
                    str(live_data / "db" / "live_monitor.db"),
                )
            ),
            spot_order_db=Path(
                os.getenv(
                    "MLBOT_SPOT_DB_PATH",
                    str(live_data / "spot_order_management.db"),
                )
            ),
            spot_ledger_db=Path(
                os.getenv(
                    "MLBOT_SPOT_LEDGER_DB_PATH",
                    str(live_data / "spot_accum_ledger.db"),
                )
            ),
            multi_leg_db=Path(
                os.getenv(
                    "MLBOT_MULTI_LEG_DB_PATH",
                    str(engine_data / "multi_leg_order_management.db"),
                )
            ),
            max_ohlcv_days=int(os.getenv("MLBOT_CONSOLE_MAX_OHLCV_DAYS", "180")),
            live_storage_bars_root=Path(
                os.getenv(
                    "MLBOT_CONSOLE_LIVE_STORAGE_BARS_ROOT",
                    str(live_data / "bars"),
                )
            ),
            stitch_live_storage=os.getenv(
                "MLBOT_CONSOLE_STITCH_LIVE_STORAGE", "1"
            ).strip().lower()
            not in ("0", "false", "off", "no"),
            macro_spot_kline_root=Path(
                os.getenv(
                    "MLBOT_CONSOLE_MACRO_SPOT_KLINE_ROOT",
                    str(live_data / "macro" / "spot_klines"),
                )
            ),
            daily_ohlcv_start=date.fromisoformat(
                os.getenv("MLBOT_CONSOLE_DAILY_OHLCV_START", "2017-01-01")
            ),
            max_daily_ohlcv_days=int(
                os.getenv("MLBOT_CONSOLE_MAX_DAILY_OHLCV_DAYS", "3650")
            ),
            map_poll_seconds=float(
                os.getenv("MLBOT_CONSOLE_MAP_POLL_SECONDS", "10")
            ),
            grafana_url=os.getenv(
                "MLBOT_CONSOLE_GRAFANA_URL", "http://127.0.0.1:3000"
            ),
            rolling_backtest_url=os.getenv("MLBOT_CONSOLE_ROLLING_BACKTEST_URL", ""),
            basic_auth_user=os.getenv("MLBOT_CONSOLE_BASIC_AUTH_USER") or None,
            basic_auth_password=os.getenv("MLBOT_CONSOLE_BASIC_AUTH_PASSWORD") or None,
        )


SETTINGS = ConsoleSettings.from_env()
