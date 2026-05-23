"""Pytest fixtures for business console (src/mlbot_console)."""

from __future__ import annotations

import io
import sys
import zipfile
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def bus_root(tmp_path: Path) -> Path:
    """Feature bus root with synthetic bars_1min + features/120T parquet."""
    import pandas as pd

    sym = "ETHUSDT"
    bars_dir = tmp_path / "bars_1min"
    bars_dir.mkdir(parents=True)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for i in range(24 * 60 * 3):
        ts = start + pd.Timedelta(minutes=i)
        price = 100.0 + i * 0.01
        rows.append(
            {
                "timestamp": ts,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price + 0.1,
                "volume": 10.0 + i,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(bars_dir / f"{sym}.parquet", index=False)

    feat_dir = tmp_path / "features" / "120T"
    feat_dir.mkdir(parents=True)
    feat_rows = []
    for i in range(0, 24 * 60, 120):
        ts = start + pd.Timedelta(minutes=i)
        close = 100.0 + i * 0.02
        feat_rows.append(
            {
                "timestamp": ts,
                "close": close,
                "weekly_ema_200_position": -0.05 if i < 600 else 0.12,
                "ema_1200": close * 0.95,
                "ema_1200_position": 0.02,
                "regime_score": 0.1 + (i % 100) * 0.001,
            }
        )
    pd.DataFrame(feat_rows).to_parquet(feat_dir / f"{sym}.parquet", index=False)

    latest_dir = tmp_path / "latest" / "bars_1min"
    latest_dir.mkdir(parents=True)
    last_ts = rows[-1]["timestamp"]
    (latest_dir / f"{sym}.json").write_text(
        f'{{"timestamp": "{last_ts.isoformat()}", "kind": "bars_1min", "rows": {len(rows)}}}',
        encoding="utf-8",
    )

    from src.live_data_stream.spot_weekly_ema_seed import seed_parquet_path

    seed_dir = tmp_path / "macro" / "spot_weekly_ema200"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "week_ts": [start - pd.Timedelta(days=7), start],
            "weekly_ema_200": [94.0, 95.0],
        }
    ).to_parquet(seed_parquet_path(seed_dir, sym), index=False)

    return tmp_path


@pytest.fixture
def trend_db(tmp_path: Path) -> Path:
    import sqlite3

    path = tmp_path / "order_management.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE positions (
            position_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            entry_time TEXT,
            exit_time TEXT,
            entry_price REAL,
            exit_price REAL,
            realized_pnl REAL,
            status TEXT,
            strategy_id TEXT,
            stop_loss_price REAL,
            take_profit_price REAL,
            current_size REAL
        );
        CREATE TABLE position_operations (
            operation_id TEXT PRIMARY KEY,
            position_id TEXT,
            operation_type TEXT,
            operation_time TEXT,
            size REAL,
            price REAL,
            reason TEXT,
            stop_loss_price REAL,
            take_profit_price REAL
        );
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            symbol TEXT,
            side TEXT,
            status TEXT,
            order_type TEXT,
            quantity REAL,
            price REAL,
            stop_price REAL,
            filled_at TEXT,
            created_at TEXT,
            updated_at TEXT,
            average_price REAL,
            filled_quantity REAL,
            position_id TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO positions VALUES (
            'p1', 'ETHUSDT', 'long',
            '2024-01-01T10:00:00+00:00', '2024-01-01T14:00:00+00:00',
            100.0, 105.0, 12.5, 'closed', 'tpc', 98.5, 106.0, 2.5
        )
        """
    )
    conn.execute(
        """
        INSERT INTO orders VALUES (
            'ord_pending', 'ETHUSDT', 'BUY', 'pending', 'limit',
            0.1, 101.0, NULL,
            NULL, '2024-01-01T11:00:00+00:00', '2024-01-01T11:00:00+00:00',
            NULL, 0.0, NULL
        )
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def spot_db(tmp_path: Path) -> Path:
    import sqlite3

    path = tmp_path / "spot_order_management.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE spot_orders (
            order_id TEXT PRIMARY KEY,
            created_at TEXT,
            updated_at TEXT,
            symbol TEXT,
            side TEXT,
            order_type TEXT,
            quantity REAL,
            price REAL,
            status TEXT,
            filled_quantity REAL,
            filled_quote_usdt REAL
        );
        """
    )
    conn.execute(
        """
        INSERT INTO spot_orders VALUES (
            's1', '2024-01-02T08:00:00+00:00', '2024-01-02T08:05:00+00:00',
            'ETHUSDT', 'buy', 'market', 0.1, 2000.0, 'filled', 0.1, 200.0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO spot_orders VALUES (
            's_pending', '2024-01-02T09:00:00+00:00', '2024-01-02T09:00:00+00:00',
            'ETHUSDT', 'buy', 'limit', 0.05, 1900.0, 'pending', 0.0, 0.0
        )
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def multi_leg_db(tmp_path: Path) -> Path:
    from src.order_management.multi_leg_storage import MultiLegStorage

    path = tmp_path / "multi_leg_order_management.db"
    storage = MultiLegStorage(str(path))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["ETHUSDT"],
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "ml_eth_entry",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "purpose": "entry",
            "quantity": 0.01,
            "price": 2000.0,
            "status": "filled",
            "filled_quantity": 0.01,
            "average_price": 2000.0,
            "filled_at": "2024-01-01T12:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "ml_eth_l2_expired",
            "symbol": "ETHUSDT",
            "side": "BUY",
            "purpose": "entry",
            "quantity": 0.01,
            "price": 1990.0,
            "status": "expired",
            "filled_at": "2024-01-01T11:00:00+00:00",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": "ml_eth_open_tp",
            "symbol": "ETHUSDT",
            "side": "SELL",
            "purpose": "take_profit",
            "quantity": 0.0,
            "price": 2010.0,
            "status": "open",
            "filled_quantity": 0.0,
            "filled_at": "2024-01-01T13:00:00+00:00",
        }
    )
    storage.record_execution_report(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "symbol": "ETHUSDT",
            "order_id": "ex1",
            "status": "FILLED",
            "execution_type": "TRADE",
            "event_time": "2024-01-01T12:05:00+00:00",
        }
    )
    storage.finish_run(run_id)
    return path


@pytest.fixture
def spot_ledger_db(tmp_path: Path) -> Path:
    import sqlite3

    p = tmp_path / "spot_accum_ledger.db"
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE state_kv (
            k TEXT PRIMARY KEY,
            v TEXT
        );
        CREATE TABLE daily_counters (
            day_key TEXT,
            symbol TEXT,
            buy_entries INTEGER,
            deploy_usdt REAL,
            PRIMARY KEY (day_key, symbol)
        );
        """
    )
    conn.commit()
    conn.close()
    return p


@pytest.fixture
def console_settings(
    bus_root, trend_db, spot_db, spot_ledger_db, multi_leg_db, tmp_path
) -> "ConsoleSettings":
    from mlbot_console.config import ConsoleSettings

    universe = tmp_path / "universe.yaml"
    universe.write_text(
        "symbols:\n  ETHUSDT: {}\n  SOLUSDT: {}\n",
        encoding="utf-8",
    )
    return ConsoleSettings(
        repo_root=tmp_path,
        feature_bus_root=bus_root,
        live_data_root=tmp_path,
        engine_data_root=tmp_path,
        live_root=tmp_path,
        constitution_yaml=tmp_path / "constitution.yaml",
        universe_yaml=universe,
        trend_order_db=trend_db,
        live_monitor_db=tmp_path / "live_monitor.db",
        spot_order_db=spot_db,
        spot_ledger_db=spot_ledger_db,
        multi_leg_db=multi_leg_db,
        max_ohlcv_days=7,
        live_storage_bars_root=tmp_path / "bars",
        stitch_live_storage=True,
        macro_spot_kline_root=tmp_path / "macro" / "spot_klines",
        macro_weekly_ema_seed_root=tmp_path / "macro" / "spot_weekly_ema200",
        daily_ohlcv_start=date(2017, 1, 1),
        max_daily_ohlcv_days=3650,
        map_poll_seconds=10.0,
        grafana_url="http://test-grafana:3000",
        rolling_backtest_url="http://test-backtest/",
        basic_auth_user=None,
        basic_auth_password=None,
        strategies_root=PROJECT_ROOT / "config" / "strategies",
    )


def _write_monthly_kline_zip(dest: Path, rows: list[list]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    csv_body = "\n".join(",".join(str(v) for v in r) for r in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("klines.csv", csv_body)
    dest.write_bytes(buf.getvalue())


@pytest.fixture
def macro_kline_root(tmp_path: Path) -> Path:
    """Binance Vision-style monthly 1d ZIP cache (ETHUSDT Jan 2024)."""
    import pandas as pd

    sym = "ETHUSDT"
    zp = tmp_path / sym / "monthly" / "1d" / f"{sym}-1d-2024-01.zip"
    base_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    rows = []
    for day in range(31):
        t = base_ms + day * 86_400_000
        price = 100.0 + day
        rows.append(
            [
                t,
                price,
                price + 1,
                price - 1,
                price + 0.5,
                1000.0,
                t + 86_399_999,
                0,
                0,
                0,
                0,
                0,
            ]
        )
    _write_monthly_kline_zip(zp, rows)
    return tmp_path


@pytest.fixture
def client(console_settings, monkeypatch):
    from fastapi.testclient import TestClient

    from mlbot_console.main import app

    for mod in (
        "mlbot_console.config",
        "mlbot_console.routers.trade_map",
        "mlbot_console.routers.bus",
        "mlbot_console.routers.health",
        "mlbot_console.routers.constitution",
        "mlbot_console.routers.spot",
        "mlbot_console.routers.orders",
        "mlbot_console.routers.account",
        "mlbot_console.routers.links",
        "mlbot_console.routers.regime",
    ):
        monkeypatch.setattr(f"{mod}.SETTINGS", console_settings)
    return TestClient(app)
