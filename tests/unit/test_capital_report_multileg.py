import json

import pytest

from scripts.capital_report import write_capital_report_from_trades


def _write_trades(path, rows: str) -> None:
    path.write_text(
        "exit_time,symbol,pnl_per_capital\n" + rows,
        encoding="utf-8",
    )


def test_capital_report_multileg_splits_total_notional(tmp_path) -> None:
    trades = tmp_path / "grid_trades.csv"
    _write_trades(
        trades,
        "2024-01-01T00:00:00+00:00,BTCUSDT,0.10\n"
        "2024-01-02T00:00:00+00:00,ETHUSDT,0.20\n",
    )
    portfolio_initial = 20_000.0  # 2 symbols × 10k each

    report = write_capital_report_from_trades(
        trades_path=trades,
        out_dir=tmp_path / "out",
        unit="capital_normalized",
        title="Multileg Capital Report",
        initial_capital=portfolio_initial,
        n_symbols=2,
        total_r=0.15,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    assert report["initial_capital"] == portfolio_initial
    assert report["final_capital"] == pytest.approx(23_000.0)
    assert report["estimated_profit_usd"] == pytest.approx(3_000.0)
    assert report["total_return"] == pytest.approx(0.15)
    assert report["total_r"] == pytest.approx(0.15)
    assert "timeline portfolio pnl_per_capital" in report["unit_explanation"]

    payload = json.loads((tmp_path / "out" / "capital_report.json").read_text())
    assert payload["final_capital"] == pytest.approx(23_000.0)


def test_capital_report_multileg_auto_detects_n_symbols(tmp_path) -> None:
    trades = tmp_path / "grid_trades.csv"
    _write_trades(
        trades,
        "2024-01-01T00:00:00+00:00,BTCUSDT,0.10\n"
        "2024-01-02T00:00:00+00:00,ETHUSDT,0.10\n"
        "2024-01-03T00:00:00+00:00,SOLUSDT,0.10\n",
    )

    report = write_capital_report_from_trades(
        trades_path=trades,
        out_dir=tmp_path,
        unit="capital_normalized",
        title="Auto-detect symbols",
        initial_capital=30_000.0,
    )

    assert report["final_capital"] == pytest.approx(33_000.0)
    assert report["total_r"] == pytest.approx(0.10)


def test_capital_report_single_symbol_unchanged(tmp_path) -> None:
    trades = tmp_path / "trades.csv"
    _write_trades(trades, "2024-01-01T00:00:00+00:00,BTCUSDT,0.08\n")

    report = write_capital_report_from_trades(
        trades_path=trades,
        out_dir=tmp_path,
        unit="capital_normalized",
        title="Single symbol",
        initial_capital=10_000.0,
    )

    assert report["final_capital"] == pytest.approx(10_800.0)
    assert report["total_r"] == pytest.approx(0.08)
