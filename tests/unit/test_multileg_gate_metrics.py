import json
from pathlib import Path

from scripts.multileg_gate import _load_strategy_gate_metrics


def test_multileg_gate_loads_grid_standalone_metrics(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "chop_grid"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "trade_summary": {
                        "trades": 12,
                        "sum_pnl_per_capital": 1.2,
                        "forced_rate": 0.1,
                        "max_drawdown": -0.02,
                    },
                    "segment_summary": {
                        "worst_segment": -0.01,
                        "segment_win_rate": 0.6,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    got = _load_strategy_gate_metrics(
        run_dir=tmp_path,
        strategy="chop_grid",
        strategy_type="grid",
    )
    assert got["n_trades"] == 12
    assert got["total_r"] == 1.2
    assert got["segment_win_rate"] == 0.6


def test_multileg_gate_prefers_rolling_summary(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "chop_grid"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "multileg_summary.json").write_text(
        json.dumps({"metrics": {"n_trades": 99, "total_r": 2.5}}),
        encoding="utf-8",
    )
    got = _load_strategy_gate_metrics(
        run_dir=tmp_path,
        strategy="chop_grid",
        strategy_type="grid",
    )
    assert got == {"n_trades": 99, "total_r": 2.5}
