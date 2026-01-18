import json
import sys
from pathlib import Path

import pandas as pd


def test_portfolio_allocation_plateau_smoke(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="4H")
    mode = pd.DataFrame(
        {
            "timestamp": idx.repeat(2),
            "symbol": ["BTCUSDT", "ETHUSDT"] * len(idx),
            "mode": ["TREND", "MEAN"] * len(idx),
        }
    )
    mode_path = tmp_path / "mode.parquet"
    mode.to_parquet(mode_path)

    metrics = {
        "rule_pcm_sharpe_mean": 0.5,
        "rule_pcm_avg_max_dd": 0.1,
        "rule_avg_max_dd": 0.1,
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    gate_yaml = tmp_path / "gate.yaml"
    gate_yaml.write_text(
        "\n".join(
            [
                "layer: portfolio_allocation",
                "hard_fail:",
                "  pa__avg_weight__GLOBAL_CASH: {min: 0.0, max: 1.0, optional: true}",
            ]
        ),
        encoding="utf-8",
    )

    cfg = tmp_path / "portfolio_assets.yaml"
    cfg.write_text(
        "\n".join(
            [
                "version: 1",
                "name: test_assets",
                "assets:",
                "  GLOBAL_TREND: {max_weight: 0.4, can_be_zero: true}",
                "  GLOBAL_MEAN: {min_weight: 0.2, max_weight: 0.35, can_be_zero: false}",
                "  GLOBAL_CASH: {min_weight: 0.1, max_weight: 1.0}",
                "  HIGH_BETA_OVERLAY: {max_weight: 0.1, can_be_zero: true}",
                "  DEFENSIVE_MEAN: {max_weight: 0.25, can_be_zero: true}",
                "router_to_weights:",
                "  global_trend: {p_trend_min: 0.6, regime_entropy_max: 0.4, max_weight: 0.4}",
                "  global_mean: {base_floor: 0.2, max_weight: 0.35}",
                "  global_cash: {min_weight: 0.1}",
                "  high_beta_overlay: {p_trend_min: 0.75, crowding_max: 0.3, confidence_min: 0.7, max_weight: 0.1}",
                "  defensive_mean: {regime_entropy_min: 0.5, max_weight: 0.25}",
                "trend_zero_law:",
                "  rules:",
                "    - name: uncertainty_high",
                "      regime_entropy_gt: 0.45",
                "    - name: portfolio_dd_protection",
                "      portfolio_drawdown_gt: 0.12",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    from scripts.diagnose_portfolio_allocation_plateau import main as plateau_main

    args = [
        "--mode",
        str(mode_path),
        "--portfolio-assets-yaml",
        str(cfg),
        "--metrics-json",
        str(metrics_path),
        "--sweep-target",
        "trend_zero_regime_entropy_gt",
        "--grid",
        "0.4,0.6",
        "--gate-yaml",
        str(gate_yaml),
        "--out",
        str(out_dir),
    ]

    old = sys.argv
    try:
        sys.argv = ["diagnose_portfolio_allocation_plateau.py"] + args
        plateau_main()
    finally:
        sys.argv = old

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert (out_dir / "plateau.csv").exists()
    assert (out_dir / "report.md").exists()
    plateau_vals = summary.get("plateau_vals") or []
    assert plateau_vals
    if len(plateau_vals) % 2 == 1:
        expected = float(sorted(plateau_vals)[len(plateau_vals) // 2])
    else:
        vals = sorted(plateau_vals)
        expected = 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
    assert summary["selected_value"] == expected
