from pathlib import Path

import scripts.auto_research_pipeline as arp


def test_parse_multileg_metrics_dual_add_empty_summary_csv(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    summary.write_text("", encoding="utf-8")

    metrics = arp._parse_multileg_metrics("dual_add_trend", tmp_path)
    assert metrics["n_trades"] == 0
    assert metrics["sharpe_r"] == 0.0
