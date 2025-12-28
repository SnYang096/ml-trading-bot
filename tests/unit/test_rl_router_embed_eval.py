import numpy as np
import pandas as pd

from src.time_series_model.rl.router_embed_eval import run_router_embed_eval


def test_router_embed_eval_smoke(tmp_path) -> None:
    n = 300
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(0)

    # Make a learnable-ish relationship: when dir_score positive -> TREND else MEAN,
    # and trend returns slightly better.
    dir_score = rng.normal(0, 1, size=n)
    mfe = np.abs(rng.normal(1.2, 0.2, size=n))
    mae = np.abs(rng.normal(0.8, 0.2, size=n))
    ttm = np.abs(rng.normal(12.0, 2.0, size=n))
    mode = np.where(mfe < 0.4, "NO_TRADE", np.where(dir_score > 0, "TREND", "MEAN"))
    ret_mean = 0.001 + rng.normal(0, 0.0005, size=n)
    ret_trend = 0.0015 + rng.normal(0, 0.0005, size=n)

    df = pd.DataFrame(
        {
            "symbol": ["AAA"] * n,
            "timestamp": ts,
            "mode": mode,
            "head_dir_score": dir_score,
            "head_mfe_atr": mfe,
            "head_mae_atr": mae,
            "head_t_to_mfe": ttm,
            "drawdown": np.zeros(n),
            "ret_mean": ret_mean,
            "ret_trend": ret_trend,
        }
    )

    out = run_router_embed_eval(df, out_dir=str(tmp_path / "embed_eval"))
    assert "baseline" in out and "embed" in out
    assert (tmp_path / "embed_eval" / "report.html").exists()
    assert (tmp_path / "embed_eval" / "summary.json").exists()
