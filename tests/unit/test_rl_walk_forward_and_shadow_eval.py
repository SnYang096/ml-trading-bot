import numpy as np
import pandas as pd

from src.time_series_model.rl.shadow_eval_3action import (
    ShadowEvalConfig,
    train_and_shadow_eval_bc3_from_logs,
)
from src.time_series_model.rl.walk_forward import (
    WalkForwardSplitConfig,
    time_ordered_split_by_symbol,
)


def test_time_ordered_split_by_symbol() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 10 + ["B"] * 10,
            "timestamp": [f"2025-01-01T{h:02d}:00:00Z" for h in range(10)]
            + [f"2025-01-02T{h:02d}:00:00Z" for h in range(10)],
            "x": list(range(20)),
        }
    )
    train_df, test_df = time_ordered_split_by_symbol(
        df, cfg=WalkForwardSplitConfig(train_ratio=0.6)
    )
    # each symbol should be split
    assert train_df["symbol"].value_counts().to_dict() == {"A": 6, "B": 6}
    assert test_df["symbol"].value_counts().to_dict() == {"A": 4, "B": 4}


def test_shadow_eval_bc3_from_logs_smoke(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 800
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    symbols = np.where(np.arange(n) % 2 == 0, "BTC", "ETH")

    dir_score = rng.normal(0, 1, size=n)
    mfe = np.abs(rng.normal(1.0, 0.3, size=n))
    mae = np.abs(rng.normal(0.8, 0.3, size=n))
    ttm = np.abs(rng.normal(1.0, 0.2, size=n))

    # simple rule mode: low mfe -> NO_TRADE; else sign(dir) -> TREND/MEAN
    mode = np.where(mfe < 0.6, "NO_TRADE", np.where(dir_score > 0, "TREND", "MEAN"))

    df = pd.DataFrame(
        {
            "symbol": symbols,
            "timestamp": ts.astype(str),
            "mode": mode,
            "head_dir_score": dir_score,
            "head_mfe_atr": mfe,
            "head_mae_atr": mae,
            "head_t_to_mfe": ttm,
            "drawdown": np.zeros(n),
        }
    )

    cfg = ShadowEvalConfig(
        state_keys=(
            "head_dir_score",
            "head_mfe_atr",
            "head_mae_atr",
            "head_t_to_mfe",
            "drawdown",
        ),
        split_cfg=WalkForwardSplitConfig(train_ratio=0.7),
    )
    _, _, metrics = train_and_shadow_eval_bc3_from_logs(
        df, cfg=cfg, out_dir=str(tmp_path / "shadow")
    )

    # Should at least learn the basic mapping reasonably well
    assert metrics["acc_vs_rule_mode"] > 0.8
    assert (tmp_path / "shadow" / "shadow_report.html").exists()
