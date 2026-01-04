import pandas as pd

from scripts.train_strategy_pipeline import determine_feature_columns
from src.time_series_model.strategy_config.loader import FeaturePipelineConfig


def test_tree_determine_feature_columns_exclude_columns_drops_atr():
    # df keeps atr available (labels/backtest may need it)
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
            "atr": [0.1, 0.2],
            "rsi": [50.0, 55.0],
            "signal": [0.0, 0.0],
        }
    )

    cfg = FeaturePipelineConfig(
        requested_features=[],
        exclude_columns=["atr"],
    )

    cols = determine_feature_columns(df, cfg)
    assert "atr" not in cols
    assert "rsi" in cols
