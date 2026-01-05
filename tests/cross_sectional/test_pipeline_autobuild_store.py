from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd


def test_pipeline_autobuild_store_triggers_builder(tmp_path: Path):
    # Import the pipeline module in-process so we can patch.
    from src.cross_sectional.scripts import pipeline as pl

    cfg = {
        "panel": {
            "source": "feature_store",
            "feature_store": {
                "root": str(tmp_path / "fs"),
                "layer": "layer_x",
                "timeframe": "240T",
                "symbols": "AAA,BBB",
                "start_date": "2024-01-01",
                "end_date": "2024-02-28",
            },
            "auto_build_store": {
                "enabled": True,
                "data_path": str(tmp_path / "data"),
                "feature_deps": "config/feature_dependencies.yaml",
            },
        },
        "factor_eval": {
            "factor_set_yaml": "config/cross_sectional/cs_factor_sets_crypto.yaml",
            "factor_set": "crypto_alpha101_cs_rank",
        },
    }

    # Create an incomplete FS layout so the checker sees missing months
    root = Path(cfg["panel"]["feature_store"]["root"])
    layer = cfg["panel"]["feature_store"]["layer"]
    tf = cfg["panel"]["feature_store"]["timeframe"]
    (root / layer / "AAA" / tf).mkdir(parents=True, exist_ok=True)
    # only write 2024-01 for AAA; BBB missing entirely
    pd.DataFrame(
        {"timestamp": [pd.Timestamp("2024-01-01")], "symbol": ["AAA"], "close": [1.0]}
    ).to_parquet(
        root / layer / "AAA" / tf / "2024-01.parquet",
        index=False,
    )

    with patch(
        "cross_sectional.feature_store_builder.build_feature_store_for_symbols"
    ) as m_build:
        # We don't want to actually build or read; just check trigger.
        pl._maybe_autobuild_feature_store(cfg, out_root=tmp_path)
        assert m_build.called
