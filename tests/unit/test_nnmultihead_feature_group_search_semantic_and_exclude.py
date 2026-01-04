from __future__ import annotations


def test_feature_selector_exclude_columns_drops_atr(tmp_path):
    import pandas as pd

    from src.time_series_model.models.nn.feature_selector import (
        select_columns_from_requested_features,
    )

    df = pd.DataFrame(
        {
            "atr": [1.0, 2.0, 3.0],
            "rsi": [10.0, 20.0, 30.0],
        }
    )
    cols = select_columns_from_requested_features(
        df,
        list(df.columns),
        requested_features=["atr_f", "rsi_f"],
        exclude_columns=["atr"],
        drop_constant=False,
    )
    assert "atr" not in cols
    assert "rsi" in cols


def test_nnmultihead_search_merges_semantic_groups(tmp_path):
    from src.time_series_model.diagnostics import nn_feature_group_search as nfgs

    groups_yaml = tmp_path / "groups.yaml"
    groups_yaml.write_text(
        "groups:\n  wick_scene:\n    - wick_scene_semantic_scores_f\n",
        encoding="utf-8",
    )
    groups = nfgs._load_groups_yaml(groups_yaml)
    assert "wick_scene" in groups
