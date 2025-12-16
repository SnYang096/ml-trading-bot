import pandas as pd

from src.features.loader.parallel_computer import _build_call_args


def test_selector_autowires_required_columns_when_no_column_mappings():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    feature_info = {
        "compute_func": "select_columns_from_series",
        "pass_full_df": False,
        "required_columns": ["a", "b"],
        "compute_params": {"output_columns": ["a", "b"]},
        # deliberately no column_mappings
    }

    args, kwargs = _build_call_args(feature_info, df)
    assert args == []
    assert "a" in kwargs and "b" in kwargs
    assert kwargs["a"].equals(df["a"])
    assert kwargs["b"].equals(df["b"])
