import pandas as pd

from src.features.loader.feature_computer import _build_call_args


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


def test_selector_preserves_required_columns_order_for_many_columns():
    cols = [f"c{i}" for i in range(30)]
    df = pd.DataFrame(
        {c: [float(i), float(i + 1), float(i + 2)] for i, c in enumerate(cols)},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    required = list(reversed(cols))

    feature_info = {
        "compute_func": "select_columns_from_series",
        "pass_full_df": False,
        "required_columns": required,
        "compute_params": {"output_columns": required},
    }
    args, kwargs = _build_call_args(feature_info, df)
    assert args == []
    out = kwargs.pop("output_columns", None)
    assert out == required
