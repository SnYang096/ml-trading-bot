"""Unit tests for data processors."""

import numpy as np
import pandas as pd
import pytest

from src.data_tools.processors import (
    FillNaProcessor,
    ClipOutlierProcessor,
    DtypeDowncastProcessor,
    ReplaceInfProcessor,
    ProcessorChain,
    get_default_processor_chain,
    get_live_processor_chain,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with various edge cases."""
    return pd.DataFrame(
        {
            "normal": [1.0, 2.0, 3.0, 4.0, 5.0],
            "with_nan": [1.0, np.nan, 3.0, np.nan, 5.0],
            "with_inf": [1.0, np.inf, 3.0, -np.inf, 5.0],
            "outliers": [1.0, 2.0, 100.0, 3.0, 4.0],  # 100 is outlier
            "int_col": [1, 2, 3, 4, 5],
        }
    )


class TestFillNaProcessor:
    def test_ffill(self, sample_df):
        proc = FillNaProcessor(method="ffill")
        result = proc.process(sample_df)
        assert result["with_nan"].isna().sum() == 0
        assert result["with_nan"].iloc[1] == 1.0  # filled with previous value

    def test_bfill(self, sample_df):
        proc = FillNaProcessor(method="bfill")
        result = proc.process(sample_df)
        assert result["with_nan"].isna().sum() == 0
        assert result["with_nan"].iloc[1] == 3.0  # filled with next value

    def test_zero_fill(self, sample_df):
        proc = FillNaProcessor(method="zero")
        result = proc.process(sample_df)
        assert result["with_nan"].isna().sum() == 0
        assert result["with_nan"].iloc[1] == 0.0

    def test_specific_columns(self, sample_df):
        proc = FillNaProcessor(method="zero", columns=["with_nan"])
        result = proc.process(sample_df)
        assert result["with_nan"].iloc[1] == 0.0


class TestClipOutlierProcessor:
    def test_std_mode(self):
        # Create a larger sample with clear outlier
        df = pd.DataFrame(
            {
                "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1000.0],
            }
        )
        proc = ClipOutlierProcessor(mode="std", std_threshold=2.0)
        result = proc.process(df)
        # 1000 should be clipped
        assert result["values"].max() < 1000

    def test_percentile_mode(self, sample_df):
        proc = ClipOutlierProcessor(mode="percentile", lower_pct=10, upper_pct=90)
        result = proc.process(sample_df)
        assert result["outliers"].max() < 100

    def test_exclude_columns(self, sample_df):
        proc = ClipOutlierProcessor(
            mode="std", std_threshold=2.0, exclude_columns=["outliers"]
        )
        result = proc.process(sample_df)
        # outliers column should not be clipped
        assert result["outliers"].max() == 100


class TestDtypeDowncastProcessor:
    def test_float64_to_float32(self, sample_df):
        # Ensure input is float64
        df = sample_df.copy()
        df["normal"] = df["normal"].astype(np.float64)

        proc = DtypeDowncastProcessor(float_dtype="float32")
        result = proc.process(df)
        assert result["normal"].dtype == np.float32

    def test_int64_to_int32(self, sample_df):
        df = sample_df.copy()
        df["int_col"] = df["int_col"].astype(np.int64)

        proc = DtypeDowncastProcessor(int_dtype="int32")
        result = proc.process(df)
        assert result["int_col"].dtype == np.int32


class TestReplaceInfProcessor:
    def test_replace_with_nan(self, sample_df):
        proc = ReplaceInfProcessor(replace_with="nan")
        result = proc.process(sample_df)
        assert not np.isinf(result["with_inf"]).any()
        assert result["with_inf"].isna().sum() == 2  # Two inf values replaced with nan

    def test_replace_with_clip(self, sample_df):
        proc = ReplaceInfProcessor(replace_with="clip")
        result = proc.process(sample_df)
        assert not np.isinf(result["with_inf"]).any()
        assert result["with_inf"].max() == 5.0  # Clipped to max finite value
        assert result["with_inf"].min() == 1.0  # Clipped to min finite value


class TestProcessorChain:
    def test_chain_order(self, sample_df):
        # First replace inf, then fill nan
        chain = ProcessorChain(
            [
                ReplaceInfProcessor(replace_with="nan"),
                FillNaProcessor(method="ffill"),
            ]
        )
        result = chain.process(sample_df)

        # Should have no inf and no nan
        assert not np.isinf(result["with_inf"]).any()
        assert result["with_inf"].isna().sum() == 0

    def test_add_processor(self):
        chain = ProcessorChain()
        chain.add(FillNaProcessor(method="zero"))
        chain.add(DtypeDowncastProcessor())

        assert len(chain.processors) == 2

    def test_empty_chain(self, sample_df):
        chain = ProcessorChain([])
        result = chain.process(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)


class TestDefaultChains:
    def test_default_chain(self, sample_df):
        chain = get_default_processor_chain()
        result = chain.process(sample_df)

        # Should handle inf and nan
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
        # Dtypes should be downcast
        assert result["normal"].dtype == np.float32

    def test_live_chain(self, sample_df):
        chain = get_live_processor_chain()
        result = chain.process(sample_df)

        # Should handle inf and nan
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
