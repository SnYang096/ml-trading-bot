from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_detection import (
    RegimeDetectorConfig,
    RegimeHMMSmoother,
    RegimeLabel,
    RuleBasedRegimeDetector,
)


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "parquet_data"
BTC_SAMPLE = DATA_ROOT / "BTCUSDT_2021-01.parquet"


@pytest.fixture(scope="module")
def sample_data() -> pd.DataFrame:
    df = pd.read_parquet(BTC_SAMPLE)
    df = df.sort_index()
    return df.iloc[:4000].copy()


def _count_switches(labels: pd.Series) -> int:
    switches = labels != labels.shift(1)
    return int(switches.sum())


def test_rule_based_regime_consistency(sample_data: pd.DataFrame) -> None:
    config = RegimeDetectorConfig(hmm_enabled=False)
    detector = RuleBasedRegimeDetector(config)
    result = detector.detect(sample_data)

    assert not result.labels.isna().all()

    trend_scores = result.decision_factors["trend_score"]
    vol_regime = result.decision_factors["vol_regime"]

    trending_mask = result.labels == RegimeLabel.TRENDING
    range_mask = result.labels == RegimeLabel.RANGE

    if trending_mask.sum() == 0 or range_mask.sum() == 0:
        pytest.skip("Insufficient data points for regime comparison.")

    trending_trend_score = trend_scores[trending_mask].mean()
    range_trend_score = trend_scores[range_mask].mean()
    assert trending_trend_score > range_trend_score

    trending_vol = vol_regime[trending_mask].mean()
    range_vol = vol_regime[range_mask].mean()
    assert trending_vol > range_vol


def test_hmm_smoothing_reduces_switching(sample_data: pd.DataFrame) -> None:
    config = RegimeDetectorConfig(hmm_enabled=False)
    detector = RuleBasedRegimeDetector(config)
    smoother = RegimeHMMSmoother(
        n_states=4,
        n_iter=150,
        min_sequence_length=200,
    )

    try:
        result = detector.detect(sample_data, hmm_smoother=smoother)
    except ImportError:
        pytest.skip("hmmlearn is required for this test.")

    if result.smoothed_labels is None or result.label_probabilities is None:
        pytest.skip("HMM smoothing did not yield results (insufficient data).")

    base_switches = _count_switches(result.labels)
    smoothed_switches = _count_switches(result.smoothed_labels)
    assert smoothed_switches <= base_switches

    row_sums = result.label_probabilities.sum(axis=1).dropna()
    assert np.allclose(row_sums.values, 1.0, atol=1e-6)


