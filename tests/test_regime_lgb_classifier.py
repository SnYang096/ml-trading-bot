import numpy as np
import pandas as pd

from regime_detection.detector import RegimeLabel
from regime_detection.lgb_classifier import LGBRegimeClassifier, RegimeClassifierConfig


def _make_dataset(n: int = 300, seed: int = 123) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    close = np.linspace(0, 10, n)
    slope = np.gradient(close)
    noise = rng.normal(0, 0.5, n)

    features = pd.DataFrame(
        {
            "trend": slope + noise,
            "volatility": np.abs(noise),
            "compression": np.cos(close) + rng.normal(0, 0.1, n),
        },
    index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )

    labels = []
    for val in features["trend"]:
        if val > 0.6:
            labels.append(RegimeLabel.TRENDING)
        elif val < -0.6:
            labels.append(RegimeLabel.COLLAPSE)
        else:
            labels.append(RegimeLabel.RANGE)
    return features, pd.Series(labels, index=features.index)


def test_lgb_regime_classifier_fit_predict():
    X, y = _make_dataset()
    config = RegimeClassifierConfig(n_estimators=50, learning_rate=0.1, random_state=7)
    clf = LGBRegimeClassifier(config=config)
    metrics = clf.fit(X, y)

    assert metrics["train_accuracy"] > 0.8
    preds = clf.predict(X)
    assert preds.index.equals(X.index)
    assert set(preds.unique()).issubset(set(RegimeLabel))

    proba = clf.predict_proba(X.head(5))
    assert np.allclose(proba.sum(axis=1).values, 1.0, atol=1e-6)
    assert list(proba.columns) == [label.value for label in RegimeLabel]

    importance = clf.feature_importances()
    assert set(importance.index) == set(X.columns)
    assert importance.max() > 0.0

