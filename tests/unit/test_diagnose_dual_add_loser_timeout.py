"""loser_timeout reseed guard in diagnose_dual_add_trend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_dual_add_trend import DualAddConfig, simulate_dual_add_segment


def _synthetic_segment(*, n: int = 80) -> pd.DataFrame:
  idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
  close = pd.Series(100.0 + np.arange(n, dtype=float) * 0.02, index=idx)
  return pd.DataFrame(
    {
      "open": close,
      "high": close + 0.05,
      "low": close - 0.05,
      "close": close,
      "atr14": 1.0,
      "trend_direction": "DOWN",
    },
    index=idx,
  )


def _run(*, reseed_on_loser_timeout: bool, max_loser_hold_bars: int) -> int:
  cfg = DualAddConfig(
    add_mode="trend",
    initial_hedge=False,
    reseed_on_flip=False,
    reseed_on_loser_timeout=reseed_on_loser_timeout,
    max_loser_hold_bars=max_loser_hold_bars,
    max_gross_exposure=4,
    max_net_exposure=2,
    max_adds_per_side=0,
    risk_stop_mode="regime_only",
    tp_pct=0.05,
    step_atr_mult=0.5,
  )
  trades, _ = simulate_dual_add_segment(
    _synthetic_segment(),
    cfg=cfg,
    symbol="BTCUSDT",
    segment_id="test_seg",
    direction="DOWN",
  )
  return len(trades)


def test_no_reseed_after_loser_timeout_reduces_trade_churn() -> None:
  churn_on = _run(reseed_on_loser_timeout=True, max_loser_hold_bars=5)
  churn_off = _run(reseed_on_loser_timeout=False, max_loser_hold_bars=5)
  assert churn_on > churn_off
  assert churn_off >= 1
