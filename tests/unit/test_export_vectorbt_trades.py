import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.export_vectorbt_trades as evt


class _DummyVectorBT:
    def run(self, df, preds, task_type="binary", **kwargs):
        return {
            "debug": {
                "trades": [
                    {
                        "Entry Timestamp": "2025-01-01 00:00:00",
                        "Exit Timestamp": "2025-01-01 04:00:00",
                        "Symbol": "BTCUSDT",
                    }
                ]
            }
        }


def test_export_vectorbt_trades_from_artifacts(tmp_path, monkeypatch):
    meta = {
        "task_type": "binary",
        "backtest_params": {"price_col": "close"},
    }
    meta_path = tmp_path / "backtest_artifacts_meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})
    df_path = tmp_path / "backtest_df_test.parquet"
    df.to_parquet(df_path)

    preds = np.array([0.1, 0.2, 0.3], dtype=float)
    preds_path = tmp_path / "backtest_preds.npy"
    np.save(preds_path, preds)

    out_path = tmp_path / "trades.json"
    monkeypatch.setattr(evt, "VectorBTBacktest", _DummyVectorBT)

    argv = [
        "export_vectorbt_trades.py",
        "--meta",
        str(meta_path),
        "--df",
        str(df_path),
        "--preds",
        str(preds_path),
        "--out",
        str(out_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    evt.main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["meta"]["n_trades"] == 1
    assert payload["trades"][0]["Symbol"] == "BTCUSDT"
