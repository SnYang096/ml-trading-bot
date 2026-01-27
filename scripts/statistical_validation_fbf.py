#!/usr/bin/env python3
"""
P2 statistical validation for FBF:
- mean return + p-value vs baseline
- bootstrap Sharpe CI
- regime mean return comparison (optional)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))


def _bootstrap_sharpe(returns: np.ndarray, n: int = 200) -> List[float]:
    rng = np.random.default_rng(42)
    if len(returns) == 0:
        return [0.0]
    samples = []
    for _ in range(n):
        idx = rng.integers(0, len(returns), size=len(returns))
        samples.append(_sharpe(returns[idx]))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="FBF statistical validation (P2)")
    parser.add_argument("--labels", required=True, help="labeled parquet")
    parser.add_argument("--ret-col", default="ret_mean", help="return column")
    parser.add_argument("--out", required=True, help="output json path")
    parser.add_argument("--regime-col", default=None, help="optional regime column")
    args = parser.parse_args()

    df = pd.read_parquet(args.labels)
    if args.ret_col not in df.columns:
        raise KeyError(f"missing return column: {args.ret_col}")

    returns = df[args.ret_col].dropna().to_numpy(dtype=float)
    mean_return = float(np.mean(returns)) if len(returns) else 0.0

    # naive p-value vs 0 using t-test approximation
    std = float(np.std(returns)) if len(returns) else 0.0
    p_value = 1.0
    if len(returns) > 2 and std > 0:
        t_stat = mean_return / (std / np.sqrt(len(returns)))
        # two-sided normal approximation
        p_value = float(2 * (1 - 0.5 * (1 + np.math.erf(abs(t_stat) / np.sqrt(2)))))

    sharpe_samples = _bootstrap_sharpe(returns)
    ci_low, ci_high = np.percentile(sharpe_samples, [2.5, 97.5]).tolist()

    output: Dict[str, object] = {
        "mean_return": mean_return,
        "p_value": p_value,
        "bootstrap_sharpe_ci": [ci_low, ci_high],
        "sharpe_mean": float(np.mean(sharpe_samples)),
    }

    if args.regime_col and args.regime_col in df.columns:
        output["regime_means"] = (
            df.groupby(args.regime_col)[args.ret_col].mean().to_dict()
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
