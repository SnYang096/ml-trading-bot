#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.time_series_model.rl.router_embed_eval import (  # noqa: E402
    RouterEmbedEvalConfig,
    run_router_embed_eval,
)
from src.time_series_model.rl.regime_embedding import (
    RegimeEmbeddingConfig,
)  # noqa: E402
from src.time_series_model.rl.shadow_eval_3action import ShadowEvalConfig  # noqa: E402
from src.time_series_model.rl.counterfactual_eval_3action import (
    CounterfactualEvalConfig,
)  # noqa: E402
from src.time_series_model.rl.walk_forward import WalkForwardSplitConfig  # noqa: E402


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="A/B eval: BC baseline vs +regime one-hot embedding."
    )
    ap.add_argument(
        "--logs",
        required=True,
        help="Logs .csv/.parquet (symbol,timestamp,mode,ret_mean,ret_trend,head_*,drawdown)",
    )
    ap.add_argument("--out", required=True, help="Output directory root for artifacts.")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--regime-buckets", type=int, default=4)
    args = ap.parse_args()

    df = _read_any(args.logs)
    cfg = RouterEmbedEvalConfig(
        regime_cfg=RegimeEmbeddingConfig(n_buckets=int(args.regime_buckets)),
        shadow_cfg=ShadowEvalConfig(
            split_cfg=WalkForwardSplitConfig(train_ratio=float(args.train_ratio))
        ),
        cf_cfg=CounterfactualEvalConfig(
            split_cfg=WalkForwardSplitConfig(train_ratio=float(args.train_ratio))
        ),
    )
    run_router_embed_eval(df, cfg=cfg, out_dir=str(args.out))
    print("✅ router embed A/B saved to:", args.out)


if __name__ == "__main__":
    main()
