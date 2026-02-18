from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from data_tools.data_utils import load_raw_data

from .crypto_factors import add_crypto_cross_sectional_factors
from .panel import FactorPanelBuilder, PanelConfig


@dataclass(frozen=True)
class PanelGenerationConfig:
    """
    Lightweight cross-sectional panel generator.

    This is intentionally simple and designed for:
    - assembling a multi-asset panel from existing OHLCV parquet data
    - computing a forward-return target
    - (optional) adding crypto-specific cross-sectional factors

    Note:
    - The output is saved as a flat table with `timestamp` and `symbol` columns,
      which downstream CS scripts load and re-index.
    """

    symbols: List[str]
    timeframe: str = "15T"
    horizon: int = 12
    data_path: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    feature_type: str = "baseline"  # baseline|comprehensive
    dropna: bool = True
    save_path: str = "results/feature_exports/cs_panel.parquet"
    include_order_flow: bool = True


def generate_cross_sectional_panel(
    cfg: PanelGenerationConfig,
) -> Tuple[pd.DataFrame, str]:
    """
    Generate a multi-asset factor panel and write to cfg.save_path.

    Returns:
        (panel_df, target_col)
    """
    if not cfg.symbols:
        raise ValueError("PanelGenerationConfig.symbols is empty")
    if cfg.horizon <= 0:
        raise ValueError("horizon must be positive")

    # Load per-symbol OHLCV frames
    frames = {}
    for sym in cfg.symbols:
        df = load_raw_data(
            data_path=cfg.data_path or "data/parquet_data",
            symbol=sym,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            timeframe=cfg.timeframe,
        )
        if df is None or df.empty:
            continue
        df = df.copy()
        if "symbol" not in df.columns:
            df["symbol"] = sym
        frames[sym] = df

    if not frames:
        raise ValueError("No data loaded for any symbols (all frames empty).")

    target_col = f"future_return_{int(cfg.horizon)}"
    panel_cfg = PanelConfig(
        timestamp_col="timestamp",
        symbol_col="symbol",
        target_col=target_col,
        forward_return_horizon=int(cfg.horizon),
        min_assets_per_ts=2,
        fill_method="ffill",
        align_intersection_only=False,
        check_duplicates=True,
        sort_index=True,
        dropna_after_fill=cfg.dropna,
    )
    builder = FactorPanelBuilder(panel_cfg)
    panel = builder.from_symbol_frames(frames)

    # Optionally add multi-asset crypto factors (cross-sectional enrichment)
    feature_type = str(cfg.feature_type or "baseline").strip().lower()
    if feature_type not in {"baseline", "comprehensive"}:
        raise ValueError("feature_type must be one of: baseline, comprehensive")
    if feature_type == "comprehensive":
        panel = add_crypto_cross_sectional_factors(panel)

    # Flatten to table for storage
    out = panel.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    if cfg.dropna:
        out = out.dropna(subset=[target_col], how="any")

    save_path = Path(cfg.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(save_path, index=False)

    return out, target_col
