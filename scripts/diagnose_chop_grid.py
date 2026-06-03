"""Offline diagnostic for chop-regime grid trading.

Hypothesis:
  Stable boxes are rare and hard to identify causally in crypto. A broader
  "no clear trend / semantic chop" regime may be better traded with a small
  neutral grid, exiting all inventory once chop ends.

This script compares where chop regimes occur and simulates a conservative
fixed-level grid inside contiguous chop segments.

Example:
    python scripts/diagnose_chop_grid.py \\
        --start 2024-01-01 --end 2024-12-31 \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_crf_edge import (  # noqa: E402
    StudyConfig,
    _atr,
    _bb_width_pctile,
    _load_symbol_1m,
    _resample_ohlcv,
    _semantic_chop,
    build_symbol_dataset,
)
from scripts.pipeline.multileg_prefilter_rules import (  # noqa: E402
    apply_prefilter_rules,
)
from src.config.multileg_config import load_multileg_effective_config  # noqa: E402
from src.config.strategy_layout import resolve_strategy_config_input  # noqa: E402
from src.features.time_series.semantic_chop_ts_quantile import (  # noqa: E402
    semantic_chop_ts_quantile,
)
from src.feature_store.layer_naming import detect_layer_for_strategy  # noqa: E402

DEFAULT_CHOP_GRID_YAML = (
    PROJECT_ROOT / "config/strategies/chop_grid/research/calibrate_roll.default.yaml"
)


def resolve_optional_repo_path(value: Any) -> str | None:
    if value is None or value == "":
        return None
    p = Path(str(value))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


def _parse_max_replenish_inv(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "null", "none"}:
        return None
    return int(raw)


def merge_chop_grid_yaml(path: Path) -> Dict[str, Any]:
    """Load strategy knobs from ``grid.yaml`` for backtests, diagnostics, sweeps.

    Includes ``regime.box_prefilter`` (StudyConfig / ``box_prefilter`` column),
    ``chop_series`` (raw vs ts_quantile), and risk/spacing used by CLIs.
    """
    if not path.exists():
        return {}
    cfg_dir, profile_path, engine_path = resolve_strategy_config_input(path)
    cfg = load_multileg_effective_config(
        config_dir=cfg_dir,
        strategy_type="grid",
        profile_path=profile_path,
        engine_path=engine_path,
    )
    regime = cfg.get("regime", {}) or {}
    inv = cfg.get("inventory", {}) or {}
    spacing = inv.get("spacing", {}) or {}
    risk = cfg.get("risk", {}) or {}
    grid_bt = cfg.get("grid_backtest", {}) or {}
    costs = grid_bt.get("costs", {}) if isinstance(grid_bt, dict) else {}
    if not isinstance(costs, dict):
        costs = {}
    box_pf = regime.get("box_prefilter") or {}
    chop_series = cfg.get("chop_series", {}) or {}
    feature_store_dir = resolve_optional_repo_path(grid_bt.get("feature_store_dir"))
    feature_store_layer = grid_bt.get("feature_store_layer")
    feature_store_timeframe = grid_bt.get("feature_store_timeframe")
    if feature_store_layer is not None:
        feature_store_layer = str(feature_store_layer).strip() or None
    if feature_store_timeframe is not None:
        feature_store_timeframe = str(feature_store_timeframe).strip() or None
    if feature_store_dir and not feature_store_layer:
        feature_store_layer = detect_layer_for_strategy(
            strategy="chop_grid",
            features_store_root=str(feature_store_dir),
            timeframe=feature_store_timeframe,
        )

    out: Dict[str, Any] = {
        "box_window": int(regime.get("box_window", 120)),
        "chop_min": float(regime.get("entry_chop_min", 0.40)),
        "exit_chop_min": float(regime.get("exit_chop_below", 0.25)),
        "grid_atr_mult": float(spacing.get("atr_mult", 0.50)),
        "grid_pct": float(spacing.get("min_pct", 0.004)),
        "max_levels": int(inv.get("max_levels_per_side", 3)),
        "max_replenish_per_level": _parse_max_replenish_inv(
            inv.get("max_replenish_per_level_per_segment")
        ),
        "same_bar_entry_exit": bool(grid_bt.get("same_bar_entry_exit", False)),
        "min_segment_bars": int(risk.get("min_segment_bars", 6)),
        "max_segment_bars": int(risk.get("max_segment_bars", 120)),
        "fee_bps": float(costs.get("fee_bps", risk.get("fee_bps", 4.0))),
        "maker_fee_bps": float(
            costs.get(
                "maker_fee_bps", risk.get("maker_fee_bps", risk.get("fee_bps", 4.0))
            )
        ),
        "taker_fee_bps": float(
            costs.get(
                "taker_fee_bps", risk.get("taker_fee_bps", risk.get("fee_bps", 4.0))
            )
        ),
        "forced_exit_slippage_bps": float(
            costs.get(
                "forced_exit_slippage_bps", risk.get("forced_exit_slippage_bps", 0.0)
            )
        ),
        "funding_cost_bps_per_8h": float(
            costs.get(
                "funding_cost_bps_per_8h", risk.get("funding_cost_bps_per_8h", 0.0)
            )
        ),
        "exclude_box": bool(regime.get("exclude_box_prefilter", True)),
        # Live chop: exclude_box_prefilter=false → block entry on stable-box bars.
        "block_stable_box": not bool(regime.get("exclude_box_prefilter", True)),
        "max_loss_per_grid": float(risk.get("max_loss_per_grid", 0.03)),
        "max_open_levels_total": int(risk.get("max_open_levels_total", 6)),
        "stability_min": float(box_pf.get("stability_min", 0.85)),
        "width_min": float(box_pf.get("width_min", 0.04)),
        "width_max": float(box_pf.get("width_max", 0.30)),
        "touches_min": int(box_pf.get("touches_min", 5)),
        "chop_signal": str(chop_series.get("chop_signal", "raw")),
        "chop_ts_window": int(chop_series.get("chop_ts_window", 1200)),
        "chop_ts_min_periods": int(chop_series.get("chop_ts_min_periods", 150)),
        "feature_store_dir": feature_store_dir,
        "feature_store_layer": feature_store_layer,
        "feature_store_timeframe": feature_store_timeframe,
        "execution_timeframe": str(grid_bt.get("execution_timeframe", "1min")),
        "agg_data_dir": resolve_optional_repo_path(
            grid_bt.get("agg_data_dir") or "data/agg_data"
        ),
        "prefilter_rules": cfg.get("rules", []) or [],
    }
    if "compute_semantic_chop_ts_q" in chop_series:
        out["compute_chop_ts_q"] = chop_series.get("compute_semantic_chop_ts_q")
    return out


def prefilter_box_range(lo: float, hi: float) -> Tuple[Dict[str, Any], ...]:
    return (
        {
            "all_of": [
                {"feature": "box_pos_60", "operator": ">=", "value": lo},
                {"feature": "box_pos_60", "operator": "<=", "value": hi},
            ]
        },
    )


def resolve_prefilter_rules(
    defaults: dict,
    *,
    box_pos_min: float | None = None,
    box_pos_max: float | None = None,
) -> Tuple[Dict[str, Any], ...]:
    if box_pos_min is not None and box_pos_max is not None:
        return prefilter_box_range(box_pos_min, box_pos_max)
    return tuple(
        x for x in (defaults.get("prefilter_rules", []) or []) if isinstance(x, dict)
    )


def recompute_box_prefilter_column(
    df: pd.DataFrame,
    *,
    stability_min: float,
    width_min: float,
    width_max: float,
    touches_min: int,
) -> pd.DataFrame:
    """Recompute ``box_prefilter`` from box_structure columns (FeatureStore path)."""
    out = df.copy()
    out["box_prefilter"] = (
        (pd.to_numeric(out["box_stability"], errors="coerce") >= stability_min)
        & (pd.to_numeric(out["box_width_pct"], errors="coerce") >= width_min)
        & (pd.to_numeric(out["box_width_pct"], errors="coerce") <= width_max)
        & (pd.to_numeric(out["box_touches_hi"], errors="coerce") >= touches_min)
        & (pd.to_numeric(out["box_touches_lo"], errors="coerce") >= touches_min)
    )
    return out


@dataclass(frozen=True)
class GridConfig:
    box_window: int = 120
    chop_min: float = 0.40
    exit_chop_min: float = 0.25
    min_segment_bars: int = 6
    max_segment_bars: int = 120
    grid_atr_mult: float = 0.75
    grid_pct: float = 0.004
    max_levels: int = 3
    fee_bps: float = 4.0
    # Regime masks: "raw" uses semantic_chop [0,1]; "ts_quantile" uses rolling
    # percentile rank of raw chop vs past window (~[0,1], causal).
    chop_signal: str = "raw"
    chop_ts_window: int = 1200
    chop_ts_min_periods: int = 150
    # None: compute ts_q column only when chop_signal == "ts_quantile" (faster raw runs).
    compute_semantic_chop_ts_q: bool | None = None
    # When chop_signal == "ts_quantile": load all strategy features from FeatureStore (no recompute).
    feature_store_dir: str | None = None
    feature_store_layer: str | None = None
    # If set, overrides bars timeframe when reading the store (e.g. store "120T" vs CLI "2h").
    feature_store_timeframe: str | None = None
    # Optional generic prefilter rules from archetypes/prefilter.yaml.
    prefilter_rules: Tuple[Dict[str, Any], ...] = ()
    # Box prefilter column (StudyConfig / build_symbol_dataset); must match grid.yaml.
    stability_min: float = 0.85
    width_min: float = 0.04
    width_max: float = 0.30
    touches_min: int = 5


def should_compute_semantic_chop_ts_q(cfg: GridConfig) -> bool:
    if cfg.compute_semantic_chop_ts_q is True:
        return True
    if cfg.compute_semantic_chop_ts_q is False:
        return False
    return cfg.chop_signal == "ts_quantile"


def regime_chop_series(df: pd.DataFrame, cfg: GridConfig) -> pd.Series:
    """Series used for chop hysteresis (same comparisons as raw semantic_chop)."""
    if cfg.chop_signal == "ts_quantile":
        if "semantic_chop_ts_q" not in df.columns:
            raise KeyError("semantic_chop_ts_q missing; call build_features first")
        return pd.to_numeric(df["semantic_chop_ts_q"], errors="coerce")
    return pd.to_numeric(df["semantic_chop"], errors="coerce")


def regime_chop_column(cfg: GridConfig) -> str:
    return "semantic_chop_ts_q" if cfg.chop_signal == "ts_quantile" else "semantic_chop"


def _chop_grid_store_requested_features() -> List[str]:
    path = PROJECT_ROOT / "config/strategies/chop_grid/features.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fp = raw.get("feature_pipeline") or {}
    feats = fp.get("requested_features")
    if not isinstance(feats, list) or not feats:
        raise ValueError(
            f"chop_grid FeatureStore path requires non-empty feature_pipeline.requested_features in {path}"
        )
    return [str(x) for x in feats]


def _materialize_chop_grid_from_store_columns(
    df: pd.DataFrame,
    cfg: GridConfig,
    study_cfg: StudyConfig,
) -> pd.DataFrame:
    out = df.copy()
    w = int(cfg.box_window)
    suf = f"_{w}"
    box_pairs = [
        ("box_width_pct", f"box_width_pct{suf}"),
        ("box_pos", f"box_pos{suf}"),
        ("box_stability", f"box_stability{suf}"),
        ("box_touches_hi", f"box_touches_hi{suf}"),
        ("box_touches_lo", f"box_touches_lo{suf}"),
    ]
    missing = [src for _, src in box_pairs if src not in out.columns]
    if missing:
        raise KeyError(
            f"FeatureStore missing box_structure columns for box_window={w}: {missing}"
        )
    for dst, src in box_pairs:
        out[dst] = pd.to_numeric(out[src], errors="coerce")

    if "bpc_semantic_chop" not in out.columns:
        raise KeyError(
            "FeatureStore missing bpc_semantic_chop (bpc_soft_phase_f output)"
        )
    out["semantic_chop"] = pd.to_numeric(out["bpc_semantic_chop"], errors="coerce")

    if cfg.chop_signal == "ts_quantile":
        if "bpc_semantic_chop_ts_q" not in out.columns:
            raise KeyError(
                "FeatureStore missing bpc_semantic_chop_ts_q "
                "(required for chop_signal=ts_quantile)"
            )
        out["semantic_chop_ts_q"] = pd.to_numeric(
            out["bpc_semantic_chop_ts_q"], errors="coerce"
        )
    elif "bpc_semantic_chop_ts_q" in out.columns and should_compute_semantic_chop_ts_q(
        cfg
    ):
        out["semantic_chop_ts_q"] = pd.to_numeric(
            out["bpc_semantic_chop_ts_q"], errors="coerce"
        )

    if "atr" in out.columns:
        out["atr14"] = pd.to_numeric(out["atr"], errors="coerce")
    elif "atr14" in out.columns:
        out["atr14"] = pd.to_numeric(out["atr14"], errors="coerce")
    else:
        raise KeyError("FeatureStore missing atr (from atr_f) or atr14")

    out["box_valid"] = (
        out["box_width_pct"].notna()
        & (out["box_width_pct"] > 0)
        & out["box_stability"].notna()
    )
    out["box_prefilter"] = (
        (out["box_stability"] >= study_cfg.stability_min)
        & (out["box_width_pct"] >= study_cfg.width_min)
        & (out["box_width_pct"] <= study_cfg.width_max)
        & (out["box_touches_hi"] >= study_cfg.touches_min)
        & (out["box_touches_lo"] >= study_cfg.touches_min)
    )
    out["edge_side"] = ""
    return out


def _build_features_from_feature_store(
    symbol: str,
    bars: pd.DataFrame,
    cfg: GridConfig,
    *,
    bars_timeframe: str,
) -> pd.DataFrame:
    if not cfg.feature_store_dir or not str(cfg.feature_store_layer or "").strip():
        raise ValueError(
            "FeatureStore-backed chop_grid backtest requires FeatureStore settings: "
            "set grid_backtest.feature_store_dir and grid_backtest.feature_store_layer "
            "in the strategy YAML (or CLI overrides)."
        )
    store_tf = (cfg.feature_store_timeframe or "").strip() or bars_timeframe.strip()
    if not store_tf:
        raise ValueError(
            "chop_signal='ts_quantile' requires bars timeframe (CLI) "
            "or grid_backtest.feature_store_timeframe in YAML."
        )

    from src.features.loader.strategy_feature_loader import StrategyFeatureLoader

    requested = _chop_grid_store_requested_features()
    base = bars.copy()
    base["symbol"] = symbol
    loader = StrategyFeatureLoader(
        feature_deps_path=str(PROJECT_ROOT / "config/feature_dependencies.yaml"),
        strategy_config_path=None,
        use_disk_cache=False,
        use_memory_cache=False,
        verbose=False,
    )
    merged = loader.load_features_from_requested(
        base,
        requested,
        fit=False,
        feature_store_dir=cfg.feature_store_dir,
        feature_store_layer=str(cfg.feature_store_layer).strip(),
        feature_store_symbol=symbol,
        feature_store_timeframe=store_tf,
        feature_store_strict=True,
    )
    study_cfg = StudyConfig(
        box_window=cfg.box_window,
        chop_min=cfg.chop_min,
        stability_min=float(cfg.stability_min),
        width_min=float(cfg.width_min),
        width_max=float(cfg.width_max),
        touches_min=int(cfg.touches_min),
    )
    return _materialize_chop_grid_from_store_columns(merged, cfg, study_cfg)


def _build_ts_quantile_features_from_feature_store(
    symbol: str,
    bars: pd.DataFrame,
    cfg: GridConfig,
    *,
    bars_timeframe: str,
) -> pd.DataFrame:
    return _build_features_from_feature_store(
        symbol,
        bars,
        cfg,
        bars_timeframe=bars_timeframe,
    )


def _segments(mask: pd.Series, *, min_len: int, max_len: int) -> List[Tuple[int, int]]:
    return _hysteresis_segments(mask, mask, min_len=min_len, max_len=max_len)


def _hysteresis_segments(
    entry_mask: pd.Series,
    hold_mask: pd.Series,
    *,
    min_len: int,
    max_len: int,
) -> List[Tuple[int, int]]:
    """Build segments that enter on entry_mask and exit once hold_mask fails."""
    entry = entry_mask.fillna(False).to_numpy(dtype=bool)
    hold = hold_mask.fillna(False).to_numpy(dtype=bool)
    segs: List[Tuple[int, int]] = []
    i = 0
    n = len(entry)
    while i < n:
        if not entry[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and hold[i + 1] and (i + 1 - start) < max_len:
            i += 1
        end = i
        if end - start + 1 >= min_len:
            segs.append((start, end))
        i += 1
    return segs


def _pnl_long(entry: float, exit_px: float, fee: float) -> float:
    return (exit_px - entry) / entry - 2.0 * fee


def _pnl_short(entry: float, exit_px: float, fee: float) -> float:
    return (entry - exit_px) / entry - 2.0 * fee


def simulate_fixed_grid(
    seg: pd.DataFrame,
    *,
    cfg: GridConfig,
) -> Dict[str, float | int | str | pd.Timestamp]:
    """Conservative fixed-level neutral grid over one chop segment.

    Levels are fixed from segment start:
      - buy levels below center; each long takes profit one spacing higher
      - sell levels above center; each short takes profit one spacing lower

    The first bar confirms the regime and anchors the grid; orders become
    eligible from the next bar. If a level fills, its target is only eligible
    from the next bar onward. This avoids using the signal bar's intrabar
    high/low after a close-confirmed signal.
    At segment exit, all open inventory is marked to the final close.
    """
    if seg.empty:
        return {"status": "empty"}
    center = float(seg["close"].iloc[0])
    atr = float(seg["atr14"].iloc[0])
    if not np.isfinite(center + atr) or center <= 0 or atr <= 0:
        return {"status": "invalid"}

    spacing = max(cfg.grid_atr_mult * atr, cfg.grid_pct * center)
    if spacing <= 0:
        return {"status": "invalid"}
    fee = cfg.fee_bps / 10000.0

    long_levels = [center - spacing * k for k in range(1, cfg.max_levels + 1)]
    short_levels = [center + spacing * k for k in range(1, cfg.max_levels + 1)]
    open_longs: Dict[int, Tuple[float, int]] = {}
    open_shorts: Dict[int, Tuple[float, int]] = {}
    realized = 0.0
    fills = 0
    cycles = 0
    max_open = 0
    pnl_path = []

    for bar_i, (_, row) in enumerate(seg.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # Close positions opened on prior bars only.
        for level_i, (entry, fill_bar) in list(open_longs.items()):
            target = entry + spacing
            if bar_i > fill_bar and high >= target:
                realized += _pnl_long(entry, target, fee)
                cycles += 1
                del open_longs[level_i]
        for level_i, (entry, fill_bar) in list(open_shorts.items()):
            target = entry - spacing
            if bar_i > fill_bar and low <= target:
                realized += _pnl_short(entry, target, fee)
                cycles += 1
                del open_shorts[level_i]

        # Fill inactive levels only after the signal bar has closed.
        if bar_i > 0:
            for level_i, px in enumerate(long_levels):
                if level_i not in open_longs and low <= px:
                    open_longs[level_i] = (px, bar_i)
                    fills += 1
            for level_i, px in enumerate(short_levels):
                if level_i not in open_shorts and high >= px:
                    open_shorts[level_i] = (px, bar_i)
                    fills += 1

        mtm = realized
        for entry, _ in open_longs.values():
            mtm += (close - entry) / entry - fee
        for entry, _ in open_shorts.values():
            mtm += (entry - close) / entry - fee
        pnl_path.append(mtm)
        max_open = max(max_open, len(open_longs) + len(open_shorts))

    exit_close = float(seg["close"].iloc[-1])
    forced = len(open_longs) + len(open_shorts)
    for entry, _ in open_longs.values():
        realized += _pnl_long(entry, exit_close, fee)
    for entry, _ in open_shorts.values():
        realized += _pnl_short(entry, exit_close, fee)

    capital_units = max(1, 2 * cfg.max_levels)
    pnl_per_capital = realized / capital_units
    max_drawdown = 0.0
    if pnl_path:
        arr = np.asarray(pnl_path, dtype=float) / capital_units
        running_max = np.maximum.accumulate(arr)
        max_drawdown = float((arr - running_max).min())

    return {
        "status": "ok",
        "start": seg.index[0],
        "end": seg.index[-1],
        "bars": len(seg),
        "center": center,
        "spacing_pct": spacing / center,
        "spacing_atr": spacing / atr,
        "fills": fills,
        "cycles": cycles,
        "forced_exits": forced,
        "max_open_levels": max_open,
        "gross_unit_pnl": realized,
        "pnl_per_capital": pnl_per_capital,
        "max_drawdown": max_drawdown,
    }


def build_features(
    symbol: str,
    bars: pd.DataFrame,
    cfg: GridConfig,
    *,
    bars_timeframe: str | None = None,
) -> pd.DataFrame:
    if cfg.feature_store_dir and str(cfg.feature_store_layer or "").strip():
        return _build_features_from_feature_store(
            symbol,
            bars,
            cfg,
            bars_timeframe=bars_timeframe or "",
        )
    if cfg.chop_signal == "ts_quantile":
        return _build_ts_quantile_features_from_feature_store(
            symbol,
            bars,
            cfg,
            bars_timeframe=bars_timeframe or "",
        )

    study_cfg = StudyConfig(
        box_window=cfg.box_window,
        chop_min=cfg.chop_min,
        stability_min=float(cfg.stability_min),
        width_min=float(cfg.width_min),
        width_max=float(cfg.width_max),
        touches_min=int(cfg.touches_min),
    )
    # Reuse CRF diagnostic features so box/chop definitions stay comparable.
    df = build_symbol_dataset(symbol, bars, study_cfg)
    df["atr14"] = _atr(df, 14)
    if "semantic_chop" not in df:
        df["bb_width_pctile"] = _bb_width_pctile(df["close"])
        df["semantic_chop"] = _semantic_chop(df["close"], df["bb_width_pctile"])
    if should_compute_semantic_chop_ts_q(cfg):
        raw_chop = pd.to_numeric(df["semantic_chop"], errors="coerce").to_numpy(
            dtype=float
        )
        df["semantic_chop_ts_q"] = semantic_chop_ts_quantile(
            raw_chop,
            df.index,
            window=int(cfg.chop_ts_window),
            min_periods=int(cfg.chop_ts_min_periods),
        )
    return df


def run_one_period(
    *,
    symbols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_dir: Path,
    timeframe: str,
    warmup_days: int,
    cfg: GridConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    segment_rows = []
    regime_rows = []
    warmup_start = start - pd.Timedelta(days=warmup_days)

    for symbol in symbols:
        raw = _load_symbol_1m(data_dir, symbol, warmup_start, end)
        if raw.empty:
            print(f"skip {symbol}: no data")
            continue
        bars = _resample_ohlcv(raw, timeframe)
        if bars.empty:
            print(f"skip {symbol}: no bars")
            continue
        df = build_features(symbol, bars, cfg, bars_timeframe=timeframe)
        df = df[(df.index >= start) & (df.index <= end)].copy()
        if df.empty:
            continue

        chop_s = regime_chop_series(df, cfg)
        chop = chop_s >= cfg.chop_min
        chop_hold = chop_s >= cfg.exit_chop_min
        rule_mask = apply_prefilter_rules(
            df,
            list(cfg.prefilter_rules),
            feature_aliases={
                "atr": "atr14",
                "bpc_semantic_chop": "semantic_chop",
                "bpc_semantic_chop_ts_q": "semantic_chop_ts_q",
            },
        )
        chop &= rule_mask
        chop_hold &= rule_mask
        box = df["box_prefilter"]
        chop_not_box = chop & ~box
        regimes = {
            "semantic_chop": (chop, chop_hold),
            "box_prefilter": (box, box),
            "chop_not_box": (chop_not_box, chop_hold & ~box),
        }

        for regime_name, (entry_mask, hold_mask) in regimes.items():
            segs = _hysteresis_segments(
                entry_mask,
                hold_mask,
                min_len=cfg.min_segment_bars,
                max_len=cfg.max_segment_bars,
            )
            regime_rows.append(
                {
                    "symbol": symbol,
                    "regime": regime_name,
                    "bars": int(entry_mask.sum()),
                    "bar_rate": float(entry_mask.mean()),
                    "hold_bars": int(hold_mask.sum()),
                    "hold_bar_rate": float(hold_mask.mean()),
                    "segments": len(segs),
                    "median_segment_bars": (
                        float(np.median([e - s + 1 for s, e in segs])) if segs else 0.0
                    ),
                }
            )
            for s, e in segs:
                seg = df.iloc[s : e + 1]
                sim = simulate_fixed_grid(seg, cfg=cfg)
                if sim.get("status") != "ok":
                    continue
                segment_rows.append(
                    {
                        "symbol": symbol,
                        "regime": regime_name,
                        "entry_chop": float(regime_chop_series(seg, cfg).iloc[0]),
                        "median_chop": float(regime_chop_series(seg, cfg).median()),
                        "entry_box_stability": float(seg["box_stability"].iloc[0]),
                        "entry_box_width_pct": float(seg["box_width_pct"].iloc[0]),
                        **sim,
                    }
                )
        print(
            f"{symbol}: chop={chop.mean():.1%}, box={box.mean():.1%}, "
            f"chop_not_box={chop_not_box.mean():.1%}"
        )

    return pd.DataFrame(segment_rows), pd.DataFrame(regime_rows)


def summarize_segments(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame()
    rows = []
    for (regime,), g in segments.groupby(["regime"], sort=True):
        rows.append(
            {
                "regime": regime,
                "segments": len(g),
                "win_rate": (g["pnl_per_capital"] > 0).mean(),
                "sum_pnl_per_capital": g["pnl_per_capital"].sum(),
                "mean_pnl_per_capital": g["pnl_per_capital"].mean(),
                "median_pnl_per_capital": g["pnl_per_capital"].median(),
                "p25_pnl": g["pnl_per_capital"].quantile(0.25),
                "p75_pnl": g["pnl_per_capital"].quantile(0.75),
                "median_bars": g["bars"].median(),
                "median_spacing_pct": g["spacing_pct"].median(),
                "median_cycles": g["cycles"].median(),
                "median_forced_exits": g["forced_exits"].median(),
                "worst_segment": g["pnl_per_capital"].min(),
                "median_max_drawdown": g["max_drawdown"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values("sum_pnl_per_capital", ascending=False)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--grid-config", default=str(DEFAULT_CHOP_GRID_YAML))
    pre_args, _ = pre.parse_known_args()
    grid_yaml = Path(pre_args.grid_config)
    if not grid_yaml.is_absolute():
        grid_yaml = PROJECT_ROOT / grid_yaml
    yd = merge_chop_grid_yaml(grid_yaml)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/parquet_data")
    parser.add_argument(
        "--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--warmup-days", type=int, default=120)
    parser.add_argument("--timeframe", default="2h")
    parser.add_argument(
        "--grid-config",
        default=str(grid_yaml),
        help="Chop grid strategy YAML (defaults for thresholds, box_prefilter, chop_series).",
    )
    parser.add_argument(
        "--box-window",
        type=int,
        default=int(yd.get("box_window", 120)),
        choices=[60, 120, 240],
    )
    parser.add_argument(
        "--chop-min", type=float, default=float(yd.get("chop_min", 0.40))
    )
    parser.add_argument(
        "--exit-chop-min", type=float, default=float(yd.get("exit_chop_min", 0.25))
    )
    parser.add_argument(
        "--min-segment-bars",
        type=int,
        default=int(yd.get("min_segment_bars", 6)),
    )
    parser.add_argument(
        "--max-segment-bars",
        type=int,
        default=int(yd.get("max_segment_bars", 120)),
    )
    parser.add_argument(
        "--grid-atr-mult", type=float, default=float(yd.get("grid_atr_mult", 0.75))
    )
    parser.add_argument(
        "--grid-pct", type=float, default=float(yd.get("grid_pct", 0.004))
    )
    parser.add_argument("--max-levels", type=int, default=int(yd.get("max_levels", 3)))
    parser.add_argument("--fee-bps", type=float, default=float(yd.get("fee_bps", 4.0)))
    parser.add_argument(
        "--chop-signal",
        choices=["raw", "ts_quantile"],
        default=str(yd.get("chop_signal", "raw")),
        help=(
            "Chop series for regime masks. ts_quantile reads semantic_chop + ts_q + box/atr strictly "
            "from FeatureStore (grid_backtest.feature_store_*); no in-process recomputation."
        ),
    )
    parser.add_argument(
        "--chop-ts-window",
        type=int,
        default=int(yd.get("chop_ts_window", 1200)),
    )
    parser.add_argument(
        "--chop-ts-min-periods",
        type=int,
        default=int(yd.get("chop_ts_min_periods", 150)),
    )
    parser.add_argument(
        "--compute-chop-ts-q",
        action=argparse.BooleanOptionalAction,
        default=yd.get("compute_chop_ts_q"),
        help="Override ts_q column build (default: from grid.yaml chop_series if set).",
    )
    parser.add_argument(
        "--stability-min",
        type=float,
        default=float(yd.get("stability_min", 0.85)),
    )
    parser.add_argument(
        "--width-min", type=float, default=float(yd.get("width_min", 0.04))
    )
    parser.add_argument(
        "--width-max", type=float, default=float(yd.get("width_max", 0.30))
    )
    parser.add_argument(
        "--touches-min", type=int, default=int(yd.get("touches_min", 5))
    )
    parser.add_argument(
        "--feature-store-dir",
        default=yd.get("feature_store_dir"),
        help="FeatureStore root; required indirectly when chop_signal=ts_quantile (strict parquet).",
    )
    parser.add_argument(
        "--feature-store-layer",
        default=yd.get("feature_store_layer"),
        help="FeatureStore layer (e.g. research layer used to materialize parquet).",
    )
    parser.add_argument(
        "--feature-store-timeframe",
        default=yd.get("feature_store_timeframe"),
        help="Optional FeatureStore timeframe if it differs from --timeframe (e.g. 120T vs 2h).",
    )
    parser.add_argument("--out-dir", default="results/chop_grid_diagnostic")
    args = parser.parse_args()

    cfg = GridConfig(
        box_window=args.box_window,
        chop_min=args.chop_min,
        exit_chop_min=args.exit_chop_min,
        min_segment_bars=args.min_segment_bars,
        max_segment_bars=args.max_segment_bars,
        grid_atr_mult=args.grid_atr_mult,
        grid_pct=args.grid_pct,
        max_levels=args.max_levels,
        fee_bps=args.fee_bps,
        chop_signal=args.chop_signal,
        chop_ts_window=args.chop_ts_window,
        chop_ts_min_periods=args.chop_ts_min_periods,
        compute_semantic_chop_ts_q=args.compute_chop_ts_q,
        stability_min=args.stability_min,
        width_min=args.width_min,
        width_max=args.width_max,
        touches_min=args.touches_min,
        feature_store_dir=resolve_optional_repo_path(args.feature_store_dir),
        feature_store_layer=(
            str(args.feature_store_layer).strip() if args.feature_store_layer else None
        )
        or None,
        feature_store_timeframe=(
            str(args.feature_store_timeframe).strip()
            if args.feature_store_timeframe
            else None
        )
        or None,
        prefilter_rules=tuple(
            x for x in (yd.get("prefilter_rules", []) or []) if isinstance(x, dict)
        ),
    )
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    segments, regimes = run_one_period(
        symbols=symbols,
        start=start,
        end=end,
        data_dir=Path(args.data_dir),
        timeframe=args.timeframe,
        warmup_days=args.warmup_days,
        cfg=cfg,
    )
    summary = summarize_segments(segments)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments.to_csv(out_dir / "grid_segments.csv", index=False)
    regimes.to_csv(out_dir / "regime_coverage.csv", index=False)
    summary.to_csv(out_dir / "grid_summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps({"args": vars(args)}, indent=2),
        encoding="utf-8",
    )

    print("\n=== Regime Coverage ===")
    if regimes.empty:
        print("(empty)")
    else:
        print(
            regimes.groupby("regime")
            .agg(
                bars=("bars", "sum"),
                mean_bar_rate=("bar_rate", "mean"),
                hold_bars=("hold_bars", "sum"),
                mean_hold_bar_rate=("hold_bar_rate", "mean"),
                segments=("segments", "sum"),
                median_segment_bars=("median_segment_bars", "median"),
            )
            .reset_index()
            .to_string(index=False)
        )
    print("\n=== Grid Summary ===")
    print(summary.to_string(index=False) if not summary.empty else "(empty)")
    print(f"\nSaved outputs -> {out_dir}")


if __name__ == "__main__":
    main()
