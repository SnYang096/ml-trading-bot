#!/usr/bin/env python3
"""
Generate FBF labels using a dual-window system:
- Execution window (dynamic): based on SL/TP/max_holding from execution_constraints
- Semantic window (fixed): fixed horizon to validate structure

Outputs:
  execution_label, semantic_label, fbf_label, label_consistency,
  exit_reason, holding_bars, execution_dir, semantic_dir

Near-miss samples (semantic=FBF, execution!=FBF) can be dumped.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec
from src.time_series_model.nnmultihead.strategy_profile import (
    load_execution_archetypes_registry,
)


@dataclass
class ExitInfo:
    exit_idx: int
    exit_reason: str
    holding_bars: int


def _parse_direction(row: pd.Series, side_col: str, dir_col: str) -> Optional[int]:
    if side_col in row and pd.notna(row[side_col]):
        side = str(row[side_col]).upper()
        if side in {"BUY", "LONG"}:
            return 1
        if side in {"SELL", "SHORT"}:
            return -1
    if dir_col in row and pd.notna(row[dir_col]):
        try:
            val = float(row[dir_col])
            if val > 0:
                return 1
            if val < 0:
                return -1
        except ValueError:
            return None
    return None


def _get_entry_price(
    row: pd.Series,
    entry_price_col: str,
    fallback_close: Optional[float],
) -> Optional[float]:
    if entry_price_col in row and pd.notna(row[entry_price_col]):
        return float(row[entry_price_col])
    if fallback_close is not None and pd.notna(fallback_close):
        return float(fallback_close)
    return None


def _read_feature_store_range(
    *,
    features_store_root: str,
    layer: str,
    symbol: str,
    timeframe: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    store = FeatureStore(str(features_store_root))
    spec = FeatureStoreSpec(
        layer=str(layer), symbol=str(symbol), timeframe=str(timeframe)
    )
    start_ts = start or pd.Timestamp("1970-01-01")
    end_ts = end or pd.Timestamp("2100-01-01")
    df = store.read_range(spec, start=start_ts, end=end_ts)
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        if getattr(df.index, "name", None) == "timestamp" or isinstance(
            df.index, pd.DatetimeIndex
        ):
            df = df.reset_index(drop=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _find_index(ts_array: np.ndarray, ts: np.datetime64) -> Optional[int]:
    idx = int(np.searchsorted(ts_array, ts))
    if idx < len(ts_array) and ts_array[idx] == ts:
        return idx
    if idx - 1 >= 0:
        return idx - 1
    return None


def _calc_exit_point(
    *,
    entry_idx: int,
    entry_price: float,
    direction: int,
    highs: np.ndarray,
    lows: np.ndarray,
    min_holding_bars: int,
    max_holding_bars: int,
    stop_loss_r: Optional[float],
    take_profit_r: Optional[float],
    atr_val: Optional[float],
) -> ExitInfo:
    end_idx = min(entry_idx + max_holding_bars, len(highs) - 1)
    sl_price = None
    tp_price = None
    if stop_loss_r is not None and take_profit_r is not None and atr_val is not None:
        sl_dist = stop_loss_r * atr_val
        tp_dist = take_profit_r * atr_val
        if direction == 1:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

    # Conservative ordering: stop-loss wins if both hit in same bar.
    start_idx = entry_idx + max(1, int(min_holding_bars or 0))
    if start_idx > end_idx:
        start_idx = end_idx
    for idx in range(start_idx, end_idx + 1):
        high = highs[idx]
        low = lows[idx]
        if sl_price is not None:
            if direction == 1 and low <= sl_price:
                return ExitInfo(
                    exit_idx=idx, exit_reason="stop_loss", holding_bars=idx - entry_idx
                )
            if direction == -1 and high >= sl_price:
                return ExitInfo(
                    exit_idx=idx, exit_reason="stop_loss", holding_bars=idx - entry_idx
                )
        if tp_price is not None:
            if direction == 1 and high >= tp_price:
                return ExitInfo(
                    exit_idx=idx,
                    exit_reason="take_profit",
                    holding_bars=idx - entry_idx,
                )
            if direction == -1 and low <= tp_price:
                return ExitInfo(
                    exit_idx=idx,
                    exit_reason="take_profit",
                    holding_bars=idx - entry_idx,
                )

    return ExitInfo(
        exit_idx=end_idx, exit_reason="max_holding", holding_bars=end_idx - entry_idx
    )


def _calc_mfe_mae(
    *,
    entry_idx: int,
    exit_idx: int,
    entry_price: float,
    direction: int,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> Dict[str, float]:
    slice_high = highs[entry_idx : exit_idx + 1]
    slice_low = lows[entry_idx : exit_idx + 1]
    if direction == 1:
        mfe = (np.max(slice_high) - entry_price) / entry_price
        mae = (np.min(slice_low) - entry_price) / entry_price
        final_return = (closes[exit_idx] - entry_price) / entry_price
    else:
        mfe = (entry_price - np.min(slice_low)) / entry_price
        mae = (entry_price - np.max(slice_high)) / entry_price
        final_return = (entry_price - closes[exit_idx]) / entry_price
    return {
        "mfe": float(mfe),
        "mae": float(mae),
        "final_return": float(final_return),
    }


def _execution_label(
    *,
    mfe: float,
    mae: float,
    holding_bars: int,
    exit_reason: str,
    final_return: float,
    max_holding_bars: int,
    exec_mfe_max: float,
    exec_mae_min: float,
    exec_holding_max: int,
) -> str:
    if holding_bars <= 0 or holding_bars > max_holding_bars:
        return "unlabeled"
    if (
        mfe <= exec_mfe_max
        and mae <= exec_mae_min
        and holding_bars <= exec_holding_max
        and exit_reason in {"stop_loss", "take_profit"}
    ):
        # execution label is outcome-driven; still require some adverse move
        return "FBF"
    if final_return == 0.0:
        return "unlabeled"
    return "unlabeled"


def _semantic_label(
    *,
    entry_idx: int,
    entry_price: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int,
    sem_extension_min: float,
    sem_failure_ratio_max: float,
    sem_regression_min: float,
) -> Tuple[str, Optional[int]]:
    end_idx = min(entry_idx + window, len(highs) - 1)
    if end_idx <= entry_idx + 3:
        return "unlabeled", None

    total_len = end_idx - entry_idx
    first_end = entry_idx + max(1, int(total_len * 0.3))
    mid_end = entry_idx + max(2, int(total_len * 0.7))

    first_high = np.max(highs[entry_idx : first_end + 1])
    first_low = np.min(lows[entry_idx : first_end + 1])
    up_move = (first_high - entry_price) / entry_price
    down_move = (entry_price - first_low) / entry_price

    initial_dir = 1 if up_move >= down_move else -1
    initial_move = up_move if initial_dir == 1 else down_move

    if initial_move < sem_extension_min:
        return "unlabeled", None

    mid_high = np.max(highs[first_end : mid_end + 1])
    mid_low = np.min(lows[first_end : mid_end + 1])
    mid_extension = (
        (mid_high - entry_price) / entry_price
        if initial_dir == 1
        else (entry_price - mid_low) / entry_price
    )
    failure_ratio = mid_extension / max(initial_move, 1e-9)
    if failure_ratio > sem_failure_ratio_max:
        return "unlabeled", None

    last_high = np.max(highs[mid_end : end_idx + 1])
    last_low = np.min(lows[mid_end : end_idx + 1])
    regression_move = (
        (entry_price - last_low) / entry_price
        if initial_dir == 1
        else (last_high - entry_price) / entry_price
    )
    regression_ratio = regression_move / max(initial_move, 1e-9)
    last_close = closes[end_idx]
    close_back_in_range = (
        last_close <= entry_price if initial_dir == 1 else last_close >= entry_price
    )

    if regression_ratio >= sem_regression_min and close_back_in_range:
        # FBF semantic: reversal of initial breakout
        semantic_dir = -initial_dir
        return "FBF", semantic_dir

    return "unlabeled", None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate FBF labels (dual-window)")
    parser.add_argument("--logs", required=True, help="logs_execution.parquet path")
    parser.add_argument("--out", required=True, help="output parquet path")
    parser.add_argument("--near-miss-out", default=None, help="near-miss output path")
    parser.add_argument(
        "--archetype", default="FailedBreakoutFade", help="archetype name"
    )
    parser.add_argument("--feature-store-root", default=None, help="FeatureStore root")
    parser.add_argument(
        "--feature-store-layer", default=None, help="FeatureStore layer"
    )
    parser.add_argument("--timeframe", default=None, help="timeframe for FeatureStore")
    parser.add_argument("--side-col", default="side", help="side column (BUY/SELL)")
    parser.add_argument(
        "--direction-col", default="direction", help="direction column (numeric)"
    )
    parser.add_argument(
        "--entry-price-col", default="entry_price", help="entry price column"
    )
    parser.add_argument("--atr-col", default="atr", help="ATR column for SL/TP")
    parser.add_argument(
        "--semantic-window", type=int, default=24, help="fixed semantic window bars"
    )
    parser.add_argument(
        "--exec-mfe-max", type=float, default=0.005, help="execution MFE max"
    )
    parser.add_argument(
        "--exec-mae-min", type=float, default=-0.01, help="execution MAE min"
    )
    parser.add_argument(
        "--exec-holding-max", type=int, default=12, help="execution holding bars max"
    )
    parser.add_argument(
        "--sem-extension-min",
        type=float,
        default=0.003,
        help="semantic initial extension min (pct)",
    )
    parser.add_argument(
        "--sem-failure-ratio-max",
        type=float,
        default=0.3,
        help="semantic failure ratio max",
    )
    parser.add_argument(
        "--sem-regression-min",
        type=float,
        default=0.5,
        help="semantic regression ratio min",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.logs)
    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=False)
        else:
            raise KeyError("logs must include a timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    arches = load_execution_archetypes_registry(
        path=str(PROJECT_ROOT / "config/nnmultihead/execution_archetypes.yaml")
    )
    arch = arches.get(args.archetype)
    if not arch:
        raise KeyError(f"archetype not found: {args.archetype}")

    exec_constraints = getattr(arch, "execution_constraints", {}) or {}
    fixed_rr = (
        exec_constraints.get("fixed_rr", {})
        if isinstance(exec_constraints, dict)
        else {}
    )
    max_holding_bars = int(fixed_rr.get("max_holding_bars", args.semantic_window))
    min_holding_bars = int(fixed_rr.get("min_holding_bars", 0))
    if min_holding_bars > max_holding_bars:
        min_holding_bars = max_holding_bars
    stop_loss_r = fixed_rr.get("stop_loss_r")
    take_profit_r = fixed_rr.get("take_profit_r")

    if not (args.feature_store_root and args.feature_store_layer and args.timeframe):
        raise ValueError(
            "feature-store-root/layer/timeframe are required for price series"
        )

    out_rows = []
    near_miss_rows = []

    for symbol, df_sym in df.groupby("symbol", sort=False):
        df_sym = df_sym.sort_values("timestamp")
        price_df = _read_feature_store_range(
            features_store_root=args.feature_store_root,
            layer=args.feature_store_layer,
            symbol=symbol,
            timeframe=args.timeframe,
            start=df_sym["timestamp"].min(),
            end=df_sym["timestamp"].max(),
        )
        if price_df.empty:
            continue
        if not {"high", "low", "close"}.issubset(set(price_df.columns)):
            raise KeyError("price data must include high/low/close columns")

        price_df = price_df.sort_values("timestamp")
        ts_array = price_df["timestamp"].to_numpy()
        highs = price_df["high"].to_numpy(dtype=float)
        lows = price_df["low"].to_numpy(dtype=float)
        closes = price_df["close"].to_numpy(dtype=float)
        atr_series = (
            price_df[args.atr_col].to_numpy(dtype=float)
            if args.atr_col in price_df.columns
            else None
        )

        for _, row in df_sym.iterrows():
            ts = pd.Timestamp(row["timestamp"]).to_datetime64()
            entry_idx = _find_index(ts_array, ts)
            if entry_idx is None:
                continue

            direction = _parse_direction(row, args.side_col, args.direction_col)
            if direction is None:
                # Try to infer from ret_mean sign if available
                if "ret_mean" in row and pd.notna(row["ret_mean"]):
                    direction = 1 if row["ret_mean"] < 0 else -1  # FR is mean reversion
                else:
                    continue

            entry_close = closes[entry_idx] if entry_idx < len(closes) else None
            entry_price = _get_entry_price(row, args.entry_price_col, entry_close)
            if entry_price is None:
                continue

            atr_val = float(atr_series[entry_idx]) if atr_series is not None else None
            exit_info = _calc_exit_point(
                entry_idx=entry_idx,
                entry_price=entry_price,
                direction=direction,
                highs=highs,
                lows=lows,
                min_holding_bars=min_holding_bars,
                max_holding_bars=max_holding_bars,
                stop_loss_r=stop_loss_r,
                take_profit_r=take_profit_r,
                atr_val=atr_val,
            )
            mfe_mae = _calc_mfe_mae(
                entry_idx=entry_idx,
                exit_idx=exit_info.exit_idx,
                entry_price=entry_price,
                direction=direction,
                highs=highs,
                lows=lows,
                closes=closes,
            )
            exec_label = _execution_label(
                mfe=mfe_mae["mfe"],
                mae=mfe_mae["mae"],
                holding_bars=exit_info.holding_bars,
                exit_reason=exit_info.exit_reason,
                final_return=mfe_mae["final_return"],
                max_holding_bars=max_holding_bars,
                exec_mfe_max=args.exec_mfe_max,
                exec_mae_min=args.exec_mae_min,
                exec_holding_max=args.exec_holding_max,
            )
            sem_label, semantic_dir = _semantic_label(
                entry_idx=entry_idx,
                entry_price=entry_price,
                highs=highs,
                lows=lows,
                closes=closes,
                window=args.semantic_window,
                sem_extension_min=args.sem_extension_min,
                sem_failure_ratio_max=args.sem_failure_ratio_max,
                sem_regression_min=args.sem_regression_min,
            )

            if exec_label == sem_label:
                final_label = exec_label
            elif exec_label == "unlabeled" and sem_label != "unlabeled":
                final_label = sem_label
            elif sem_label == "unlabeled" and exec_label != "unlabeled":
                final_label = exec_label
            else:
                final_label = "unlabeled"

            execution_match = exec_label == "FBF"
            structure_match = sem_label == "FBF"
            direction_match = semantic_dir is not None and direction == semantic_dir

            row_out = row.to_dict()
            row_out.update(
                {
                    "execution_label": exec_label,
                    "semantic_label": sem_label,
                    "fbf_label": final_label,
                    "label_consistency": exec_label == sem_label,
                    "structure_match": structure_match,
                    "execution_match": execution_match,
                    "direction_match": direction_match,
                    "execution_dir": direction,
                    "semantic_dir": semantic_dir,
                    "exit_reason": exit_info.exit_reason,
                    "holding_bars": exit_info.holding_bars,
                    "execution_mfe": mfe_mae["mfe"],
                    "execution_mae": mfe_mae["mae"],
                    "execution_return": mfe_mae["final_return"],
                }
            )
            out_rows.append(row_out)

            if sem_label == "FBF" and exec_label != "FBF":
                near_miss_rows.append(row_out)

    print(f"📊 Generated {len(out_rows)} labeled rows")
    if not out_rows:
        print("⚠️  No labels generated! Check input data and feature store.")
        return 1
    out_df = pd.DataFrame(out_rows)
    out_df.to_parquet(args.out, index=False)
    print(f"✅ Saved labels to {args.out}")

    near_miss_path = args.near_miss_out
    if near_miss_path is None:
        near_miss_path = (
            str(Path(args.out).with_suffix("").as_posix()) + "_near_miss.parquet"
        )
    if near_miss_rows:
        pd.DataFrame(near_miss_rows).to_parquet(near_miss_path, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
