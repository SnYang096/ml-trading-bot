#!/usr/bin/env python3
from __future__ import annotations

"""
Quick diagnostic script for SR Reversal:

- Runs the existing feature pipeline for a given strategy (default: sr_reversal)
- Auto-generates SR reversal signals using the same heuristics as label generation
- Applies a pure rule-based RR strategy:
    * Entry: every non-zero SR signal (no ML model, no pred thresholds)
    * Exit: 1R stop loss / 2R take profit / max_holding_bars, via compute_rr_label logic
- Outputs:
    * Basic stats: number of signals, valid trades, win rate, average R, total R
    * Optional CSV with per-signal labels for further analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional
import json

import numpy as np
import pandas as pd

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_strategy_pipeline as strategy_runner  # noqa: E402
from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.strategy_config import StrategyConfigLoader  # noqa: E402
from src.time_series_model.pipeline.training.label_utils import (  # noqa: E402
    compute_rr_label,
    compute_adaptive_rr_label_with_future_vol,
    compute_rr_label_with_details,
)
from src.time_series_model.strategies.labels.sr_reversal_label import (  # noqa: E402
    SRSignalConfig,
    _ensure_atr,
    _generate_sr_reversal_signals,
)
from src.data_tools.tick_loader import build_tick_loader_payload  # noqa: E402


def _timeframe_to_minutes(tf: str) -> Optional[int]:
    tf = (tf or "").strip().upper()
    if tf.endswith("T"):
        try:
            return int(float(tf[:-1]))
        except ValueError:
            return None
    if tf.endswith("H"):
        try:
            return int(float(tf[:-1]) * 60)
        except ValueError:
            return None
    if tf.endswith("D"):
        try:
            return int(float(tf[:-1]) * 1440)
        except ValueError:
            return None
    if tf.isdigit():
        return int(tf)
    return None


def _should_use_tick_data(mode: str, timeframe: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SR Reversal rule-based RR baseline (no ML, SR signal + RR exit only)."
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        default="config/strategies/sr_reversal",
        help="Path to SR reversal strategy config directory.",
    )
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="data/parquet_data")
    parser.add_argument("--timeframe", type=str, default="240T")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--max-holding-bars", type=int, default=50)
    parser.add_argument("--stop-loss-r", type=float, default=1.0)
    parser.add_argument("--take-profit-r", type=float, default=2.0)
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument(
        "--entry-offset",
        type=int,
        default=1,
        help="Bars after signal to enter (1 = next bar open).",
    )
    parser.add_argument(
        "--entry-price-col",
        type=str,
        default="open",
        help="Column to use as entry price (default: open).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/sr_reversal_rule_baseline.csv",
        help="Path to save per-signal diagnostics CSV.",
    )
    parser.add_argument(
        "--use-adaptive-rr",
        action="store_true",
        help="Use adaptive R/R based on future volatility (for label generation only).",
    )
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=10,
        help="Window size for calculating future volatility (default: 10).",
    )
    parser.add_argument(
        "--tick-data-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Whether to load real tick data for VPIN (auto enables for <=120-minute timeframes).",
    )
    parser.add_argument(
        "--ticks-dir",
        type=str,
        default="data/parquet_data",
        help="Directory containing monthly tick parquet files (after conversion).",
    )
    parser.add_argument(
        "--ticks-lookback-minutes",
        type=int,
        default=60,
        help="Extra minutes of tick history to load before/after the data window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_dir = Path(args.strategy_config).resolve()
    loader = StrategyConfigLoader(cfg_dir)
    strategy_cfg = loader.load()

    print(f"📂 Strategy config: {cfg_dir}")
    print(f"📈 Loading data for {args.symbol} [{args.timeframe}] from {args.data_path}")

    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )
    print(f"   ✅ Loaded {len(df_raw)} bars.")

    use_tick_data = _should_use_tick_data(args.tick_data_mode, args.timeframe)
    tick_loader_json: Optional[str] = None
    if use_tick_data:
        if df_raw.empty:
            raise ValueError("No bars available for tick-loader configuration.")
        print("   📦 Tick data mode enabled for VPIN (mode=%s)" % args.tick_data_mode)
        tick_loader_json = build_tick_loader_payload(
            symbol=args.symbol.upper(),
            start_ts=df_raw.index.min().isoformat(),
            end_ts=df_raw.index.max().isoformat(),
            ticks_dir=args.ticks_dir,
            lookback_minutes=args.ticks_lookback_minutes,
        )
    else:
        print("   ℹ️ Tick data mode disabled (mode=%s)" % args.tick_data_mode)

    feature_loader = StrategyFeatureLoader()
    if tick_loader_json:
        vpin_feature = feature_loader.feature_deps.get("features", {}).get(
            "vpin_features"
        )
        if vpin_feature is not None:
            vpin_feature.setdefault("compute_params", {})[
                "ticks_loader_json"
            ] = tick_loader_json
        else:
            print(
                "   ⚠️  vpin_features missing in feature dependencies; cannot pass tick loader."
            )
    df_features = strategy_runner.run_feature_pipeline(
        df_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )

    # Ensure ATR and SR reversal signals are available, using same logic as label generator
    atr_series = _ensure_atr(
        df_features,
        atr_col="atr",
        price_col="close",
        high_col="high",
        low_col="low",
        atr_window=args.atr_window,
    )
    df_features["atr"] = atr_series

    # 配置 SR 信号生成，启用 VPIN 过滤
    sr_cfg = SRSignalConfig(
        use_vpin_filter=True,  # 启用 VPIN 过滤
        min_vpin=0.4,  # 多头信号：VPIN >= 0.4（买压较大，适合反转做多）
        max_vpin=0.6,  # 空头信号：VPIN <= 0.6（卖压较大，适合反转做空）
    )
    auto_signals = _generate_sr_reversal_signals(
        df_features,
        price_col="close",
        high_col="high",
        low_col="low",
        atr_series=atr_series,
        cfg=sr_cfg,
    )
    df_features["signal"] = auto_signals

    n_signals_total = int((auto_signals != 0).sum())
    print(f"   ✅ Generated SR signals: {n_signals_total} (non-zero entries)")

    # Compute RR labels using exactly the same R/R logic as training
    if args.use_adaptive_rr:
        # 自适应版本：基于未来波动率的动态 R/R
        print("   📊 Using adaptive R/R based on future volatility...")
        labels_standard = compute_adaptive_rr_label_with_future_vol(
            df_features.copy(),
            signal_col="signal",
            price_col="close",
            atr_col="atr",
            atr_window=args.atr_window,
            max_holding_bars=args.max_holding_bars,
            stop_loss_multiplier=args.stop_loss_r,
            take_profit_multiplier=args.take_profit_r,
            volatility_window=args.volatility_window,
            use_breakeven_stop=False,
            entry_price_col=args.entry_price_col,
            entry_offset=args.entry_offset,
        )

        labels_breakeven = compute_adaptive_rr_label_with_future_vol(
            df_features.copy(),
            signal_col="signal",
            price_col="close",
            atr_col="atr",
            atr_window=args.atr_window,
            max_holding_bars=args.max_holding_bars,
            stop_loss_multiplier=args.stop_loss_r,
            take_profit_multiplier=args.take_profit_r,
            volatility_window=args.volatility_window,
            use_breakeven_stop=True,
            entry_price_col=args.entry_price_col,
            entry_offset=args.entry_offset,
        )
    else:
        # 标准版本：固定 R/R（基于 ATR）
        labels_standard = compute_rr_label(
            df_features.copy(),
            signal_col="signal",
            price_col="close",
            atr_col="atr",
            atr_window=args.atr_window,
            rr_ratio=args.take_profit_r,
            max_holding_bars=args.max_holding_bars,
            stop_loss_r=args.stop_loss_r,
            take_profit_r=args.take_profit_r,
            use_continuous_label=False,
            entry_price_col=args.entry_price_col,
            entry_offset=args.entry_offset,
            use_breakeven_stop=False,  # 标准版本：不使用保本止损
        )

        # 保本版本：1R 上移到保本，2R 止盈（使用详细信息版本）
        details_breakeven = compute_rr_label_with_details(
            df_features.copy(),
            signal_col="signal",
            price_col="close",
            atr_col="atr",
            atr_window=args.atr_window,
            rr_ratio=args.take_profit_r,
            max_holding_bars=args.max_holding_bars,
            stop_loss_r=args.stop_loss_r,
            take_profit_r=args.take_profit_r,
            use_continuous_label=False,
            entry_price_col=args.entry_price_col,
            entry_offset=args.entry_offset,
            use_breakeven_stop=True,  # 保本版本：使用保本止损
        )
        labels_breakeven = details_breakeven["label"]

    # Only consider bars where we had a signal and a valid label
    mask_valid_standard = (auto_signals != 0) & labels_standard.notna()
    mask_valid_breakeven = (auto_signals != 0) & labels_breakeven.notna()

    df_trades_standard = pd.DataFrame(
        {
            "signal": auto_signals[mask_valid_standard],
            "label": labels_standard[mask_valid_standard],
        }
    )

    df_trades_breakeven = pd.DataFrame(
        {
            "signal": auto_signals[mask_valid_breakeven],
            "label": labels_breakeven[mask_valid_breakeven],
        }
    )

    # 添加详细信息到保本版本
    if not args.use_adaptive_rr:
        details_breakeven_valid = details_breakeven[mask_valid_breakeven]
        df_trades_breakeven = pd.concat(
            [
                df_trades_breakeven,
                details_breakeven_valid[
                    ["breakeven_activated", "hit_tp", "hit_sl", "final_result"]
                ],
            ],
            axis=1,
        )

    n_trades_standard = len(df_trades_standard)
    n_trades_breakeven = len(df_trades_breakeven)

    if n_trades_standard == 0:
        print(
            "   ⚠️ No valid SR reversal trades found with current heuristics + RR params."
        )
        return

    # 标准版本统计
    n_win_standard = int((df_trades_standard["label"] == 1.0).sum())
    win_rate_standard = (
        n_win_standard / n_trades_standard if n_trades_standard > 0 else 0.0
    )

    tp_r = float(args.take_profit_r)
    sl_r = float(args.stop_loss_r)
    realized_r_standard = np.where(
        df_trades_standard["label"].values == 1.0, tp_r, -sl_r
    )
    avg_r_standard = float(realized_r_standard.mean())
    total_r_standard = float(realized_r_standard.sum())

    # 保本版本统计
    n_win_breakeven = int((df_trades_breakeven["label"] == 1.0).sum())
    win_rate_breakeven = (
        n_win_breakeven / n_trades_breakeven if n_trades_breakeven > 0 else 0.0
    )

    # 计算保本+胜利的比例
    if "final_result" in df_trades_breakeven.columns:
        n_breakeven_win = int(
            (df_trades_breakeven["final_result"] == "breakeven_win").sum()
        )
        n_breakeven_loss = int(
            (df_trades_breakeven["final_result"] == "breakeven_loss").sum()
        )
        n_breakeven_total = n_breakeven_win + n_breakeven_loss
        breakeven_win_rate = (
            n_breakeven_win / n_breakeven_total if n_breakeven_total > 0 else 0.0
        )
        breakeven_ratio = (
            n_breakeven_total / n_trades_breakeven if n_trades_breakeven > 0 else 0.0
        )
    else:
        n_breakeven_win = 0
        n_breakeven_loss = 0
        n_breakeven_total = 0
        breakeven_win_rate = 0.0
        breakeven_ratio = 0.0

    # 保本版本：盈利时 +2R，亏损时 0（因为止损在保本）
    realized_r_breakeven = np.where(
        df_trades_breakeven["label"].values == 1.0,
        tp_r,  # 盈利：+2R
        0.0,  # 亏损：0（保本止损）
    )
    avg_r_breakeven = float(realized_r_breakeven.mean())
    total_r_breakeven = float(realized_r_breakeven.sum())

    print("\n📊 SR Reversal Rule-Based RR Baseline (no ML)")
    print("\n【标准版本：1R 止损，2R 止盈】")
    print(f"   Trades          : {n_trades_standard}")
    print(f"   Wins            : {n_win_standard}")
    print(f"   Win rate        : {win_rate_standard:.2%}")
    print(f"   Avg R per trade : {avg_r_standard:.3f}")
    print(f"   Total R         : {total_r_standard:.3f}")

    print("\n【保本版本：1R 上移到保本，2R 止盈】")
    print(f"   Trades          : {n_trades_breakeven}")
    print(f"   Wins            : {n_win_breakeven}")
    print(f"   Win rate        : {win_rate_breakeven:.2%}")
    if n_breakeven_total > 0:
        print(f"   保本触发次数    : {n_breakeven_total} ({breakeven_ratio:.2%})")
        print(f"   保本+胜利       : {n_breakeven_win} ({breakeven_win_rate:.2%})")
        print(f"   保本+亏损       : {n_breakeven_loss}")
    print(f"   Avg R per trade : {avg_r_breakeven:.3f}")
    print(f"   Total R         : {total_r_breakeven:.3f}")

    # Save diagnostics CSV for further analysis
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_trades_out = df_trades_standard.copy()
    df_trades_out["realized_r_standard"] = realized_r_standard
    df_trades_out["label_breakeven"] = (
        labels_breakeven[mask_valid_standard]
        if len(labels_breakeven[mask_valid_standard]) == len(df_trades_out)
        else np.nan
    )
    df_trades_out["realized_r_breakeven"] = (
        realized_r_breakeven[: len(df_trades_out)]
        if len(realized_r_breakeven) >= len(df_trades_out)
        else np.nan
    )
    df_trades_out.to_csv(out_path, index_label="index")
    print(f"\n   💾 Saved per-trade diagnostics to {out_path}")


if __name__ == "__main__":
    main()
