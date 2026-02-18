#!/usr/bin/env python3
"""
Train cross-sectional models (boosting or Fama-MacBeth) on multi-asset factor panels.

Example:
    python scripts/cross_sectional/train_cross_sectional_model.py \
        --input "results/training/*/features/*.parquet" \
        --symbols "BTCUSDT,ETHUSDT,SOLUSDT" \
        --horizon 12 \
        --model boosting \
        --output-dir results/cross_sectional/models/demo
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import dump

from cross_sectional import (
    FactorPanelBuilder,
    PanelConfig,
    CrossSectionalBoostingModel,
    CrossSectionalRegressor,
    ReportContext,
    add_crypto_cross_sectional_factors,
    cross_sectional_zscore,
    winsorize_by_sigma,
    generate_markdown_report,
    write_report,
)
from cross_sectional.factor_selection import (
    apply_factor_selection,
    compute_cross_sectional_ic,
)
from cross_sectional.model_portfolio_backtest import (
    PortfolioBacktestConfig,
    portfolio_backtest_from_signal,
    portfolio_backtest_with_rebalance_log,
)


def _filter_numeric_factor_cols(
    panel: pd.DataFrame, factor_cols: List[str], target_col: str
) -> List[str]:
    cols: List[str] = []
    dropped: List[str] = []
    for c in factor_cols:
        if c == target_col:
            continue
        if c not in panel.columns:
            continue
        if not pd.api.types.is_numeric_dtype(panel[c]):
            dropped.append(c)
            continue
        cols.append(c)
    if dropped:
        print(
            f"⚠️  Dropped {len(dropped)} non-numeric factor columns (e.g. {dropped[:5]})"
        )
    return cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train cross-sectional models on multi-asset factor panels."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Parquet/CSV files or glob patterns containing engineered features (must include timestamp & symbol).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cross_sectional/models",
        help="Directory to save model artefacts and diagnostics.",
    )
    parser.add_argument(
        "--model",
        choices=["boosting", "fama_macbeth"],
        default="boosting",
        help="Cross-sectional model type (default: boosting).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to keep. Default: use all symbols present.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forward return horizon in bars used as the target (future_return_{horizon}).",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Optional comma-separated feature list. If omitted, numeric columns excluding OHLCV/labels are used.",
    )
    parser.add_argument(
        "--feature-file",
        type=str,
        default=None,
        help="Path to text file containing feature names (one per line). Overrides --feature-cols.",
    )
    parser.add_argument(
        "--winsor",
        type=float,
        default=3.0,
        help="Sigma threshold for cross-sectional winsorisation (<=0 disables).",
    )
    parser.add_argument(
        "--zscore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply cross-sectional z-scoring per timestamp.",
    )
    parser.add_argument(
        "--crypto-factors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augment panel with crypto-specific cross-sectional factors.",
    )
    parser.add_argument(
        "--periods-per-year",
        type=str,
        default="auto",
        help="Annualisation factor for report metrics (e.g., 17520 for 5-min bars). Use 'auto' to infer from data.",
    )
    parser.add_argument(
        "--backtest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run an OOS portfolio backtest on model predictions (with lag/holding/constraints).",
    )
    parser.add_argument(
        "--bt-fee-bps",
        type=float,
        default=2.0,
        help="Fee (bps) applied to turnover in model prediction backtest.",
    )
    parser.add_argument(
        "--bt-slippage-bps",
        type=float,
        default=0.0,
        help="Slippage (bps) applied to turnover in model prediction backtest.",
    )
    parser.add_argument(
        "--bt-min-assets",
        type=int,
        default=12,
        help="Minimum assets per timestamp for model prediction backtest.",
    )
    parser.add_argument(
        "--bt-mode",
        type=str,
        default="long_only,market_neutral",
        help="Comma-separated backtest modes: long_only,market_neutral",
    )
    parser.add_argument(
        "--bt-holding",
        type=int,
        default=None,
        help="Holding period in bars (default: horizon).",
    )
    parser.add_argument(
        "--bt-lag",
        type=int,
        default=1,
        help="Execution lag in bars (default: 1).",
    )
    parser.add_argument(
        "--bt-topk",
        type=int,
        default=10,
        help="Top-K selection size for long leg.",
    )
    parser.add_argument(
        "--bt-bottomk",
        type=int,
        default=10,
        help="Bottom-K selection size for short leg (market-neutral).",
    )
    parser.add_argument(
        "--bt-gross-leverage",
        type=float,
        default=1.0,
        help="Gross leverage cap (sum abs weights).",
    )
    parser.add_argument(
        "--bt-max-weight",
        type=float,
        default=0.10,
        help="Max absolute weight per asset.",
    )
    parser.add_argument(
        "--bt-turnover-limit",
        type=float,
        default=None,
        help="Optional turnover limit per rebalance (e.g., 0.5).",
    )
    parser.add_argument(
        "--bt-cash-buffer",
        type=float,
        default=0.0,
        help="Cash buffer fraction (0..1). Uninvested capital.",
    )
    parser.add_argument(
        "--bt-equity-mode",
        type=str,
        default="compound",
        help="Equity curve mode: simple|compound|log",
    )
    parser.add_argument(
        "--bt-funding-bps-per-bar",
        type=float,
        default=0.0,
        help="Funding cost (bps per bar) applied to short exposure (market-neutral).",
    )
    parser.add_argument(
        "--bt-borrow-bps-per-bar",
        type=float,
        default=0.0,
        help="Borrow cost (bps per bar) applied to short exposure (market-neutral).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="OOS split: fraction of timestamps used for training (rest used for test backtest).",
    )
    parser.add_argument(
        "--wf-folds",
        type=int,
        default=1,
        help="Walk-forward folds (>1 enables expanding walk-forward with multiple test segments).",
    )
    parser.add_argument(
        "--wf-embargo-bars",
        type=int,
        default=None,
        help="Embargo bars between train and test to avoid label overlap (default: horizon + bt_lag).",
    )
    parser.add_argument(
        "--save-markdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a Markdown diagnostics report alongside metrics.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cs_boosting.joblib",
        help="File name for saved model (if boosting).",
    )
    parser.add_argument(
        "--predictions-name",
        type=str,
        default="predictions.parquet",
        help="File name for saved predictions (MultiIndex parquet).",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default="metrics.json",
        help="File name for saved evaluation metrics.",
    )
    parser.add_argument(
        "--auto-select",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically score factors via IC/IR and keep the best subset.",
    )
    parser.add_argument(
        "--select-topk",
        type=int,
        default=0,
        help="Keep only the top-K factors ranked by the chosen statistic.",
    )
    parser.add_argument(
        "--ic-threshold",
        type=float,
        default=None,
        help="Minimum absolute IC mean required to retain a factor.",
    )
    parser.add_argument(
        "--ir-threshold",
        type=float,
        default=None,
        help="Minimum absolute IC IR required to retain a factor.",
    )
    parser.add_argument(
        "--selection-stat",
        choices=["ic", "ir"],
        default="ic",
        help="Statistic used to rank factors when selecting top-K (default: ic).",
    )
    parser.add_argument(
        "--selection-output",
        type=str,
        default="selection_metrics.json",
        help="File name for saving IC/IR selection diagnostics (relative to output dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = collect_inputs(args.input)
    raw_df = load_frames(input_paths)
    print(f"📥 Loaded {len(input_paths)} files -> raw shape {raw_df.shape}")
    filtered_df = filter_symbols(raw_df, args.symbols)
    print(f"📑 After symbol filter: {filtered_df.shape}")

    if filtered_df.empty:
        raise ValueError("No data available after applying symbol filters.")

    if isinstance(filtered_df.index, pd.MultiIndex):
        cols_to_drop = [
            col for col in ["timestamp", "symbol"] if col in filtered_df.columns
        ]
        if cols_to_drop:
            filtered_df = filtered_df.drop(columns=cols_to_drop)
        filtered_df = filtered_df.reset_index()

    filtered_df, target_col = ensure_future_return_column(filtered_df, args.horizon)

    feature_cols = None
    if args.feature_file:
        feature_path = Path(args.feature_file)
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        feature_cols = [
            line.strip()
            for line in feature_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"📄 Loaded {len(feature_cols)} features from {feature_path}")
    elif args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    if feature_cols:
        feature_cols = list(dict.fromkeys(feature_cols))
        available = set(filtered_df.columns)
        missing = [c for c in feature_cols if c not in available]
        if missing:
            print(
                f"⚠️  Warning: {len(missing)} requested features not present: {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}"
            )
        feature_cols = [c for c in feature_cols if c in available]
        if not feature_cols:
            raise ValueError(
                "No valid features remaining after filtering against dataframe columns."
            )

    symbol_count = filtered_df["symbol"].nunique()
    min_assets_required = 3
    if symbol_count < min_assets_required:
        min_assets_required = max(1, symbol_count)

    panel, base_features = build_panel(
        filtered_df,
        target_col=target_col,
        feature_cols=feature_cols,
        min_assets=min_assets_required,
        horizon=args.horizon,
    )
    factor_cols = list(feature_cols) if feature_cols else list(base_features)
    print(f"📦 Initial factor count: {len(factor_cols)}")

    if args.crypto_factors:
        panel = add_crypto_cross_sectional_factors(panel)
        crypto_cols = [
            col
            for col in panel.columns
            if col.startswith("cs_crypto_")
            and col not in factor_cols
            and col != target_col
        ]
        factor_cols.extend(crypto_cols)
    print(f"📦 Factor count after augmentation: {len(factor_cols)}")

    factor_cols = _filter_numeric_factor_cols(panel, factor_cols, target_col)

    processed_panel = preprocess_panel(panel, factor_cols, args.winsor, args.zscore)
    periods_per_year = resolve_periods_per_year(
        args.periods_per_year, processed_panel.index
    )
    print(
        f"📊 Panel ready: {processed_panel.shape[0]} observations, "
        f"{len(factor_cols)} factors, periods_per_year={periods_per_year:.2f}"
    )

    selection_metrics: Optional[pd.DataFrame] = None
    if (
        args.auto_select
        or (args.select_topk and args.select_topk > 0)
        or args.ic_threshold is not None
        or args.ir_threshold is not None
    ):
        print("🔍 Running factor selection (IC / IR scoring)...")
        selection_metrics = compute_cross_sectional_ic(
            processed_panel, factor_cols, target_col
        )
        if selection_metrics.empty:
            print(
                "   ⚠️  Unable to compute IC/IR metrics; retaining original factor set."
            )
        else:
            factor_cols = apply_factor_selection(
                selection_metrics,
                factor_cols,
                select_topk=args.select_topk,
                ic_threshold=args.ic_threshold,
                ir_threshold=args.ir_threshold,
                ranking_stat=args.selection_stat,
            )
            if not factor_cols:
                print(
                    "   ⚠️  Selection filters removed all factors; reverting to original list."
                )
                factor_cols = list(selection_metrics.index)
            else:
                print(
                    f"   ✅ Selected {len(factor_cols)} factors after applying IC/IR criteria."
                )
                preview = (
                    selection_metrics.loc[factor_cols]
                    .sort_values(by="ic_mean", ascending=False)
                    .head(min(10, len(factor_cols)))
                )
                for factor, row in preview.iterrows():
                    print(
                        f"      {factor}: ic_mean={row['ic_mean']:.4f}, ic_ir={row['ic_ir']:.4f}, count={int(row['ic_count'])}"
                    )

    if args.model == "boosting":

        def _factor_combo_signal(p: pd.DataFrame, cols: List[str]) -> pd.Series:
            if not cols:
                return pd.Series(index=p.index, dtype=float)
            df = p[cols].copy().replace([np.inf, -np.inf], np.nan)
            grp = df.groupby(level=0)
            z = grp.transform(
                lambda x: (x - x.mean())
                / (x.std(ddof=0) if float(x.std(ddof=0)) > 0 else 1.0)
            )
            sig = z.mean(axis=1)
            sig.name = "factor_combo"
            return sig

        def _agg_metrics_from_ts(tsdf: pd.DataFrame, *, ppy: float) -> Dict[str, float]:
            r = tsdf["net_return"].dropna()
            if r.empty or float(r.std(ddof=0)) <= 0:
                sharpe = float("nan")
            else:
                sharpe = float(r.mean() / r.std(ddof=0) * np.sqrt(ppy))
            eq = (1.0 + r.fillna(0.0)).cumprod()
            peak = eq.cummax()
            dd = eq / peak - 1.0
            return {
                "n_timestamps": float(len(tsdf)),
                "avg_net_return": (
                    float(tsdf["net_return"].mean()) if not tsdf.empty else float("nan")
                ),
                "avg_gross_return": (
                    float(tsdf["gross_return"].mean())
                    if not tsdf.empty
                    else float("nan")
                ),
                "avg_turnover": (
                    float(tsdf.get("turnover", pd.Series(dtype=float)).mean())
                    if not tsdf.empty
                    else float("nan")
                ),
                "sharpe_net": sharpe,
                "total_return_net": (
                    float(eq.iloc[-1] - 1.0) if not eq.empty else float("nan")
                ),
                "max_drawdown": float(dd.min()) if not dd.empty else float("nan"),
                "periods_per_year": float(ppy),
                "fee_bps": float(args.bt_fee_bps),
                "slippage_bps": float(args.bt_slippage_bps),
            }

        modes = [m.strip() for m in str(args.bt_mode).split(",") if m.strip()]
        holding = (
            int(args.bt_holding) if args.bt_holding is not None else int(args.horizon)
        )
        embargo = args.wf_embargo_bars
        if embargo is None:
            embargo = int(args.horizon) + int(args.bt_lag)
        embargo = max(0, int(embargo))

        def _bt_cfg(mode: str) -> PortfolioBacktestConfig:
            return PortfolioBacktestConfig(
                mode=mode,
                holding_period_bars=holding,
                execution_lag_bars=int(args.bt_lag),
                top_k=int(args.bt_topk),
                bottom_k=int(args.bt_bottomk),
                gross_leverage=float(args.bt_gross_leverage),
                max_weight=float(args.bt_max_weight),
                turnover_limit=(
                    float(args.bt_turnover_limit)
                    if args.bt_turnover_limit is not None
                    else None
                ),
                fee_bps=float(args.bt_fee_bps),
                slippage_bps=float(args.bt_slippage_bps),
                funding_bps_per_bar=float(args.bt_funding_bps_per_bar),
                borrow_bps_per_bar=float(args.bt_borrow_bps_per_bar),
                min_assets=int(args.bt_min_assets),
                periods_per_year=float(periods_per_year) if periods_per_year else None,
                cash_buffer=float(args.bt_cash_buffer),
                equity_mode=str(args.bt_equity_mode),
                initial_capital=1.0,
            )

        # Time index
        ts = processed_panel.index.get_level_values(0)
        uniq = pd.Index(sorted(pd.to_datetime(ts, utc=True).unique()))

        # Storage for walk-forward aggregation
        preds_all_parts: List[pd.Series] = []
        fold_rows: List[dict] = []
        ts_collect_model: Dict[str, List[pd.DataFrame]] = {m: [] for m in modes}
        ts_collect_factor: Dict[str, List[pd.DataFrame]] = {m: [] for m in modes}
        rb_collect_model: Dict[str, List[pd.DataFrame]] = {m: [] for m in modes}
        rb_collect_factor: Dict[str, List[pd.DataFrame]] = {m: [] for m in modes}

        def _run_on_test_segment(
            *, test_panel: pd.DataFrame, preds: pd.Series, prefix: str
        ) -> None:
            if not bool(args.backtest):
                return
            base = test_panel[["close"]].copy()
            base["model_prediction"] = preds.reindex(base.index)
            base["factor_combo"] = _factor_combo_signal(
                test_panel, list(factor_cols)
            ).reindex(base.index)
            for mode in modes:
                cfg = _bt_cfg(mode)
                ts_m, met_m, rb_m = portfolio_backtest_with_rebalance_log(
                    base, signal_col="model_prediction", close_col="close", cfg=cfg
                )
                ts_f, met_f, rb_f = portfolio_backtest_with_rebalance_log(
                    base, signal_col="factor_combo", close_col="close", cfg=cfg
                )
                # per-fold artifacts
                ts_m.to_csv(output_dir / f"{prefix}model_bt_timeseries__{mode}.csv")
                (output_dir / f"{prefix}model_bt_metrics__{mode}.json").write_text(
                    json.dumps(met_m, indent=2), encoding="utf-8"
                )
                if not rb_m.empty:
                    rb_m.to_csv(
                        output_dir / f"{prefix}model_rebalance_log__{mode}.csv",
                        index=False,
                    )
                ts_f.to_csv(
                    output_dir / f"{prefix}factor_combo_bt_timeseries__{mode}.csv"
                )
                (
                    output_dir / f"{prefix}factor_combo_bt_metrics__{mode}.json"
                ).write_text(json.dumps(met_f, indent=2), encoding="utf-8")
                if not rb_f.empty:
                    rb_f.to_csv(
                        output_dir / f"{prefix}factor_combo_rebalance_log__{mode}.csv",
                        index=False,
                    )

                # Optional: trade list + small HTML report per fold segment (for auditability)
                try:
                    from cross_sectional.rebalance_trade_list import (
                        TradeListConfig,
                        build_trade_list_from_rebalance_log,
                    )

                    close_ser = base["close"].astype(float)
                    tl_cfg = TradeListConfig(
                        mode=str(cfg.mode),
                        gross_leverage=float(cfg.gross_leverage),
                        max_weight=float(cfg.max_weight),
                        cash_buffer=float(cfg.cash_buffer),
                    )
                    # entry timestamps are the rebalance timestamps in the timeseries index
                    if not ts_m.empty and not rb_m.empty:
                        tr_m = build_trade_list_from_rebalance_log(
                            close=close_ser,
                            rb=rb_m,
                            cfg=tl_cfg,
                            entry_timestamps=pd.DatetimeIndex(ts_m.index),
                        )
                        if not tr_m.empty:
                            tr_m.to_csv(
                                output_dir / f"{prefix}model_trades__{mode}.csv",
                                index=False,
                            )
                    if not ts_f.empty and not rb_f.empty:
                        tr_f = build_trade_list_from_rebalance_log(
                            close=close_ser,
                            rb=rb_f,
                            cfg=tl_cfg,
                            entry_timestamps=pd.DatetimeIndex(ts_f.index),
                        )
                        if not tr_f.empty:
                            tr_f.to_csv(
                                output_dir / f"{prefix}factor_combo_trades__{mode}.csv",
                                index=False,
                            )

                    # simple HTML linking artifacts (kept tiny; pipeline index.html provides main conclusions)
                    html_path = output_dir / f"{prefix}backtest_report__{mode}.html"

                    def _fmt(x):
                        try:
                            import math

                            if x is None:
                                return "NA"
                            if isinstance(x, float):
                                if math.isnan(x):
                                    return "NA"
                                return f"{x:.6g}"
                            return str(x)
                        except Exception:
                            return str(x)

                    html = f"""
<html><head><meta charset=\"utf-8\"/>
<title>CS Backtest Report ({mode})</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; padding: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
th {{ background: #f6f6f6; position: sticky; top: 0; }}
code {{ background: #f6f8fa; padding: 2px 4px; }}
</style></head>
<body>
<h1>CS Backtest Report ({mode})</h1>
<h2>Summary</h2>
<ul>
  <li><b>Sharpe(net)</b> model={_fmt(met_m.get('sharpe_net'))} vs factor-combo={_fmt(met_f.get('sharpe_net'))}</li>
  <li><b>Total return(net)</b> model={_fmt(met_m.get('total_return_net'))} vs factor-combo={_fmt(met_f.get('total_return_net'))}</li>
  <li><b>Max drawdown</b> model={_fmt(met_m.get('max_drawdown'))} vs factor-combo={_fmt(met_f.get('max_drawdown'))}</li>
</ul>
<h2>Artifacts</h2>
<ul>
  <li><a href=\"{prefix}model_bt_timeseries__{mode}.csv\">model timeseries</a> / <a href=\"{prefix}model_bt_metrics__{mode}.json\">model metrics</a> / <a href=\"{prefix}model_rebalance_log__{mode}.csv\">model rebalance log</a> / <a href=\"{prefix}model_trades__{mode}.csv\">model trades</a></li>
  <li><a href=\"{prefix}factor_combo_bt_timeseries__{mode}.csv\">factor-combo timeseries</a> / <a href=\"{prefix}factor_combo_bt_metrics__{mode}.json\">factor-combo metrics</a> / <a href=\"{prefix}factor_combo_rebalance_log__{mode}.csv\">factor-combo rebalance log</a> / <a href=\"{prefix}factor_combo_trades__{mode}.csv\">factor-combo trades</a></li>
</ul>
</body></html>
"""
                    html_path.write_text(html, encoding="utf-8")
                except Exception:
                    pass
                # collect for aggregation
                if not ts_m.empty:
                    ts_collect_model[mode].append(ts_m)
                if not ts_f.empty:
                    ts_collect_factor[mode].append(ts_f)
                if not rb_m.empty:
                    rb_collect_model[mode].append(rb_m)
                if not rb_f.empty:
                    rb_collect_factor[mode].append(rb_f)

        # Walk-forward expanding folds if enabled
        if int(args.wf_folds or 1) > 1:
            k = int(args.wf_folds)
            edges = np.linspace(0, len(uniq), k + 1).astype(int)
            for fi in range(k):
                test_start = int(edges[fi])
                test_end = int(edges[fi + 1])
                if test_end - test_start <= 0:
                    continue
                train_end = max(0, test_start - embargo)
                if train_end <= 1:
                    continue
                train_ts = set(uniq[:train_end])
                test_ts = set(uniq[test_start:test_end])
                train_panel = processed_panel.loc[
                    processed_panel.index.get_level_values(0).isin(train_ts)
                ]
                test_panel = processed_panel.loc[
                    processed_panel.index.get_level_values(0).isin(test_ts)
                ]
                if train_panel.empty or test_panel.empty:
                    continue
                model = CrossSectionalBoostingModel()
                model.fit(train_panel, feature_cols=factor_cols, target_col=target_col)
                preds = model.predict(test_panel)
                preds_all_parts.append(preds)
                eval_tmp = model.evaluate(
                    test_panel, predictions=preds, target_col=target_col
                )
                fold_rows.append(
                    {
                        "fold": fi + 1,
                        "train_timestamps": int(
                            train_panel.index.get_level_values(0).nunique()
                        ),
                        "test_timestamps": int(
                            test_panel.index.get_level_values(0).nunique()
                        ),
                        "ic_mean": float(eval_tmp.information_coefficients.mean()),
                        "rank_ic_mean": float(eval_tmp.rank_ic.mean()),
                    }
                )
                _run_on_test_segment(
                    test_panel=test_panel, preds=preds, prefix=f"wf{fi+1}_"
                )

            # concatenate all fold test predictions
            predictions_test = (
                pd.concat(preds_all_parts).sort_index()
                if preds_all_parts
                else pd.Series(dtype=float)
            )
            (output_dir / "walk_forward_summary.json").write_text(
                json.dumps(fold_rows, indent=2), encoding="utf-8"
            )

            # predictive IC on concatenated test predictions (using processed_panel targets)
            aligned = (
                processed_panel[[target_col]]
                .join(predictions_test.rename("pred"), how="inner")
                .dropna()
            )
            ic = aligned.groupby(level=0).apply(lambda x: x[target_col].corr(x["pred"]))
            ric = aligned.groupby(level=0).apply(
                lambda x: x[target_col].corr(x["pred"], method="spearman")
            )
            eval_result = type(
                "Eval",
                (),
                {
                    "information_coefficients": ic,
                    "rank_ic": ric,
                    "mse_by_timestamp": pd.Series(dtype=float),
                },
            )()

            # Aggregate portfolio metrics across folds and write as the canonical files (for HTML)
            if bool(args.backtest):
                for mode in modes:
                    # model
                    ts_all = (
                        pd.concat(ts_collect_model[mode]).sort_index()
                        if ts_collect_model[mode]
                        else pd.DataFrame()
                    )
                    if not ts_all.empty:
                        ts_all.to_csv(output_dir / f"model_bt_timeseries__{mode}.csv")
                        ppy = (
                            float(
                                ts_all.get("periods_per_year", pd.Series([np.nan]))
                                .dropna()
                                .iloc[0]
                            )
                            if "periods_per_year" in ts_all.columns
                            else float(365.0 * 24.0 / (4.0 * holding))
                        )
                        met = _agg_metrics_from_ts(
                            ts_all, ppy=float(periods_per_year) / float(holding)
                        )
                        (output_dir / f"model_bt_metrics__{mode}.json").write_text(
                            json.dumps(met, indent=2), encoding="utf-8"
                        )
                    rb_all = (
                        pd.concat(rb_collect_model[mode]).sort_values("rebalance_ts")
                        if rb_collect_model[mode]
                        else pd.DataFrame()
                    )
                    if not rb_all.empty:
                        rb_all.to_csv(
                            output_dir / f"model_rebalance_log__{mode}.csv", index=False
                        )
                    # factor combo
                    ts_all2 = (
                        pd.concat(ts_collect_factor[mode]).sort_index()
                        if ts_collect_factor[mode]
                        else pd.DataFrame()
                    )
                    if not ts_all2.empty:
                        ts_all2.to_csv(
                            output_dir / f"factor_combo_bt_timeseries__{mode}.csv"
                        )
                        met2 = _agg_metrics_from_ts(
                            ts_all2, ppy=float(periods_per_year) / float(holding)
                        )
                        (
                            output_dir / f"factor_combo_bt_metrics__{mode}.json"
                        ).write_text(json.dumps(met2, indent=2), encoding="utf-8")
                    rb_all2 = (
                        pd.concat(rb_collect_factor[mode]).sort_values("rebalance_ts")
                        if rb_collect_factor[mode]
                        else pd.DataFrame()
                    )
                    if not rb_all2.empty:
                        rb_all2.to_csv(
                            output_dir / f"factor_combo_rebalance_log__{mode}.csv",
                            index=False,
                        )

                    # Also build aggregated trade lists + a small HTML report at the end
                    try:
                        from cross_sectional.rebalance_trade_list import (
                            TradeListConfig,
                            build_trade_list_from_rebalance_log,
                        )

                        # close prices from processed_panel (test periods are already subset in aggregated ts)
                        close_ser = processed_panel[["close"]].copy()
                        close_ser["close"] = pd.to_numeric(
                            close_ser["close"], errors="coerce"
                        )
                        close = close_ser["close"]
                        tl_cfg = TradeListConfig(
                            mode=str(mode),
                            gross_leverage=float(args.bt_gross_leverage),
                            max_weight=float(args.bt_max_weight),
                            cash_buffer=float(args.bt_cash_buffer),
                        )
                        # Use timestamps from the aggregated model ts as entry timestamps
                        ts_all = pd.read_csv(
                            output_dir / f"model_bt_timeseries__{mode}.csv",
                            parse_dates=["timestamp"],
                        )
                        ts_all["timestamp"] = pd.to_datetime(
                            ts_all["timestamp"], utc=True, errors="coerce"
                        )
                        entry_ts = pd.DatetimeIndex(
                            ts_all["timestamp"].dropna().unique()
                        )
                        if not rb_all.empty and len(entry_ts) >= 2:
                            tr_m = build_trade_list_from_rebalance_log(
                                close=close,
                                rb=rb_all,
                                cfg=tl_cfg,
                                entry_timestamps=entry_ts,
                            )
                            if not tr_m.empty:
                                tr_m.to_csv(
                                    output_dir / f"model_trades__{mode}.csv",
                                    index=False,
                                )
                        if not rb_all2.empty and len(entry_ts) >= 2:
                            tr_f = build_trade_list_from_rebalance_log(
                                close=close,
                                rb=rb_all2,
                                cfg=tl_cfg,
                                entry_timestamps=entry_ts,
                            )
                            if not tr_f.empty:
                                tr_f.to_csv(
                                    output_dir / f"factor_combo_trades__{mode}.csv",
                                    index=False,
                                )
                        # one HTML per mode at train root
                        html_path = output_dir / f"backtest_report__{mode}.html"
                        m_met = (
                            json.loads(
                                (
                                    output_dir / f"model_bt_metrics__{mode}.json"
                                ).read_text(encoding="utf-8")
                            )
                            if (output_dir / f"model_bt_metrics__{mode}.json").exists()
                            else {}
                        )
                        f_met = (
                            json.loads(
                                (
                                    output_dir / f"factor_combo_bt_metrics__{mode}.json"
                                ).read_text(encoding="utf-8")
                            )
                            if (
                                output_dir / f"factor_combo_bt_metrics__{mode}.json"
                            ).exists()
                            else {}
                        )

                        def _fmt(x):
                            try:
                                import math

                                if x is None:
                                    return "NA"
                                if isinstance(x, float):
                                    if math.isnan(x):
                                        return "NA"
                                    return f"{x:.6g}"
                                return str(x)
                            except Exception:
                                return str(x)

                        html = f"""
<html><head><meta charset=\"utf-8\"/>
<title>CS Backtest Report ({mode})</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; padding: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
th {{ background: #f6f6f6; position: sticky; top: 0; }}
code {{ background: #f6f8fa; padding: 2px 4px; }}
</style></head>
<body>
<h1>CS Backtest Report ({mode})</h1>
<h2>Conclusion</h2>
<ul>
  <li><b>Sharpe(net)</b> model={_fmt(m_met.get('sharpe_net'))} vs factor-combo={_fmt(f_met.get('sharpe_net'))}</li>
  <li><b>Total return(net)</b> model={_fmt(m_met.get('total_return_net'))} vs factor-combo={_fmt(f_met.get('total_return_net'))}</li>
  <li><b>Max drawdown</b> model={_fmt(m_met.get('max_drawdown'))} vs factor-combo={_fmt(f_met.get('max_drawdown'))}</li>
</ul>
<h2>Artifacts</h2>
<ul>
  <li><a href=\"model_bt_timeseries__{mode}.csv\">model timeseries</a> / <a href=\"model_bt_metrics__{mode}.json\">model metrics</a> / <a href=\"model_rebalance_log__{mode}.csv\">model rebalance log</a> / <a href=\"model_trades__{mode}.csv\">model trades</a></li>
  <li><a href=\"factor_combo_bt_timeseries__{mode}.csv\">factor-combo timeseries</a> / <a href=\"factor_combo_bt_metrics__{mode}.json\">factor-combo metrics</a> / <a href=\"factor_combo_rebalance_log__{mode}.csv\">factor-combo rebalance log</a> / <a href=\"factor_combo_trades__{mode}.csv\">factor-combo trades</a></li>
</ul>
</body></html>
"""
                        html_path.write_text(html, encoding="utf-8")
                    except Exception:
                        pass

            # for model dump, re-fit on full pre-test data (best effort)
            model = CrossSectionalBoostingModel()
            split = int(np.floor(len(uniq) * float(args.train_ratio)))
            train_end = max(0, split - embargo)
            train_panel = processed_panel.loc[
                processed_panel.index.get_level_values(0).isin(set(uniq[:train_end]))
            ]
            if not train_panel.empty:
                model.fit(train_panel, feature_cols=factor_cols, target_col=target_col)
        else:
            # Single split with embargo
            model = CrossSectionalBoostingModel()
            split = int(np.floor(len(uniq) * float(args.train_ratio)))
            split = min(max(split, 1), max(len(uniq) - 1, 1))
            train_end = max(0, split - embargo)
            train_ts = set(uniq[:train_end]) if train_end > 0 else set(uniq[:split])
            test_ts = set(uniq[split:])
            train_panel = processed_panel.loc[
                processed_panel.index.get_level_values(0).isin(train_ts)
            ]
            test_panel = processed_panel.loc[
                processed_panel.index.get_level_values(0).isin(test_ts)
            ]

            model.fit(train_panel, feature_cols=factor_cols, target_col=target_col)
            predictions_test = model.predict(test_panel)
            eval_result = model.evaluate(
                test_panel, predictions=predictions_test, target_col=target_col
            )
            _run_on_test_segment(
                test_panel=test_panel, preds=predictions_test, prefix=""
            )

        save_predictions(predictions_test, output_dir / args.predictions_name)
        save_metrics(
            eval_result,
            output_dir / args.metrics_name,
            selected_features=factor_cols,
            selection_metrics=selection_metrics,
            periods_per_year=periods_per_year,
        )
        dump(model, output_dir / args.model_name)

        if args.save_markdown:
            # Use a representative test panel for diagnostics.
            # For walk-forward, a single contiguous test_panel variable may not exist here.
            _panel_for_report = (
                processed_panel.loc[
                    processed_panel.index.get_level_values(0).isin(
                        set(uniq[int(np.floor(len(uniq) * float(args.train_ratio))) :])
                    )
                ]
                if int(args.wf_folds or 1) <= 1
                else processed_panel.loc[
                    processed_panel.index.get_level_values(0).isin(
                        set(uniq[int(len(uniq) * 0.7) :])
                    )
                ]
            )
            report = build_report_from_eval(
                _panel_for_report,
                factor_cols,
                eval_result,
                args,
            )
            write_report(output_dir / "boosting_report.md", report)

        if selection_metrics is not None and not selection_metrics.empty:
            selection_path = output_dir / args.selection_output
            selection_path.write_text(
                selection_metrics.to_json(orient="index", indent=2), encoding="utf-8"
            )
            print(f"   📄 Factor selection metrics saved to {selection_path}")

        print(f"✅ Boosting model trained. Artefacts saved under {output_dir}")

    elif args.model == "fama_macbeth":
        reg = CrossSectionalRegressor(add_intercept=True, min_assets=3)
        result = reg.fit(
            processed_panel, factor_cols=factor_cols, target_col=target_col
        )
        metrics = {
            "factor_summary": result.factor_summary(periods_per_year).to_dict(),
            "ic_summary": result.ic_summary(periods_per_year).to_dict(),
            "newey_west": result.newey_west_summary(
                max_lag=5, periods_per_year=periods_per_year
            ).to_dict(),
            "selected_features": factor_cols,
        }
        if selection_metrics is not None and not selection_metrics.empty:
            metrics["selection_metrics"] = selection_metrics.to_dict(orient="index")
        (output_dir / args.metrics_name).write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        if args.save_markdown:
            context = FactorPanelBuilder.describe_panel(processed_panel)
            report = generate_markdown_report(
                result,
                ReportContext(
                    title="Cross-Sectional Fama-MacBeth Training Report",
                    max_lag=5,
                    periods_per_year=periods_per_year,
                    preprocessing=_describe_preprocessing(args.winsor, args.zscore),
                    symbols=args.symbols
                    or ", ".join(sorted(filtered_df["symbol"].unique())),
                    horizon=args.horizon,
                    observations=int(context.get("num_observations", 0)),
                    timestamps=int(context.get("num_timestamps", 0)),
                    assets_per_timestamp=float(
                        context.get("mean_assets_per_timestamp", 0.0)
                    ),
                ),
            )
            write_report(output_dir / "fama_macbeth_report.md", report)

        if selection_metrics is not None and not selection_metrics.empty:
            selection_path = output_dir / args.selection_output
            selection_path.write_text(
                selection_metrics.to_json(orient="index", indent=2), encoding="utf-8"
            )
            print(f"   📄 Factor selection metrics saved to {selection_path}")

        print(f"✅ Fama-MacBeth regression complete. Metrics saved under {output_dir}")
    else:
        raise ValueError(f"Unsupported model: {args.model}")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def collect_inputs(patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded and Path(pattern).exists():
            expanded = [pattern]
        files.extend(expanded)
    unique = sorted({os.path.abspath(p) for p in files})
    if not unique:
        raise FileNotFoundError(f"No input files match: {patterns}")
    return unique


def load_frames(paths: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        ext = Path(path).suffix.lower()
        if ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext in {".csv", ".txt"}:
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {path}")
        if df.empty:
            continue
        if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
        if "timestamp" not in df.columns:
            raise ValueError(f"'timestamp' column missing in {path}")
        if "symbol" not in df.columns:
            df["symbol"] = _infer_symbol_from_path(path)
        frames.append(df)
    if not frames:
        raise ValueError("All input frames are empty.")
    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"], utc=True, errors="coerce"
    )
    combined = combined.dropna(subset=["timestamp", "symbol"])
    combined = combined.sort_values(["timestamp", "symbol"])
    combined = combined.set_index(["timestamp", "symbol"])
    combined["timestamp"] = combined.index.get_level_values("timestamp")
    combined["symbol"] = combined.index.get_level_values("symbol")
    return combined


def filter_symbols(df: pd.DataFrame, symbols: Optional[str]) -> pd.DataFrame:
    if not symbols:
        return df
    symbol_list = [
        s.strip().upper() for s in symbols.replace(" ", ",").split(",") if s.strip()
    ]
    if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
        mask = df.index.get_level_values("symbol").str.upper().isin(symbol_list)
        return df[mask].copy()
    if "symbol" in df.columns:
        return df[df["symbol"].str.upper().isin(symbol_list)].copy()
    raise ValueError("Symbol filtering requested but 'symbol' column/level not found.")


def ensure_future_return_column(
    df: pd.DataFrame,
    horizon: int,
    price_col: str = "close",
) -> tuple[pd.DataFrame, str]:
    col_name = f"future_return_{horizon}"
    if col_name in df.columns:
        return df, col_name
    if price_col not in df.columns:
        raise ValueError(f"{price_col} column missing; cannot compute forward return.")
    df_sorted = df.sort_values(["symbol", "timestamp"]).copy()
    df_sorted[col_name] = df_sorted.groupby("symbol")[price_col].apply(
        lambda x: x.shift(-horizon) / x - 1.0
    )
    return df_sorted, col_name


def build_panel(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]],
    min_assets: int,
    horizon: int,
) -> tuple[pd.DataFrame, List[str]]:
    config = PanelConfig(
        feature_cols=feature_cols,
        target_col=target_col,
        forward_return_horizon=horizon,
        min_assets_per_ts=min_assets,
        fill_method="ffill",
        dropna_after_fill=False,
        align_intersection_only=False,
    )
    builder = FactorPanelBuilder(config)
    panel = builder.from_concat_frame(df)
    if feature_cols:
        return panel, list(feature_cols)

    exclude_cols = {
        target_col,
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp",
        "symbol",
    }
    numeric_cols = [
        col
        for col in panel.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(panel[col])
    ]
    return panel, numeric_cols


def preprocess_panel(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    winsor_sigma: float,
    apply_zscore: bool,
) -> pd.DataFrame:
    processed = panel.copy()
    if winsor_sigma and winsor_sigma > 0:
        processed = winsorize_by_sigma(processed, factor_cols, sigma=winsor_sigma)
    if apply_zscore:
        processed = cross_sectional_zscore(processed, factor_cols)
    return processed


def save_predictions(predictions: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = predictions.to_frame(name="predicted_return")
    df.to_parquet(path)


def save_metrics(
    eval_result,
    path: Path,
    *,
    selected_features: Sequence[str],
    selection_metrics: Optional[pd.DataFrame],
    periods_per_year: float,
) -> None:
    metrics = {
        "ic_mean": float(eval_result.information_coefficients.mean()),
        "ic_std": float(eval_result.information_coefficients.std(ddof=0)),
        "rank_ic_mean": float(eval_result.rank_ic.mean()),
        "rank_ic_std": float(eval_result.rank_ic.std(ddof=0)),
        "selected_features": list(selected_features),
        "periods_per_year": periods_per_year,
    }
    if selection_metrics is not None and not selection_metrics.empty:
        metrics["selection_metrics"] = selection_metrics.to_dict(orient="index")
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def build_report_from_eval(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    eval_result,
    args: argparse.Namespace,
) -> str:
    ic_summary = eval_result.information_coefficients.describe().to_dict()
    rank_ic_summary = eval_result.rank_ic.describe().to_dict()
    mse_summary = eval_result.mse_by_timestamp.describe().to_dict()

    diagnostics = FactorPanelBuilder.describe_panel(panel)

    lines = [
        "# Cross-Sectional Boosting Training Report",
        "",
        f"- Symbols: {args.symbols or 'all'}",
        f"- Horizon: {args.horizon} bars",
        f"- Periods/year: {args.periods_per_year}",
        f"- Preprocessing: {_describe_preprocessing(args.winsor, args.zscore)}",
        f"- Features used: {len(factor_cols)}",
        "",
        "## Panel diagnostics",
        "",
        json.dumps(diagnostics, indent=2),
        "",
        "## Information coefficient summary",
        "",
        json.dumps(ic_summary, indent=2),
        "",
        "## Rank IC summary",
        "",
        json.dumps(rank_ic_summary, indent=2),
        "",
        "## MSE by timestamp summary",
        "",
        json.dumps(mse_summary, indent=2),
    ]
    return "\n".join(lines) + "\n"


def _describe_preprocessing(winsor_sigma: float, apply_zscore: bool) -> str:
    steps = []
    if winsor_sigma and winsor_sigma > 0:
        steps.append(f"winsor |σ|<{winsor_sigma}")
    if apply_zscore:
        steps.append("z-score")
    if not steps:
        return "none"
    return " + ".join(steps)


def resolve_periods_per_year(arg_value: str, index: pd.Index) -> float:
    value = (arg_value or "auto").strip().lower()
    if value != "auto":
        try:
            parsed = float(value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    timestamps = index
    if isinstance(timestamps, pd.MultiIndex):
        timestamps = timestamps.get_level_values(0)
    timestamps = pd.to_datetime(timestamps)
    timestamps = timestamps.sort_values().unique()
    if len(timestamps) < 2:
        return 252.0

    diffs_series = pd.Series(timestamps)
    diffs = diffs_series.diff().dropna()
    if diffs.empty:
        return 252.0
    if diffs.nunique() > 1:
        raise ValueError(
            "Detected multiple bar intervals in panel; please provide a single timeframe per run."
        )

    median_seconds = diffs.dt.total_seconds().iloc[0]
    if not median_seconds or median_seconds <= 0:
        return 252.0

    seconds_per_year = 365.0 * 24.0 * 3600.0
    inferred = seconds_per_year / median_seconds
    return float(inferred)


def _infer_symbol_from_path(path: str) -> str:
    stem = Path(path).stem.upper()
    for sep in ("_", "-"):
        if sep in stem:
            return stem.split(sep)[0]
    return stem


if __name__ == "__main__":
    main()
