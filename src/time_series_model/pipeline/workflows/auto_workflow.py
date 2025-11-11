"""Automated end-to-end workflow for feature selection, training, and rolling evaluation.

Steps:
    1. (Optional) Run dimensionality comparison to pick top factors.
    2. Train production model via training pipeline.
    3. Run rolling evaluation with the same configuration.
    4. Analyse rolling summary for drift and optionally re-run comparison.

This module orchestrates the existing CLI entry points programmatically so that
`make` can expose a single command (`make auto-workflow`) for streamlined usage.

Typical usage (from Makefile):

    PYTHONPATH=src python -m time_series_model.pipeline.workflows.auto_workflow \\
        --data-dir /workspace/data/parquet_data \\
        --symbols BTCUSDT \\
        --feature-type baseline \\
        --compare-start 2025-01-01 --compare-end 2025-06-30 \\
        --train-start 2025-01-01 --train-end 2025-06-30 \\
        --rolling-start 2025-01 --rolling-end 2025-10 \\
        --freqs 15T \\
        --forward-bars-train 5,15 \\
        --forward-bars-rolling 5 \\
        --cv-folds 5 \\
        --oos-months 2
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from time_series_model.pipeline.dimensionality import dimensionality_comparison as dim_compare
from time_series_model.pipeline.training import rolling as rolling_module
from time_series_model.pipeline.training import train as train_module
from time_series_model.pipeline.training import train_regime_gated as gated_train_module
from time_series_model.pipeline.workflows import gated_to_position as gated_workflow_module
from time_series_model.monitoring import online_monitors as monitors_module
from time_series_model.monitoring.model_registry import ModelRegistry


def _print_header(msg: str) -> None:
    bar = "=" * len(msg)
    print(f"\n{bar}\n{msg}\n{bar}")


@contextmanager
def _argv_context(args: Sequence[str]) -> Iterable[None]:
    original = sys.argv[:]
    sys.argv = [sys.argv[0], *args]
    try:
        yield
    finally:
        sys.argv = original


def _list_dirs(path: Path) -> set[Path]:
    return {p for p in path.glob("*") if p.is_dir()}


def _detect_new_dir(base_dir: Path, before: set[Path]) -> Optional[Path]:
    if not base_dir.exists():
        return None
    after = _list_dirs(base_dir)
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if new_dirs:
        return new_dirs[0]
    # Fallback: pick most recent directory if nothing new detected
    if after:
        return sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def _normalize_month(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    cleaned = date_str.strip()
    if len(cleaned) >= 10:
        cleaned = cleaned[:7]
    try:
        dt = datetime.strptime(cleaned, "%Y-%m")
    except ValueError:
        return None
    return dt.strftime("%Y-%m")


def _subtract_months(end_month: str, months: int) -> str:
    dt = datetime.strptime(end_month, "%Y-%m")
    period = pd.Period(dt.strftime("%Y-%m"), freq="M")
    new_period = period - months
    return f"{new_period.year}-{new_period.month:02d}"


def _run_training(args_list: List[str]) -> Optional[Path]:
    base_dir = Path("results/training")
    before = _list_dirs(base_dir) if base_dir.exists() else set()
    with _argv_context(args_list):
        train_module.main()
    return _detect_new_dir(base_dir, before)


def _run_rolling(args_list: List[str]) -> Optional[Path]:
    results_glob = Path("results")
    before = {p for p in results_glob.glob("rolling_*") if p.is_dir()}
    with _argv_context(args_list):
        rolling_module.main()
    return _detect_new_dir(results_glob, before)


THRESHOLDS: Dict[str, Tuple[float, bool]] = {
    "cls_accuracy": (0.5, False),
    "cls_precision": (0.5, False),
    "cls_recall": (0.5, False),
    "cls_f1": (0.5, False),
    "cls_auc": (0.5, False),
    "cls_pr_auc": (0.5, False),
    "cls_ic_spearman": (0.05, True),
    "cls_ic_pearson": (0.05, True),
    "test_r2_return": (0.0, False),
}


def _check_drift(summary_path: Path) -> Tuple[bool, List[str]]:
    if not summary_path.exists():
        return False, []

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    monthly = summary.get("monthly_results", [])
    failing: List[str] = []
    for row in monthly:
        period = row.get("test_month") or row.get("quarter") or "N/A"
        issues = []
        for metric, (threshold, use_abs) in THRESHOLDS.items():
            value = row.get(metric)
            if value is None:
                continue
            comp_value = abs(value) if use_abs else value
            if comp_value < threshold:
                issues.append(f"{metric}={value:.4f} < {threshold:.2f}")
        if issues:
            failing.append(f"{period}: {', '.join(issues)}")

    return bool(failing), failing


def _load_top_factors(results_dir: str) -> Optional[str]:
    if not results_dir:
        return None
    candidate = Path(results_dir) / "top_factors.json"
    return str(candidate) if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated workflow: feature comparison → train → rolling evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/parquet_data")
    parser.add_argument("--symbols", default="BTCUSDT")
    parser.add_argument("--feature-type", default="baseline")
    parser.add_argument("--compare-start", default=None)
    parser.add_argument("--compare-end", default=None)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--rolling-start", default=None)
    parser.add_argument("--rolling-end", default=None)
    parser.add_argument("--freqs", default="15T")
    parser.add_argument("--forward-bars-train", default="5")
    parser.add_argument("--forward-bars-rolling", default="5")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--oos-months", type=int, default=2)
    parser.add_argument("--initial-train-months", type=int, default=3)
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--direction-threshold", default="f1_optimize")
    parser.add_argument(
        "--model-type",
        default="classification",
        choices=["classification", "quantile"],
        help="Model type passed to training pipeline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=120,
        help="Number of top factors to keep from dimensionality comparison.",
    )
    parser.add_argument(
        "--shap-analysis",
        action="store_true",
        help="Generate SHAP explainability outputs for representative features.",
    )
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument(
        "--retry-train-months",
        type=int,
        default=6,
        help="Months of history to use when re-running dim comparison on drift.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip dimensionality comparison (requires --top-factors).",
    )
    parser.add_argument("--top-factors", default=None, help="Existing top_factors.json path.")
    parser.add_argument(
        "--auto-recompare",
        action="store_true",
        help="If drift detected, re-run comparison with recent window (limited by --max-iterations).",
    )
    parser.add_argument("--gpu", action="store_true", default=False)
    # Optional: gated experts path
    parser.add_argument(
        "--use-gated-experts",
        action="store_true",
        help="Enable training of Momentum/MeanReversion/Breakout experts and generate positions.",
    )
    parser.add_argument(
        "--run-forward-selection",
        action="store_true",
        help="Run forward horizon selection (information efficiency) before training.",
    )
    parser.add_argument(
        "--gated-timeframes",
        default="15T,60T,240T",
        help="Comma-separated timeframes for gated experts (e.g., 15T,60T,240T).",
    )
    parser.add_argument(
        "--gated-save-dir",
        default="results/gated_positions",
        help="Where to save gated positions outputs.",
    )
    parser.add_argument(
        "--gated-horizons",
        default="",
        help="Comma-separated forward horizons for gated multi-horizon fusion, e.g., 2,6,12 (optional).",
    )
    parser.add_argument(
        "--run-monitors",
        action="store_true",
        help="Run online monitoring (calibration & drift) after gated positions.",
    )
    parser.add_argument(
        "--registry-path",
        default="",
        help="Optional path to JSON registry for logging model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path("results").mkdir(exist_ok=True)

    registry: Optional[ModelRegistry] = None
    if args.registry_path:
        registry = ModelRegistry(args.registry_path)

    compare_start = args.compare_start
    compare_end = args.compare_end
    gated_timeframes = args.gated_timeframes
    train_start = args.train_start
    train_end = args.train_end
    rolling_start = args.rolling_start
    rolling_end = args.rolling_end

    top_factors_path = args.top_factors
    compare_results_dir: Optional[str] = None

    for iteration in range(1, args.max_iterations + 1):
        _print_header(f"AUTO WORKFLOW ITERATION #{iteration}")

        # ---------------- Optional: Forward selection ----------------
        if args.run_forward_selection:
            _print_header("Pre-Stage: Forward Horizon Selection")
            try:
                from time_series_model.pipeline.training import forward_selection as fsel
                fsel_args: List[str] = [
                    "--data-dir",
                    args.data_dir,
                    "--symbol",
                    args.symbols,
                    "--timeframes",
                    gated_timeframes,
                    "--max-forward",
                    "48",
                    "--save-dir",
                    "results/forward_selection",
                ]
                with _argv_context(fsel_args):
                    fsel.main()
            except Exception as e:
                print(f"⚠️ Forward selection failed: {e}")

        # ---------------- Stage 1: Dimensionality comparison ----------------
        if not args.skip_compare or (args.auto_recompare and iteration > 1):
            _print_header("Stage 1: Dimensionality Comparison")
            dim_args: List[str] = [
                "--data-path",
                args.data_dir,
                "--symbol",
                args.symbols,
                "--feature-type",
                args.feature_type,
                "--top-k",
                str(args.top_k),
            ]
            if compare_start:
                dim_args.extend(["--train-start", compare_start])
            if compare_end:
                dim_args.extend(["--train-end", compare_end])
            if args.shap_analysis:
                dim_args.append("--shap-analysis")
            with _argv_context(dim_args):
                results, _, autoencoder, compare_dir = dim_compare.main()
            compare_results_dir = compare_dir
            derived_top = _load_top_factors(compare_dir)
            if derived_top:
                top_factors_path = derived_top
                print(f"✅ Using top factors from: {derived_top}")
            else:
                print("⚠️ No top_factors.json found; proceeding without feature filtering.")
            shap_dir = results.get("explainability", {}).get("stage3_shap_dir")
            if shap_dir:
                print(f"📊 SHAP explainability artifacts: {shap_dir}")
            if registry and compare_dir:
                registry.log(
                    pipeline="dimension_selection",
                    symbol=args.symbols,
                    artifact_path=str(compare_dir),
                    metrics={},
                    params={"top_k": args.top_k},
                    notes=f"shap_dir={shap_dir}" if shap_dir else None,
                )
        elif not top_factors_path:
            raise ValueError("Top factors not provided; cannot skip comparison stage.")

        # ---------------- Stage 2: Training ----------------
        _print_header("Stage 2: Model Training")
        train_args: List[str] = [
            "--data-dir",
            args.data_dir,
            "--symbol",
            args.symbols,
            "--freq",
            args.freqs,
            "--forward-bars",
            args.forward_bars_train,
            "--cv-folds",
            str(args.cv_folds),
            "--feature-type",
            args.feature_type,
            "--oos-months",
            str(args.oos_months),
            "--direction-threshold",
            args.direction_threshold,
            "--model-type",
            args.model_type,
        ]

        normalized_train_start = _normalize_month(train_start)
        normalized_train_end = _normalize_month(train_end)
        if normalized_train_start:
            train_args.extend(["--start", normalized_train_start])
        if normalized_train_end:
            train_args.extend(["--end", normalized_train_end])
        if top_factors_path:
            train_args.extend(["--use-top-factors", top_factors_path])
        # Enable safe multi-asset handling (mirrors Makefile default)
        train_args.append("--safe-multi-asset")
        if args.gpu:
            train_args.append("--gpu")
        training_dir = _run_training(train_args)
        print(f"📁 Training results directory: {training_dir or 'N/A'}")
        if registry and training_dir:
            registry.log(
                pipeline="ts_training",
                symbol=args.symbols,
                artifact_path=str(training_dir),
                metrics={},
                params={
                    "freq": args.freqs,
                    "forward_bars_train": args.forward_bars_train,
                    "cv_folds": args.cv_folds,
                },
            )

        # ---------------- Stage 2b: Regime-gated experts (optional) ----------------
        if args.use_gated_experts:
            _print_header("Stage 2b: Gated Experts (Momentum@1h, MeanReversion@15m, Breakout@1h/4h)")
            # Train experts
            gated_train_args: List[str] = [
                "--data-dir",
                args.data_dir,
                "--symbol",
                args.symbols,
                "--feature-type",
                args.feature_type,
                "--timeframes",
                args.gated_timeframes,
            ]
            with _argv_context(gated_train_args):
                gated_train_module.main()
            # Generate positions with RiskManager
            gated_positions_args: List[str] = [
                "--data-dir",
                args.data_dir,
                "--symbol",
                args.symbols,
                "--timeframes",
                args.gated_timeframes,
                "--save-dir",
                args.gated_save_dir,
            ]
            if args.gated_horizons:
                gated_positions_args.extend(["--multi-horizons", args.gated_horizons])
            with _argv_context(gated_positions_args):
                gated_workflow_module.main()
            print(f"✅ Gated experts trained and positions generated in {args.gated_save_dir}")
            pos_path = Path(args.gated_save_dir) / args.symbols / "positions.parquet"
            if registry and pos_path.exists():
                registry.log(
                    pipeline="ts_gated_positions",
                    symbol=args.symbols,
                    artifact_path=str(pos_path),
                    metrics={},
                    params={
                        "timeframes": args.gated_timeframes,
                        "horizons": args.gated_horizons or "6",
                    },
                )
            # Online monitoring
            if args.run_monitors:
                _print_header("Stage 2c: Online Monitoring (Calibration & Drift)")
                try:
                    pos_path = Path(args.gated_save_dir) / args.symbols / "positions.parquet"
                    if pos_path.exists():
                        mon_args: List[str] = [
                            "--data-dir",
                            args.data_dir,
                            "--symbol",
                            args.symbols,
                            "--positions",
                            str(pos_path),
                            "--price-tf",
                            "60T",
                            "--forward-bars",
                            "6",
                            "--save-dir",
                            "results/monitoring",
                        ]
                        with _argv_context(mon_args):
                            monitors_module.main()
                        if registry:
                            registry.log(
                                pipeline="ts_monitoring",
                                symbol=args.symbols,
                                artifact_path=str(Path("results/monitoring") / args.symbols),
                                metrics={},
                                params={},
                            )
                    else:
                        print(f"⚠️ Positions file not found for monitoring: {pos_path}")
                except Exception as e:
                    print(f"⚠️ Monitoring failed: {e}")

        # ---------------- Stage 3: Rolling ----------------
        _print_header("Stage 3: Rolling Evaluation")
        rolling_args: List[str] = [
            "--data-dir",
            args.data_dir,
            "--symbol",
            args.symbols,
            "--freq",
            args.freqs,
            "--forward-bars",
            args.forward_bars_rolling,
            "--initial-train-months",
            str(args.initial_train_months),
            "--min-train-months",
            str(args.min_train_months),
            "--cv-folds",
            str(args.cv_folds),
            "--direction-threshold",
            args.direction_threshold,
            "--feature-type",
            args.feature_type,
        ]
        normalized_roll_start = _normalize_month(rolling_start)
        normalized_roll_end = _normalize_month(rolling_end)
        if normalized_roll_start:
            rolling_args.extend(["--start", normalized_roll_start])
        if normalized_roll_end:
            rolling_args.extend(["--end", normalized_roll_end])
        if top_factors_path:
            rolling_args.extend(["--use-top-factors", top_factors_path])
        if args.gpu:
            rolling_args.append("--gpu")
        rolling_dir = _run_rolling(rolling_args)
        if not rolling_dir:
            print("⚠️ Rolling output directory not detected.")
            return
        print(f"📁 Rolling results directory: {rolling_dir}")
        if registry and rolling_dir:
            registry.log(
                pipeline="ts_rolling",
                symbol=args.symbols,
                artifact_path=str(rolling_dir),
                metrics={},
                params={
                    "timeframes": args.freqs,
                    "forward_bars_rolling": args.forward_bars_rolling,
                },
            )

        summary_path = Path(rolling_dir) / "summary.json"
        drift, issues = _check_drift(summary_path)
        if not drift:
            print("✅ Rolling evaluation passed threshold checks.")
            if compare_results_dir:
                print(f"Top factors directory: {compare_results_dir}")
            print(f"Training directory: {training_dir}")
            print(f"Rolling directory: {rolling_dir}")
            return

        print("⚠️ Drift detected in rolling evaluation:")
        for issue in issues:
            print(f"   - {issue}")

        if not args.auto_recompare or iteration == args.max_iterations:
            print("⛔ Auto re-compare disabled or max iterations reached; stopping with drift warnings.")
            print(f"Rolling directory: {rolling_dir}")
            return

        if not normalized_roll_end:
            print("⚠️ Cannot perform auto re-compare because rolling end date is undefined.")
            return

        months_back = max(args.retry_train_months - 1, 0)
        new_start = _subtract_months(normalized_roll_end, months_back)
        print(f"🔁 Re-running comparison with recent window starting {new_start}")

        compare_start = new_start
        train_start = new_start
        rolling_start = new_start


if __name__ == "__main__":
    main()

