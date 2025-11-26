#!/usr/bin/env python3
"""Compare strategy performance across different feature configurations."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("strategy_feature_compare")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger.setLevel(logging.INFO)

from scripts import train_strategy_pipeline as strategy_runner  # noqa: E402
from src.data_tools.data_utils import load_raw_data  # noqa: E402
from src.features.loader.strategy_feature_loader import (
    StrategyFeatureLoader,
)  # noqa: E402
from src.strategy_config import StrategyConfig, StrategyConfigLoader  # noqa: E402

VENDOR_DIR = PROJECT_ROOT / "vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare strategy feature configurations."
    )
    parser.add_argument(
        "--strategy-config", required=True, help="Base strategy directory"
    )
    parser.add_argument(
        "--feature-overrides",
        nargs="*",
        default=[],
        help="List of variant definitions in the form name=path/to/features.yaml",
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--timeframe", default="240T")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional inclusive start date (e.g. 2022-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive end date (e.g. 2023-01-01)",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output-dir", default="results/strategy_compare")
    parser.add_argument("--run-rolling", action="store_true")
    parser.add_argument("--rolling-train-bars", type=int, default=5000)
    parser.add_argument("--rolling-test-bars", type=int, default=1000)
    parser.add_argument("--rolling-step-bars", type=int, default=1000)
    parser.add_argument("--rolling-max-windows", type=int, default=5)
    parser.add_argument(
        "--test-warmup-bars",
        type=int,
        default=200,
        help="Extra bars before the test split for feature warm-up",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(path: Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def merge_features(base_path: Path, override_path: Path, variant_name: str) -> Dict:
    base_data = load_yaml(base_path)
    override_data = load_yaml(override_path)
    merged = dict(base_data)
    for key, value in override_data.items():
        if key == "feature_pipeline" and isinstance(value, dict):
            merged.setdefault("feature_pipeline", {})
            merged["feature_pipeline"].update(value)
        else:
            merged[key] = value
    merged["name"] = f"{base_data.get('name', 'strategy')}_{variant_name}"
    return merged


def update_meta(meta_path: Path, variant_name: str) -> None:
    data = load_yaml(meta_path)
    strategy_info = data.get("strategy", {})
    base_name = strategy_info.get("name", variant_name)
    strategy_info["name"] = f"{base_name}_{variant_name}"
    data["strategy"] = strategy_info
    dump_yaml(meta_path, data)


@dataclass
class VariantSpec:
    name: str
    config_dir: Path
    is_temp: bool = False


def build_variants(
    base_dir: Path, overrides: List[str]
) -> Tuple[List[VariantSpec], List[Path]]:
    variants = [VariantSpec(name="base", config_dir=base_dir, is_temp=False)]
    temp_dirs: List[Path] = []

    for entry in overrides:
        if "=" in entry:
            variant_name, override_path = entry.split("=", 1)
        else:
            variant_name = Path(entry).stem
            override_path = entry
        variant_name = variant_name.strip()
        override_path = Path(override_path).resolve()
        if not override_path.exists():
            raise FileNotFoundError(f"Override file not found: {override_path}")

        temp_dir = Path(tempfile.mkdtemp(prefix=f"strategy_variant_{variant_name}_"))
        shutil.copytree(base_dir, temp_dir, dirs_exist_ok=True)
        merged_features = merge_features(
            base_dir / "features.yaml", override_path, variant_name
        )
        dump_yaml(temp_dir / "features.yaml", merged_features)
        meta_path = temp_dir / "meta.yaml"
        if meta_path.exists():
            update_meta(meta_path, variant_name)
        variants.append(
            VariantSpec(name=variant_name, config_dir=temp_dir, is_temp=True)
        )
        temp_dirs.append(temp_dir)

    return variants, temp_dirs


def execute_single_run(
    strategy_cfg: StrategyConfig,
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    test_warmup_bars: int = 0,
    variant_name: str = "unknown",
) -> Optional[Dict]:
    if df_train_raw.empty or df_test_raw.empty:
        logger.warning(
            "Variant %s has empty train/test split (train=%d, test=%d)",
            variant_name,
            len(df_train_raw),
            len(df_test_raw),
        )
        return None

    logger.info(
        "Variant %s raw samples → train=%d, test=%d",
        variant_name,
        len(df_train_raw),
        len(df_test_raw),
    )

    feature_loader = StrategyFeatureLoader()
    df_train_features = strategy_runner.run_feature_pipeline(
        df_train_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )
    df_test_features = strategy_runner.run_feature_pipeline(
        df_test_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=False,
    )

    if test_warmup_bars > 0 and len(df_test_features) > test_warmup_bars:
        df_test_features = df_test_features.iloc[test_warmup_bars:].copy()

    feature_cols = strategy_runner.determine_feature_columns(
        df_train_features, strategy_cfg.features
    )
    label_func = strategy_runner.import_callable(
        strategy_cfg.labels.generator.module, strategy_cfg.labels.generator.function
    )
    target_col = strategy_cfg.labels.target_column
    df_train_features[target_col] = label_func(
        df_train_features.copy(), **strategy_cfg.labels.generator.params
    )
    df_test_features[target_col] = label_func(
        df_test_features.copy(), **strategy_cfg.labels.generator.params
    )
    logger.info(
        "Variant %s labels computed → train targets=%d (NaN=%d) test targets=%d (NaN=%d)",
        variant_name,
        len(df_train_features),
        int(df_train_features[target_col].isna().sum()),
        len(df_test_features),
        int(df_test_features[target_col].isna().sum()),
    )

    df_train_filtered = strategy_runner.apply_filters(
        df_train_features, strategy_cfg.labels.filters
    )
    df_test_filtered = strategy_runner.apply_filters(
        df_test_features, strategy_cfg.labels.filters
    )
    logger.info(
        "Variant %s after label filters → train=%d, test=%d",
        variant_name,
        len(df_train_filtered),
        len(df_test_filtered),
    )

    df_train_filtered = strategy_runner.apply_post_label_filters(
        df_train_filtered, strategy_cfg.labels.post_label_filters, feature_cols
    )
    df_test_filtered = strategy_runner.apply_post_label_filters(
        df_test_filtered, strategy_cfg.labels.post_label_filters, feature_cols
    )

    logger.info(
        "Variant %s after post-label filters → train=%d, test=%d",
        variant_name,
        len(df_train_filtered),
        len(df_test_filtered),
    )

    if len(df_train_filtered) < 50 or len(df_test_filtered) < 10:
        logger.warning(
            "Variant %s skipped: insufficient samples after filters (train=%d, test=%d)",
            variant_name,
            len(df_train_filtered),
            len(df_test_filtered),
        )
        return None

    trainer_func = strategy_runner.import_callable(
        strategy_cfg.model.trainer.module, strategy_cfg.model.trainer.function
    )
    trainer_params = dict(strategy_cfg.model.trainer.params)
    target_col = trainer_params.pop("target_col", target_col)
    model_type = trainer_params.get("model_type", "xgboost")
    task_type = trainer_params.get("task_type", "regression")

    models, avg_metric, cv_results, used_features = trainer_func(
        df_train_filtered,
        feature_cols=feature_cols,
        target_col=target_col,
        **trainer_params,
    )

    X_test = df_test_filtered[used_features].values
    y_test = df_test_filtered[target_col].values
    preds = strategy_runner.generate_predictions(
        models=models,
        model_type=model_type,
        task_type=task_type,
        X=X_test,
    )

    evaluation_results = strategy_runner.evaluate_predictions(
        preds, y_test, strategy_cfg.evaluation
    )
    backtest_results = strategy_runner.run_vectorbt_backtest(
        df_test_filtered, preds, strategy_cfg.backtest, task_type
    )

    logger.info(
        "Variant %s finished training with %d features, CV metric %.4f",
        variant_name,
        len(used_features),
        float(avg_metric) if avg_metric is not None else float("nan"),
    )

    return {
        "avg_cv_metric": float(avg_metric),
        "evaluation": evaluation_results,
        "backtest": backtest_results,
        "used_features": used_features,
        "n_train": int(len(df_train_filtered)),
        "n_test": int(len(df_test_filtered)),
    }


def run_rolling_evaluation(
    strategy_cfg: StrategyConfig,
    df_raw: pd.DataFrame,
    params: argparse.Namespace,
    variant_name: str = "unknown",
) -> Optional[Dict]:
    train_size = params.rolling_train_bars
    test_size = params.rolling_test_bars
    step = params.rolling_step_bars
    max_windows = params.rolling_max_windows

    windows: List[Dict] = []
    start = 0
    while start + train_size + test_size <= len(df_raw) and len(windows) < max_windows:
        train_raw = df_raw.iloc[start : start + train_size].copy()
        test_raw = df_raw.iloc[
            start + train_size : start + train_size + test_size
        ].copy()
        result = execute_single_run(
            strategy_cfg, train_raw, test_raw, variant_name=variant_name
        )
        if result:
            result["window_start"] = str(train_raw.index[0])
            result["window_end"] = str(test_raw.index[-1])
            windows.append(result)
        start += step

    if not windows:
        return None

    eval_keys = sorted({k for w in windows for k in w["evaluation"].keys()})
    aggregate_eval = {
        key: float(np.nanmean([w["evaluation"].get(key, np.nan) for w in windows]))
        for key in eval_keys
    }
    if any(w.get("backtest") for w in windows):
        bt_keys = sorted(
            {k for w in windows if w.get("backtest") for k in w["backtest"].keys()}
        )
        aggregate_bt = {
            key: float(
                np.nanmean(
                    [
                        w["backtest"].get(key, np.nan)
                        for w in windows
                        if w.get("backtest")
                    ]
                )
            )
            for key in bt_keys
        }
    else:
        aggregate_bt = None

    avg_cv = float(np.nanmean([w["avg_cv_metric"] for w in windows]))
    return {
        "windows": windows,
        "aggregate": {
            "avg_cv_metric": avg_cv,
            "evaluation": aggregate_eval,
            "backtest": aggregate_bt,
            "n_windows": len(windows),
        },
    }


def summarize_results(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in results:
        row = {
            "variant": item["variant"],
            "avg_cv_metric": (
                item["base"]["avg_cv_metric"] if item.get("base") else np.nan
            ),
            "n_train": item["base"].get("n_train", 0) if item.get("base") else 0,
            "n_test": item["base"].get("n_test", 0) if item.get("base") else 0,
        }
        evaluation = item["base"].get("evaluation", {}) if item.get("base") else {}
        for key, value in evaluation.items():
            row[f"eval_{key}"] = value
        backtest = item["base"].get("backtest") if item.get("base") else None
        if backtest:
            for key, value in backtest.items():
                # 跳过 debug 等非标量字段，避免污染 summary CSV
                if key == "debug":
                    continue
                row[f"bt_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def generate_html_report(
    results: List[Dict],
    summary_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    test_size: float,
    output_dir: Path,
) -> Path:
    """Generate HTML report comparing strategy variants."""

    # Extract all evaluation and backtest metrics
    eval_metrics = set()
    bt_metrics = set()
    for item in results:
        if item.get("base") and item["base"].get("evaluation"):
            eval_metrics.update(item["base"]["evaluation"].keys())
        if item.get("base") and item["base"].get("backtest"):
            bt_metrics.update(
                k for k in item["base"]["backtest"].keys() if k != "debug"
            )

    eval_metrics = sorted(list(eval_metrics))
    bt_metrics = sorted(list(bt_metrics))

    # Prepare data for charts
    variants = [item["variant"] for item in results]

    # Evaluation metrics chart data
    eval_chart_data = {}
    for metric in eval_metrics:
        values = []
        for item in results:
            val = item.get("base", {}).get("evaluation", {}).get(metric, 0.0)
            if val is None or np.isnan(val):
                val = 0.0
            values.append(float(val))
        eval_chart_data[metric] = values

    # Backtest metrics chart data
    bt_chart_data = {}
    for metric in bt_metrics:
        values = []
        for item in results:
            val = item.get("base", {}).get("backtest", {}).get(metric, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            values.append(float(val))
        bt_chart_data[metric] = values

    # CV metric chart data
    cv_metrics = []
    for item in results:
        val = item.get("base", {}).get("avg_cv_metric", 0.0)
        if val is None or np.isnan(val):
            val = 0.0
        cv_metrics.append(float(val))

    # Find best variant for each metric
    def find_best_variant(
        metric_values: List[float], higher_is_better: bool = True
    ) -> int:
        """Find index of best variant. Returns -1 if all are invalid."""
        valid_values = [
            (i, v) for i, v in enumerate(metric_values) if v != 0.0 and not np.isnan(v)
        ]
        if not valid_values:
            return -1
        if higher_is_better:
            return max(valid_values, key=lambda x: x[1])[0]
        else:
            return min(valid_values, key=lambda x: x[1])[0]

    # Generate summary table HTML
    table_rows = []
    for idx, item in enumerate(results):
        variant = item["variant"]
        base = item.get("base", {})

        # Determine row color based on performance
        row_class = ""
        if base:
            cv_val = base.get("avg_cv_metric", 0.0)
            if cv_val and not np.isnan(cv_val):
                best_cv_idx = find_best_variant(cv_metrics)
                if best_cv_idx == idx:
                    row_class = "best-row"

        cells = [f'<td class="{row_class}"><strong>{variant}</strong></td>']

        # CV metric
        cv_val = base.get("avg_cv_metric", np.nan) if base else np.nan
        cv_display = (
            f"{cv_val:.4f}" if cv_val is not None and not np.isnan(cv_val) else "N/A"
        )
        cells.append(f'<td class="{row_class}">{cv_display}</td>')

        # Evaluation metrics
        eval_data = base.get("evaluation", {}) if base else {}
        for metric in eval_metrics:
            val = eval_data.get(metric, np.nan)
            val_display = (
                f"{val:.4f}" if val is not None and not np.isnan(val) else "N/A"
            )
            best_idx = find_best_variant(eval_chart_data.get(metric, []))
            cell_class = row_class
            if best_idx == idx:
                cell_class = "best-cell"
            cells.append(f'<td class="{cell_class}">{val_display}</td>')

        # Backtest metrics
        bt_data = base.get("backtest", {}) if base else {}
        for metric in bt_metrics:
            val = bt_data.get(metric, np.nan)
            if metric in ["total_return_pct", "sharpe"]:
                val_display = (
                    f"{val:.2f}" if val is not None and not np.isnan(val) else "N/A"
                )
            elif metric == "max_drawdown_pct":
                val_display = (
                    f"{val:.2f}%" if val is not None and not np.isnan(val) else "N/A"
                )
            else:
                val_display = (
                    f"{val:.4f}" if val is not None and not np.isnan(val) else "N/A"
                )
            best_idx = find_best_variant(
                bt_chart_data.get(metric, []),
                higher_is_better=(metric != "max_drawdown_pct"),
            )
            cell_class = row_class
            if best_idx == idx:
                cell_class = "best-cell"
            cells.append(f'<td class="{cell_class}">{val_display}</td>')

        # Sample sizes
        n_train = base.get("n_train", 0) if base else 0
        n_test = base.get("n_test", 0) if base else 0
        cells.append(f'<td class="{row_class}">{n_train}</td>')
        cells.append(f'<td class="{row_class}">{n_test}</td>')

        table_rows.append(f"<tr>{''.join(cells)}</tr>")

    # Table header
    header_cells = ["<th>Variant</th>", "<th>CV Metric</th>"]
    for metric in eval_metrics:
        header_cells.append(f"<th>Eval: {metric}</th>")
    for metric in bt_metrics:
        header_cells.append(f"<th>BT: {metric}</th>")
    header_cells.extend(["<th>Train Samples</th>", "<th>Test Samples</th>"])
    table_header = f"<tr>{''.join(header_cells)}</tr>"

    # Generate chart scripts
    chart_scripts = []

    # CV Metric chart
    if cv_metrics:
        chart_scripts.append(
            f"""
        // CV Metric Comparison
        const ctx_cv = document.getElementById('cvChart');
        if (ctx_cv) {{
            new Chart(ctx_cv, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(variants)},
                    datasets: [{{
                        label: 'CV Metric',
                        data: {json.dumps(cv_metrics)},
                        backgroundColor: 'rgba(33, 150, 243, 0.6)',
                        borderColor: 'rgb(33, 150, 243)',
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Cross-Validation Metric Comparison',
                            font: {{ size: 16, weight: 'bold' }}
                        }},
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{
                            title: {{ display: true, text: 'CV Metric Value' }},
                            grid: {{ color: 'rgba(0, 0, 0, 0.05)' }}
                        }}
                    }}
                }}
            }});
        }}
        """
        )

    # Evaluation metrics chart
    if eval_metrics:
        datasets = []
        colors = [
            ("rgba(33, 150, 243, 0.6)", "rgb(33, 150, 243)"),
            ("rgba(76, 175, 80, 0.6)", "rgb(76, 175, 80)"),
            ("rgba(255, 152, 0, 0.6)", "rgb(255, 152, 0)"),
            ("rgba(156, 39, 176, 0.6)", "rgb(156, 39, 176)"),
            ("rgba(244, 67, 54, 0.6)", "rgb(244, 67, 54)"),
        ]
        for idx, metric in enumerate(eval_metrics):
            color_idx = idx % len(colors)
            bg_color, border_color = colors[color_idx]
            datasets.append(
                f"""{{
                label: '{metric}',
                data: {json.dumps(eval_chart_data[metric])},
                backgroundColor: '{bg_color}',
                borderColor: '{border_color}',
                borderWidth: 2
            }}"""
            )

        chart_scripts.append(
            f"""
        // Evaluation Metrics Comparison
        const ctx_eval = document.getElementById('evalChart');
        if (ctx_eval) {{
            new Chart(ctx_eval, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(variants)},
                    datasets: [{','.join(datasets)}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Evaluation Metrics Comparison',
                            font: {{ size: 16, weight: 'bold' }}
                        }},
                        legend: {{ display: true, position: 'top' }}
                    }},
                    scales: {{
                        y: {{
                            title: {{ display: true, text: 'Metric Value' }},
                            grid: {{ color: 'rgba(0, 0, 0, 0.05)' }}
                        }}
                    }}
                }}
            }});
        }}
        """
        )

    # Backtest metrics chart
    if bt_metrics:
        datasets = []
        colors = [
            ("rgba(33, 150, 243, 0.6)", "rgb(33, 150, 243)"),
            ("rgba(76, 175, 80, 0.6)", "rgb(76, 175, 80)"),
            ("rgba(255, 152, 0, 0.6)", "rgb(255, 152, 0)"),
            ("rgba(156, 39, 176, 0.6)", "rgb(156, 39, 176)"),
            ("rgba(244, 67, 54, 0.6)", "rgb(244, 67, 54)"),
        ]
        for idx, metric in enumerate(bt_metrics):
            color_idx = idx % len(colors)
            bg_color, border_color = colors[color_idx]
            datasets.append(
                f"""{{
                label: '{metric}',
                data: {json.dumps(bt_chart_data[metric])},
                backgroundColor: '{bg_color}',
                borderColor: '{border_color}',
                borderWidth: 2
            }}"""
            )

        chart_scripts.append(
            f"""
        // Backtest Metrics Comparison
        const ctx_bt = document.getElementById('btChart');
        if (ctx_bt) {{
            new Chart(ctx_bt, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(variants)},
                    datasets: [{','.join(datasets)}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Backtest Metrics Comparison',
                            font: {{ size: 16, weight: 'bold' }}
                        }},
                        legend: {{ display: true, position: 'top' }}
                    }},
                    scales: {{
                        y: {{
                            title: {{ display: true, text: 'Metric Value' }},
                            grid: {{ color: 'rgba(0, 0, 0, 0.05)' }}
                        }}
                    }}
                }}
            }});
        }}
        """
        )

    # ---------------------------------------------------------------------
    # Optional detailed debug report (signals & trades per variant)
    # ---------------------------------------------------------------------
    debug_sections: List[str] = []
    for item in results:
        variant = item["variant"]
        base = item.get("base", {})
        bt_data = base.get("backtest", {}) if base else {}
        debug_data = bt_data.get("debug") if isinstance(bt_data, dict) else None
        if not debug_data:
            continue

        # Summary
        summary = debug_data.get("summary", {})
        trades_meta = debug_data.get("trades_meta", {})
        returns_stats = debug_data.get("returns_stats", {})

        section_parts: List[str] = []
        section_parts.append(f'<h2 id="variant-{variant}">Variant: {variant}</h2>')
        section_parts.append("<div class='info-box'><ul>")
        if summary:
            section_parts.append(
                f"<li><strong>Total Return:</strong> {summary.get('total_return_pct', 0.0):.2f}%</li>"
            )
            section_parts.append(
                f"<li><strong>Sharpe:</strong> {summary.get('sharpe', 0.0):.2f}</li>"
            )
            section_parts.append(
                f"<li><strong>Max DD:</strong> {summary.get('max_drawdown_pct', 0.0):.2f}%</li>"
            )
            section_parts.append(
                f"<li><strong>Win Rate:</strong> {summary.get('win_rate_pct', 0.0):.2f}%</li>"
            )
        if trades_meta:
            section_parts.append(
                f"<li><strong>Trades:</strong> {trades_meta.get('n_trades', 0)} "
                f"(wins={trades_meta.get('n_win', 0)}, "
                f"win_rate_manual={trades_meta.get('win_rate_manual', 0.0):.2f}%)</li>"
            )
        if returns_stats:
            section_parts.append(
                f"<li><strong>Returns mean/std:</strong> "
                f"{returns_stats.get('mean', 0.0):.3e} / "
                f"{returns_stats.get('std', 0.0):.3e}</li>"
            )
        section_parts.append("</ul></div>")

        # Helper to render table from list[dict]
        def build_table(records: List[Dict[str, Any]], title: str) -> str:
            if not records:
                return f"<h3>{title}</h3><p>No records.</p>"
            cols = list(records[0].keys())
            header = "".join(f"<th>{c}</th>" for c in cols)
            rows_html = []
            for row in records:
                cells = "".join(f"<td>{row.get(c, '')}</td>" for c in cols)
                rows_html.append(f"<tr>{cells}</tr>")
            return (
                f"<h3>{title}</h3>"
                "<div class='table-wrapper'><table>"
                f"<thead><tr>{header}</tr></thead>"
                f"<tbody>{''.join(rows_html)}</tbody>"
                "</table></div>"
            )

        signals = debug_data.get("signals") or []
        trades = debug_data.get("trades") or []
        section_parts.append(build_table(signals, "Entry Signals (sample)"))
        section_parts.append(build_table(trades, "Trades (sample)"))

        debug_sections.append("".join(section_parts))

    if debug_sections:
        debug_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Strategy Feature Debug Details: {symbol}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 95%;
                    width: 100%;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                    border-left: 4px solid #2196F3;
                    padding-left: 10px;
                }}
                .table-wrapper {{
                    width: 100%;
                    overflow-x: auto;
                }}
                table {{
                    width: 100%;
                    min-width: 900px;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 12px;
                }}
                th, td {{
                    white-space: nowrap;
                    padding: 6px 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #2196F3;
                    color: white;
                    font-weight: 600;
                    position: sticky;
                    top: 0;
                    z-index: 2;
                }}
                .info-box {{
                    background: #e3f2fd;
                    border-left: 4px solid #2196F3;
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Strategy Feature Debug Details</h1>
                <p>This page shows sample signals and trades for each variant where backtest.debug is enabled.</p>
                {''.join(debug_sections)}
            </div>
        </body>
        </html>
        """
        debug_path = output_dir / "strategy_feature_compare_debug.html"
        with open(debug_path, "w", encoding="utf-8") as fh:
            fh.write(debug_html)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Strategy Feature Comparison: {symbol}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 95%;
                width: 100%;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
                border-left: 4px solid #2196F3;
                padding-left: 10px;
            }}
            .table-wrapper {{
                width: 100%;
                overflow-x: auto;
            }}
            table {{
                width: 100%;
                min-width: 900px;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 13px;
            }}
            th, td {{
                white-space: nowrap;
                padding: 8px 10px;
                text-align: left;
            }}
            th {{
                background-color: #2196F3;
                color: white;
                padding: 12px;
                font-weight: 600;
                position: sticky;
                top: 0;
                z-index: 2;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .best-row {{
                background-color: #e8f5e9 !important;
            }}
            .best-cell {{
                background-color: #c8e6c9 !important;
                font-weight: bold;
            }}
            .chart-container {{
                margin: 30px 0;
                padding: 20px;
                background: #f9f9f9;
                border-radius: 8px;
                height: 400px;
            }}
            .info-box {{
                background: #e3f2fd;
                border-left: 4px solid #2196F3;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #f8f9fa;
                border-left: 4px solid #4CAF50;
                padding: 15px;
                border-radius: 4px;
            }}
            .metric-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🆚 Strategy Feature Comparison Report</h1>
            
            <div class="info-box">
                <strong>Configuration:</strong><br>
                Symbol: {symbol} | Timeframe: {timeframe} | Test Size: {test_size:.1%}<br>
                Variants Compared: {len(variants)}<br>
                Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                Debug Details: <a href="strategy_feature_compare_debug.html" target="_blank">Open detailed signals & trades</a> (only for variants with backtest.debug enabled)
            </div>
            
            <h2>📊 Summary Table</h2>
            <div class="table-wrapper">
                <table>
                    <thead>
                        {table_header}
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
            </div>
            
            <h2>📈 Performance Charts</h2>
            
            <h3>Cross-Validation Metric</h3>
            <div class="chart-container">
                <canvas id="cvChart"></canvas>
            </div>
            
            {f'<h3>Evaluation Metrics</h3><div class="chart-container"><canvas id="evalChart"></canvas></div>' if eval_metrics else ''}
            
            {f'<h3>Backtest Metrics</h3><div class="chart-container"><canvas id="btChart"></canvas></div>' if bt_metrics else ''}
            
            <h2>📝 Notes</h2>
            <div class="info-box">
                <ul>
                    <li><strong>Best Performance:</strong> Highlighted in green</li>
                    <li><strong>CV Metric:</strong> Average cross-validation score</li>
                    <li><strong>Evaluation Metrics:</strong> Model performance on test set (e.g., correlation, rank IC)</li>
                    <li><strong>Backtest Metrics:</strong> Simulated trading performance (e.g., Sharpe ratio, total return, max drawdown)</li>
                </ul>
            </div>
        </div>
        
        <script>
            {''.join(chart_scripts)}
        </script>
    </body>
    </html>
    """

    html_path = output_dir / "strategy_feature_compare_report.html"
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    return html_path


def main() -> None:
    args = parse_args()
    base_dir = Path(args.strategy_config).resolve()
    variants, temp_dirs = build_variants(base_dir, args.feature_overrides)

    logger.info(
        "Loading data for %s [%s] from %s (%s → %s)",
        args.symbol,
        args.timeframe,
        args.data_path,
        args.start_date or "beginning",
        args.end_date or "latest",
    )

    df_raw = load_raw_data(
        data_path=args.data_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )
    logger.info("Loaded %d bars for %s", len(df_raw), args.symbol)
    split_idx = int(len(df_raw) * (1 - args.test_size))
    df_train_raw = df_raw.iloc[:split_idx].copy()
    test_warmup = min(args.test_warmup_bars, len(df_train_raw))
    df_test_raw = df_raw.iloc[split_idx - test_warmup :].copy()
    logger.info(
        "Split data → train=%d (%.1f%%) test=%d (%.1f%%)",
        len(df_train_raw),
        100 * (1 - args.test_size),
        len(df_test_raw),
        100 * args.test_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_results = []
    try:
        for variant in variants:
            loader = StrategyConfigLoader(variant.config_dir)
            strategy_cfg = loader.load()
            logger.info("Running variant %s ...", variant.name)
            base_result = execute_single_run(
                strategy_cfg,
                df_train_raw,
                df_test_raw,
                test_warmup_bars=test_warmup,
                variant_name=variant.name,
            )
            rolling_result = None
            if args.run_rolling:
                logger.info(
                    "Starting rolling evaluation for variant %s (%d windows max)",
                    variant.name,
                    args.rolling_max_windows,
                )
                rolling_result = run_rolling_evaluation(
                    strategy_cfg, df_raw, args, variant_name=variant.name
                )
            comparison_results.append(
                {
                    "variant": variant.name,
                    "base": base_result or {},
                    "rolling": rolling_result or {},
                }
            )
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

    summary_df = summarize_results(comparison_results)
    summary_csv = output_dir / "strategy_feature_compare_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    detailed_json = output_dir / "strategy_feature_compare_summary.json"
    with open(detailed_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "test_size": args.test_size,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "results": comparison_results,
            },
            fh,
            indent=2,
            default=str,
        )

    print(f"✅ Saved summary CSV to {summary_csv}")
    print(f"✅ Saved summary JSON to {detailed_json}")

    # Generate HTML report
    html_path = generate_html_report(
        comparison_results,
        summary_df,
        args.symbol,
        args.timeframe,
        args.test_size,
        output_dir,
    )
    print(f"✅ Saved HTML report to {html_path}")


if __name__ == "__main__":
    main()
