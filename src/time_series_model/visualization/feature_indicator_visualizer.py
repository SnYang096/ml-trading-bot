#!/usr/bin/env python3
"""
Feature Indicators Visualization

Generate HTML visualization for feature indicators based on configuration file.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import yaml

try:
    from bokeh.plotting import figure
    from bokeh.layouts import column, gridplot
    from bokeh.embed import components
    from bokeh.models import HoverTool, CrosshairTool, ColumnDataSource
    from bokeh.resources import CDN
    import bokeh

    _BOKEH_AVAILABLE = True
    _BOKEH_VERSION = bokeh.__version__
except ImportError:
    _BOKEH_AVAILABLE = False
    _BOKEH_VERSION = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # ml_trading_bot repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data


def load_data_with_strategy_features(
    df_raw: pd.DataFrame,
    strategy_config_dir: Path,
    symbol: str,
    timeframe: str,
    feature_store_dir: str = "feature_store",
    use_cache: bool = False,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Run strategy feature pipeline to get feature-enriched df.

    Args:
        df_raw: Raw OHLCV DataFrame
        strategy_config_dir: Path to strategy config directory
        symbol: Trading symbol (e.g., BTCUSDT)
        timeframe: Timeframe (e.g., 240T)
        feature_store_dir: FeatureStore root directory
        use_cache: If True, use FeatureStore cache; if False, compute fresh
        force_rebuild: If True, force rebuild FeatureStore cache even if exists
    """
    from src.features.loader.strategy_feature_loader import StrategyFeatureLoader
    from src.time_series_model.strategy_config import StrategyConfigLoader
    from src.feature_store.layer_naming import resolve_layer_name

    config_dir = Path(strategy_config_dir)
    if not config_dir.is_absolute():
        config_dir = PROJECT_ROOT / strategy_config_dir
    if not config_dir.exists():
        raise FileNotFoundError(f"Strategy config not found: {config_dir}")

    loader = StrategyConfigLoader(config_dir)
    strategy_config = loader.load()
    requested = list(
        getattr(strategy_config.features, "requested_features", None) or []
    )
    invert = list(getattr(strategy_config.features, "invert_features", None) or [])
    effective_requested = requested + [c for c in invert if c not in requested]
    if not effective_requested:
        return df_raw

    # Ensure DatetimeIndex
    df = df_raw.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("datetime", "timestamp", "date"):
            if col in df.columns:
                df.index = pd.to_datetime(df[col])
                break
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return df_raw

    feature_loader = StrategyFeatureLoader(
        feature_deps_path=str(PROJECT_ROOT / "config" / "feature_dependencies.yaml"),
    )

    # Determine FeatureStore parameters based on options
    if use_cache:
        fs_dir = feature_store_dir
        fs_layer = resolve_layer_name(None, config_dir)
        fs_symbol = symbol
        fs_timeframe = timeframe

        # If force_rebuild, delete existing cache for this symbol/timeframe
        if force_rebuild:
            try:
                from src.feature_store.feature_store import (
                    FeatureStore,
                    FeatureStoreSpec,
                )

                store = FeatureStore(fs_dir)
                spec = FeatureStoreSpec(
                    layer=fs_layer, symbol=fs_symbol, timeframe=fs_timeframe
                )
                # Get all months in the data range
                months = pd.period_range(
                    start=df.index.min(), end=df.index.max(), freq="M"
                )
                deleted_count = 0
                for p in months:
                    month_str = f"{p.year:04d}-{p.month:02d}"
                    if store.has_month(spec, month_str):
                        store.delete_month(spec, month_str)
                        deleted_count += 1
                if deleted_count > 0:
                    print(
                        f"   🗑️  Deleted {deleted_count} months from FeatureStore cache (force rebuild)"
                    )
            except Exception as e:
                print(f"   ⚠️  Failed to delete cache: {e}")

        print(f"   📦 Using FeatureStore: layer={fs_layer}")
    else:
        fs_dir = None
        fs_layer = None
        fs_symbol = None
        fs_timeframe = None
        print("   🔄 Computing features fresh (no cache)")

    try:
        df_feat = feature_loader.load_features_from_requested(
            df,
            effective_requested,
            fit=True,
            feature_store_dir=fs_dir,
            feature_store_layer=fs_layer,
            feature_store_symbol=fs_symbol,
            feature_store_timeframe=fs_timeframe,
        )
        return df_feat
    except Exception as e:
        print(f"   ⚠️  Strategy feature pipeline failed: {e}")
        return df_raw


def load_config(config_path: Path) -> Dict:
    """Load visualization configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate feature indicators visualization from configuration"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to parquet data directory",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        required=True,
        help="Timeframe (e.g., 15T, 240T)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/visualization/feature_indicators.yaml",
        help="Path to feature indicators configuration file",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (if not provided, will auto-generate with timestamp)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/feature_indicators",
        help="Output directory for auto-generated HTML file",
    )
    parser.add_argument(
        "--strategy-config",
        type=str,
        default=None,
        help="Strategy config dir (e.g. config/strategies/compression_breakout). When set, run feature pipeline to get feature columns for visualization.",
    )
    parser.add_argument(
        "--feature-store-dir",
        type=str,
        default="feature_store",
        help="FeatureStore root dir when using --strategy-config",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Use FeatureStore cache (default: compute fresh)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Force rebuild FeatureStore cache even if exists (requires --use-cache)",
    )
    return parser.parse_args()


def generate_output_filename(
    symbol: str,
    timeframe: str,
    config_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: str = "results/feature_indicators",
) -> Path:
    """Generate output filename with timestamp, symbol, timeframe, and config name."""
    # Get config name from file path
    config_name = config_path.stem  # e.g., "feature_indicators"

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build filename components
    parts = [
        symbol,
        timeframe.replace("T", "min").replace("H", "h"),
        config_name,
    ]

    # Add date range if provided
    if start_date:
        start_tag = start_date.replace("-", "")
        parts.append(f"from{start_tag}")
    if end_date:
        end_tag = end_date.replace("-", "")
        parts.append(f"to{end_tag}")

    # Add timestamp
    parts.append(timestamp)

    # Join parts
    filename = "_".join(parts) + ".html"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path / filename


def find_matching_columns(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    """Find columns matching any of the given patterns."""
    matching = []
    for col in df.columns:
        for pattern in patterns:
            if pattern.lower() in col.lower():
                matching.append(col)
                break
    return matching


def _load_feature_deps_features() -> Dict[str, Dict]:
    """Load feature_defs['features'] from config/feature_dependencies.yaml."""
    deps_path = PROJECT_ROOT / "config" / "feature_dependencies.yaml"
    if not deps_path.exists():
        return {}
    try:
        with open(deps_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("features", {}) or {}
    except Exception:
        return {}


def _build_strategy_feature_groups(
    df: pd.DataFrame,
    strategy_config_dir: Optional[str],
) -> List[Dict]:
    """
    Build feature groups directly from a strategy's requested_features/invert_features.

    Each feature node becomes one group; its columns come from feature_dependencies.yaml
    (output_columns) intersected with df.columns.
    """
    if not strategy_config_dir:
        return []

    try:
        from src.time_series_model.strategy_config import StrategyConfigLoader
    except Exception:
        return []

    config_dir = Path(strategy_config_dir)
    if not config_dir.is_absolute():
        config_dir = PROJECT_ROOT / config_dir
    if not config_dir.exists():
        return []

    try:
        loader = StrategyConfigLoader(config_dir)
        strategy_config = loader.load()
    except Exception:
        return []

    requested = list(
        getattr(strategy_config.features, "requested_features", None) or []
    )
    invert = list(getattr(strategy_config.features, "invert_features", None) or [])
    effective_requested = requested + [c for c in invert if c not in requested]

    features_cfg = _load_feature_deps_features()
    groups: List[Dict] = []
    seen: set = set()

    for feat_name in effective_requested:
        if feat_name in seen:
            continue
        seen.add(feat_name)

        cols: List[str] = []
        if feat_name in features_cfg:
            cols = features_cfg[feat_name].get("output_columns", [feat_name]) or [
                feat_name
            ]
        else:
            # Fallback: use a simple substring match based on the node name
            base = feat_name[:-2] if feat_name.endswith("_f") else feat_name
            cols = [c for c in df.columns if base in c]

        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue

        groups.append(
            {
                "key": feat_name,
                "display_name": feat_name,
                "description": features_cfg.get(feat_name, {}).get("description", ""),
                "columns": cols,
                "count": len(cols),
            }
        )

    return groups


def _plot_feature_time_series_bokeh(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    max_cols: int = 6,
    height_per_row: int = 250,
) -> Optional[str]:
    """Plot interactive time series for given columns using Bokeh; return HTML div string."""
    if not _BOKEH_AVAILABLE or not columns:
        return None
    cols = columns[:max_cols]
    n = len(cols)
    if n == 0:
        return None
    try:
        plots = []
        for col in cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            source = ColumnDataSource(
                data={
                    "x": series.index,
                    "y": series.values,
                    "col_name": [col] * len(series),
                }
            )

            p = figure(
                title=col,
                x_axis_type="datetime",
                height=height_per_row,
                width=400,
                tools="pan,box_zoom,wheel_zoom,reset,save",
                sizing_mode="stretch_width",
            )
            p.line("x", "y", source=source, line_width=1.5, alpha=0.8, color="#1f77b4")

            hover = HoverTool(
                tooltips=[
                    ("Date", "@x{%F %H:%M}"),
                    ("Value", "@y{0.0000}"),
                ],
                formatters={"@x": "datetime"},
                mode="vline",
            )
            p.add_tools(hover)
            p.add_tools(CrosshairTool(dimensions="both"))
            p.xaxis.axis_label = "Time"
            p.yaxis.axis_label = col
            plots.append(p)

        if not plots:
            return None

        # Arrange in grid (3 columns max)
        ncols = min(3, len(plots))
        grid_rows = []
        for i in range(0, len(plots), ncols):
            grid_rows.append(plots[i : i + ncols])

        grid = gridplot(grid_rows, sizing_mode="stretch_width")
        script, div = components(grid)
        return f"{script}\n{div}"
    except Exception as e:
        print(f"   ⚠️  Bokeh plot error: {e}")
        return None


def generate_html_report(
    df: pd.DataFrame,
    config: Dict,
    symbol: str,
    timeframe: str,
    output_path: Path,
    strategy_config_dir: Optional[str] = None,
) -> None:
    """Generate HTML visualization report from configuration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Basic statistics
    total_bars = len(df)
    date_range = f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "N/A"

    # Display settings from config
    display_config = config.get("display", {})

    show_plots = display_config.get("show_plots", True)
    max_cols_per_chart = display_config.get("max_columns_per_chart", 6)
    chart_height_per_row = display_config.get("chart_height_per_row", 200)

    # Build feature availability list from strategy's requested_features
    if not strategy_config_dir:
        raise ValueError("--strategy-config is required for feature visualization")
    available_features = _build_strategy_feature_groups(df, strategy_config_dir)

    # Generate time-series plots for each feature type that has columns
    plot_html_sections: List[str] = []
    if show_plots and _BOKEH_AVAILABLE:
        for feat in available_features:
            if feat["count"] == 0:
                continue
            bokeh_html = _plot_feature_time_series_bokeh(
                df,
                feat["columns"],
                title=f"{feat['display_name']} — {symbol} {timeframe}",
                max_cols=max_cols_per_chart,
                height_per_row=chart_height_per_row,
            )
            if bokeh_html:
                plot_html_sections.append(
                    f'<div class="chart-block">'
                    f'<h3>{feat["display_name"]}</h3>'
                    f'<p class="chart-desc">{feat["description"]}</p>'
                    f"{bokeh_html}"
                    f"</div>"
                )
    elif show_plots and not _BOKEH_AVAILABLE:
        plot_html_sections.append(
            '<p class="chart-warn">Charts require bokeh. Install with: <code>pip install bokeh</code></p>'
        )

    plots_section_html = ""
    if plot_html_sections:
        plots_section_html = (
            "<h2>Feature Plots</h2>"
            "<p>Time-series of indicator values over the dataset period.</p>"
            + "\n".join(plot_html_sections)
        )

    # Build feature table rows
    feature_rows = []
    for feat in available_features:
        status_class = "status-ok" if feat["count"] > 0 else "status-none"
        status_icon = "✓" if feat["count"] > 0 else "✗"
        status_text = (
            f'Available ({feat["count"]} columns)' if feat["count"] > 0 else "Not found"
        )

        feature_rows.append(
            f"""
                <tr>
                    <td><strong>{feat["display_name"]}</strong></td>
                    <td>{feat["count"]}</td>
                    <td><span class="{status_class}">{status_icon} {status_text}</span></td>
                    <td>{feat["description"]}</td>
                </tr>"""
        )

        # Add sample columns if enabled
        if display_config.get("show_samples", True) and feat["count"] > 0:
            sample_cols = feat["columns"][: display_config.get("sample_count", 5)]
            sample_text = ", ".join(sample_cols)
            if len(feat["columns"]) > display_config.get("sample_count", 5):
                sample_text += f" ... (+{len(feat['columns']) - display_config.get('sample_count', 5)} more)"
            feature_rows.append(
                f"""
                <tr class="sample-row">
                    <td colspan="4" style="padding-left: 30px; font-size: 0.9em; color: #666;">
                        <em>Sample columns: {sample_text}</em>
                    </td>
                </tr>"""
            )

    # Bokeh CDN with matching version
    bokeh_version = _BOKEH_VERSION or "3.3.4"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Feature Indicators Visualization - {symbol} {timeframe}</title>
    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-{bokeh_version}.min.js"></script>
    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-{bokeh_version}.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .info-box {{
            background-color: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
        }}
        .feature-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .feature-table th,
        .feature-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .feature-table th {{
            background-color: #4CAF50;
            color: white;
        }}
        .feature-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .feature-table tr.sample-row {{
            background-color: #f5f5f5;
        }}
        .status-ok {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .status-none {{
            color: #f44336;
            font-weight: bold;
        }}
        .note {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
        .config-info {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
        }}
        .chart-block {{
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .chart-block h3 {{
            margin-top: 0;
            color: #2e7d32;
        }}
        .chart-desc {{
            color: #555;
            font-size: 0.95em;
            margin-bottom: 15px;
        }}
        .feature-chart {{
            max-width: 100%;
            height: auto;
            display: block;
        }}
        .chart-warn {{
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Feature Indicators Visualization</h1>
        
        <div class="info-box">
            <h2>Dataset Information</h2>
            <p><strong>Symbol:</strong> {symbol}</p>
            <p><strong>Timeframe:</strong> {timeframe}</p>
            <p><strong>Total Bars:</strong> {total_bars:,}</p>
            <p><strong>Date Range:</strong> {date_range}</p>
        </div>

        <div class="config-info">
            <h2>Configuration</h2>
            <p>This report is generated from configuration file: <code>config/visualization/feature_indicators.yaml</code></p>
            <p>To customize which features are visualized, edit the configuration file.</p>
        </div>

        <h2>Feature Types Status</h2>
        <table class="feature-table">
            <thead>
                <tr>
                    <th>Feature Type</th>
                    <th>Available Columns</th>
                    <th>Status</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
{''.join(feature_rows)}
            </tbody>
        </table>

        {plots_section_html}

        <div class="note">
            <h3>📝 Note</h3>
            <p>This visualization shows the availability of feature indicators in the dataset based on the configuration file.</p>
            <p>To modify which features are checked, edit <code>config/visualization/feature_indicators.yaml</code>.</p>
            <p><strong>Total columns in dataset:</strong> {len(df.columns)}</p>
            <p><strong>Sample columns:</strong> {', '.join(df.columns[:10].tolist())}{'...' if len(df.columns) > 10 else ''}</p>
        </div>

        <div class="info-box">
            <h3>Next Steps</h3>
            <ul>
                <li>Use <code>make rolling</code> for config-driven rolling training with feature exports</li>
                <li>Check <code>results/feature_exports/</code> for exported feature data</li>
                <li>Use diagnostic tools in <code>src/diagnostics/</code> for detailed analysis</li>
                <li>Edit <code>config/visualization/feature_indicators.yaml</code> to customize feature visualization</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Feature indicators visualization saved to {output_path}")


def main() -> None:
    """Main function."""
    args = parse_args()

    print("📈 Generating feature indicators visualization...")
    print(f"   Symbol: {args.symbol}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Config: {args.config}")
    if args.start_date:
        print(f"   Start Date: {args.start_date}")
    if args.end_date:
        print(f"   End Date: {args.end_date}")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    try:
        config = load_config(config_path)
        print(f"✅ Loaded configuration from {config_path}")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)

    # Load data
    try:
        df = load_raw_data(
            data_path=args.data_path,
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
        )
        print(f"✅ Loaded {len(df)} bars")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # Optionally run strategy feature pipeline to get feature columns (Hurst, Hilbert, WPT, etc.)
    if getattr(args, "strategy_config", None):
        use_cache = getattr(args, "use_cache", False)
        force_rebuild = getattr(args, "force_rebuild", False)
        try:
            cache_mode = "cache" if use_cache else "fresh"
            rebuild_hint = " (force rebuild)" if force_rebuild else ""
            print(
                f"   Running feature pipeline for strategy: {args.strategy_config} [{cache_mode}{rebuild_hint}]"
            )
            df = load_data_with_strategy_features(
                df,
                strategy_config_dir=args.strategy_config,
                symbol=args.symbol,
                timeframe=args.timeframe,
                feature_store_dir=getattr(args, "feature_store_dir", "feature_store"),
                use_cache=use_cache,
                force_rebuild=force_rebuild,
            )
            print(f"✅ Feature pipeline done: {len(df.columns)} columns")
        except Exception as e:
            print(f"   ⚠️  Strategy features skipped: {e}")

    # Generate output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate filename
        output_path = generate_output_filename(
            symbol=args.symbol,
            timeframe=args.timeframe,
            config_path=config_path,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
        )
        print(f"   Auto-generated output path: {output_path}")

    # Generate report
    generate_html_report(
        df,
        config,
        args.symbol,
        args.timeframe,
        output_path,
        strategy_config_dir=getattr(args, "strategy_config", None),
    )


if __name__ == "__main__":
    main()
