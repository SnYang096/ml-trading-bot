#!/usr/bin/env python3
"""
Feature Indicators Visualization

Generate HTML visualization for feature indicators based on configuration file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.data_utils import load_raw_data


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


def generate_html_report(
    df: pd.DataFrame,
    config: Dict,
    symbol: str,
    timeframe: str,
    output_path: Path,
) -> None:
    """Generate HTML visualization report from configuration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Basic statistics
    total_bars = len(df)
    date_range = f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "N/A"

    # Get feature types from config
    feature_types_config = config.get("feature_types", {})
    display_config = config.get("display", {})
    output_config = config.get("output", {})

    # Check feature availability for each configured type
    available_features = []
    for ft_key, ft_config in feature_types_config.items():
        if not ft_config.get("enabled", True):
            continue

        patterns = ft_config.get("column_patterns", [])
        matching_cols = find_matching_columns(df, patterns)

        available_features.append(
            {
                "key": ft_key,
                "display_name": ft_config.get("display_name", ft_key.title()),
                "description": ft_config.get("description", ""),
                "columns": matching_cols,
                "count": len(matching_cols),
            }
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

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Feature Indicators Visualization - {symbol} {timeframe}</title>
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
    generate_html_report(df, config, args.symbol, args.timeframe, output_path)


if __name__ == "__main__":
    main()
