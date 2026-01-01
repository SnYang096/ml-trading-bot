"""
Backtest Results Visualizer

Generate interactive HTML reports with:
- Candlestick charts with trade markers
- Trade list with SHAP feature importance
- Performance metrics summary

Usage:
    from src.time_series_model.visualization.backtest_visualizer import BacktestVisualizer

    visualizer = BacktestVisualizer(
        ohlcv_df=df,
        trades=trades_list,
        model_artifact=artifact,
    )
    visualizer.generate_report("results/backtest/report.html")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class BacktestVisualizer:
    """
    Generate interactive backtest visualization reports.
    """

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        trades: List[Dict[str, Any]],
        model_artifact: Optional[Any] = None,
        strategy_name: str = "Strategy",
        symbol: str = "UNKNOWN",
    ):
        """
        Args:
            ohlcv_df: DataFrame with OHLCV data (index should be datetime)
            trades: List of trade dictionaries with keys:
                - entry_time, exit_time
                - entry_price, exit_price
                - direction (LONG/SHORT or 1/-1)
                - pnl_pct
                - exit_reason
                - features_at_entry (optional)
            model_artifact: ModelArtifact for SHAP analysis (optional)
            strategy_name: Strategy name for report title
            symbol: Trading symbol
        """
        self.ohlcv_df = ohlcv_df.copy()
        self.trades = trades
        self.model_artifact = model_artifact
        self.strategy_name = strategy_name
        self.symbol = symbol

        # Ensure datetime index
        if not isinstance(self.ohlcv_df.index, pd.DatetimeIndex):
            if "datetime" in self.ohlcv_df.columns:
                self.ohlcv_df = self.ohlcv_df.set_index("datetime")
            else:
                self.ohlcv_df.index = pd.to_datetime(self.ohlcv_df.index)

        # Compute SHAP values if model available
        self.shap_values = None
        self.shap_explainer = None
        if model_artifact and SHAP_AVAILABLE:
            self._compute_shap()

    def _compute_shap(self):
        """Compute SHAP values for the model"""
        try:
            model = self.model_artifact.model
            if isinstance(model, list):
                model = model[0]  # Use first model for SHAP

            # Get feature data
            feature_cols = self.model_artifact.used_features
            available_cols = [c for c in feature_cols if c in self.ohlcv_df.columns]

            if len(available_cols) < len(feature_cols) * 0.5:
                print(
                    f"⚠️ SHAP: Not enough features available ({len(available_cols)}/{len(feature_cols)})"
                )
                return

            X = self.ohlcv_df[available_cols].dropna()

            if len(X) == 0:
                return

            # Create explainer
            if hasattr(model, "booster_") or "lightgbm" in str(type(model)).lower():
                self.shap_explainer = shap.TreeExplainer(model)
            elif hasattr(model, "get_booster") or "xgboost" in str(type(model)).lower():
                self.shap_explainer = shap.TreeExplainer(model)
            else:
                # Fallback to KernelExplainer (slower)
                self.shap_explainer = shap.KernelExplainer(
                    model.predict, X.sample(min(100, len(X)))
                )

            # Sample for speed
            X_sample = X.sample(min(500, len(X)))
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            self.shap_features = available_cols
            self.shap_df = X_sample

            print(f"✅ SHAP values computed for {len(available_cols)} features")

        except Exception as e:
            print(f"⚠️ SHAP computation failed: {e}")

    def generate_report(self, output_path: str) -> str:
        """
        Generate interactive HTML report.

        Args:
            output_path: Path to save HTML report

        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate components
        candlestick_html = self._generate_candlestick_chart()
        trades_table_html = self._generate_trades_table()
        metrics_html = self._generate_metrics_summary()
        shap_html = self._generate_shap_section()

        # Combine into full report
        html = self._build_full_report(
            candlestick_html=candlestick_html,
            trades_table_html=trades_table_html,
            metrics_html=metrics_html,
            shap_html=shap_html,
        )

        output_path.write_text(html, encoding="utf-8")
        print(f"✅ Report saved: {output_path}")

        return str(output_path)

    def _generate_candlestick_chart(self) -> str:
        """Generate Plotly candlestick chart with trade markers"""
        if not PLOTLY_AVAILABLE:
            return "<p>⚠️ Plotly not installed. Install with: pip install plotly</p>"

        # Create figure with candlestick
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Price", "Volume"),
            row_heights=[0.7, 0.3],
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.ohlcv_df.index,
                open=self.ohlcv_df["open"],
                high=self.ohlcv_df["high"],
                low=self.ohlcv_df["low"],
                close=self.ohlcv_df["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Volume bars
        colors = [
            "red" if c < o else "green"
            for c, o in zip(self.ohlcv_df["close"], self.ohlcv_df["open"])
        ]
        fig.add_trace(
            go.Bar(
                x=self.ohlcv_df.index,
                y=self.ohlcv_df["volume"],
                marker_color=colors,
                name="Volume",
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

        # Add trade markers
        for trade in self.trades:
            entry_time = pd.to_datetime(trade["entry_time"])
            exit_time = pd.to_datetime(trade["exit_time"])
            entry_price = trade["entry_price"]
            exit_price = trade["exit_price"]
            direction = trade.get("direction", "LONG")
            if isinstance(direction, int):
                direction = "LONG" if direction == 1 else "SHORT"
            pnl_pct = trade.get("pnl_pct", 0)
            exit_reason = trade.get("exit_reason", "unknown")

            # Entry marker
            entry_color = "green" if direction == "LONG" else "red"
            entry_symbol = "triangle-up" if direction == "LONG" else "triangle-down"

            fig.add_trace(
                go.Scatter(
                    x=[entry_time],
                    y=[entry_price],
                    mode="markers",
                    marker=dict(
                        symbol=entry_symbol,
                        size=12,
                        color=entry_color,
                        line=dict(width=1, color="black"),
                    ),
                    name=f"Entry {direction}",
                    hovertemplate=f"<b>ENTRY {direction}</b><br>"
                    f"Time: {entry_time}<br>"
                    f"Price: {entry_price:.2f}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Exit marker
            exit_color = "blue" if pnl_pct > 0 else "orange"
            exit_symbol = (
                "star"
                if exit_reason == "take_profit"
                else "x" if exit_reason == "stop_loss" else "circle"
            )

            fig.add_trace(
                go.Scatter(
                    x=[exit_time],
                    y=[exit_price],
                    mode="markers",
                    marker=dict(
                        symbol=exit_symbol,
                        size=10,
                        color=exit_color,
                        line=dict(width=1, color="black"),
                    ),
                    name=f"Exit ({exit_reason})",
                    hovertemplate=f"<b>EXIT ({exit_reason})</b><br>"
                    f"Time: {exit_time}<br>"
                    f"Price: {exit_price:.2f}<br>"
                    f"PnL: {pnl_pct:+.2%}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Connection line
            line_color = "green" if pnl_pct > 0 else "red"
            fig.add_trace(
                go.Scatter(
                    x=[entry_time, exit_time],
                    y=[entry_price, exit_price],
                    mode="lines",
                    line=dict(color=line_color, width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title=f"{self.strategy_name} - {self.symbol}",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def _generate_trades_table(self) -> str:
        """Generate trades table HTML"""
        if not self.trades:
            return "<p>No trades to display</p>"

        rows = []
        for i, trade in enumerate(self.trades):
            direction = trade.get("direction", "LONG")
            if isinstance(direction, int):
                direction = "LONG" if direction == 1 else "SHORT"

            pnl_pct = trade.get("pnl_pct", 0)
            pnl_class = "positive" if pnl_pct > 0 else "negative"

            # Get top features if available
            features_str = ""
            if "features_at_entry" in trade and trade["features_at_entry"]:
                top_features = sorted(
                    trade["features_at_entry"].items(),
                    key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                    reverse=True,
                )[:3]
                features_str = ", ".join(
                    [
                        f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in top_features
                    ]
                )

            rows.append(
                f"""
            <tr>
                <td>{i+1}</td>
                <td>{trade.get('entry_time', '-')}</td>
                <td>{trade.get('exit_time', '-')}</td>
                <td class="{direction.lower()}">{direction}</td>
                <td>{trade.get('entry_price', 0):.2f}</td>
                <td>{trade.get('exit_price', 0):.2f}</td>
                <td class="{pnl_class}">{pnl_pct:+.2%}</td>
                <td>{trade.get('exit_reason', '-')}</td>
                <td>{trade.get('bars_held', '-')}</td>
                <td class="features">{features_str}</td>
            </tr>
            """
            )

        return f"""
        <table class="trades-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Direction</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>PnL</th>
                    <th>Exit Reason</th>
                    <th>Bars Held</th>
                    <th>Top Features</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def _generate_metrics_summary(self) -> str:
        """Generate metrics summary HTML"""
        if not self.trades:
            return "<p>No trades to calculate metrics</p>"

        pnl_list = [t.get("pnl_pct", 0) for t in self.trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        total_trades = len(self.trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        total_return = sum(pnl_list)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        sharpe = (
            (np.mean(pnl_list) / np.std(pnl_list) * np.sqrt(252))
            if np.std(pnl_list) > 0
            else 0
        )

        # Max drawdown
        cumulative = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Exit reasons
        exit_reasons = {}
        for t in self.trades:
            reason = t.get("exit_reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        exit_reasons_html = ", ".join([f"{k}: {v}" for k, v in exit_reasons.items()])

        return f"""
        <div class="metrics-grid">
            <div class="metric-box">
                <span class="metric-label">Total Trades</span>
                <span class="metric-value">{total_trades}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">Win Rate</span>
                <span class="metric-value">{win_rate:.1%}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">Total Return</span>
                <span class="metric-value {'positive' if total_return > 0 else 'negative'}">{total_return:+.2%}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">Sharpe Ratio</span>
                <span class="metric-value">{sharpe:.2f}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">Max Drawdown</span>
                <span class="metric-value negative">{max_drawdown:.2%}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">Avg Win</span>
                <span class="metric-value positive">{avg_win:+.2%}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">Avg Loss</span>
                <span class="metric-value negative">{avg_loss:+.2%}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">Exit Reasons</span>
                <span class="metric-value small">{exit_reasons_html}</span>
            </div>
        </div>
        """

    def _generate_shap_section(self) -> str:
        """Generate SHAP feature importance section"""
        if not SHAP_AVAILABLE:
            return "<p>⚠️ SHAP not installed. Install with: pip install shap</p>"

        if self.shap_values is None:
            return "<p>⚠️ SHAP values not available (model not provided or computation failed)</p>"

        try:
            # Handle different SHAP value formats
            if isinstance(self.shap_values, list):
                # Multi-class case - use absolute mean
                shap_abs = np.abs(np.array(self.shap_values)).mean(axis=0).mean(axis=0)
            else:
                shap_abs = np.abs(self.shap_values).mean(axis=0)

            # Create feature importance DataFrame
            importance_df = pd.DataFrame(
                {
                    "feature": self.shap_features,
                    "importance": shap_abs,
                }
            ).sort_values("importance", ascending=False)

            # Generate table
            rows = []
            for i, row in importance_df.head(20).iterrows():
                bar_width = min(
                    100, row["importance"] / importance_df["importance"].max() * 100
                )
                rows.append(
                    f"""
                <tr>
                    <td class="feature-name">{row['feature']}</td>
                    <td class="importance-value">{row['importance']:.4f}</td>
                    <td class="importance-bar">
                        <div class="bar" style="width: {bar_width}%"></div>
                    </td>
                </tr>
                """
                )

            return f"""
            <h3>Feature Importance (SHAP)</h3>
            <table class="shap-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Mean |SHAP|</th>
                        <th>Importance</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
            """
        except Exception as e:
            return f"<p>⚠️ SHAP visualization error: {e}</p>"

    def _build_full_report(
        self,
        candlestick_html: str,
        trades_table_html: str,
        metrics_html: str,
        shap_html: str,
    ) -> str:
        """Build full HTML report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {self.strategy_name}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --positive: #16a34a;
            --negative: #dc2626;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text: #1e293b;
            --text-secondary: #64748b;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: var(--bg);
            color: var(--text);
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            color: var(--primary);
            margin-bottom: 8px;
        }}
        
        h2 {{
            color: var(--text);
            border-bottom: 2px solid var(--border);
            padding-bottom: 8px;
            margin-top: 32px;
        }}
        
        h3 {{
            color: var(--text);
            margin-top: 24px;
        }}
        
        .timestamp {{
            color: var(--text-secondary);
            font-size: 14px;
            margin-bottom: 24px;
        }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
        }}
        
        .metric-box {{
            background: var(--bg);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        
        .metric-label {{
            display: block;
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        
        .metric-value {{
            font-size: 24px;
            font-weight: 600;
        }}
        
        .metric-value.small {{
            font-size: 14px;
        }}
        
        .positive {{ color: var(--positive); }}
        .negative {{ color: var(--negative); }}
        .long {{ color: var(--positive); font-weight: 600; }}
        .short {{ color: var(--negative); font-weight: 600; }}
        
        .trades-table, .shap-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        
        .trades-table th, .trades-table td,
        .shap-table th, .shap-table td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        .trades-table th, .shap-table th {{
            background: var(--bg);
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        
        .trades-table tbody tr:hover {{
            background: var(--bg);
        }}
        
        .features {{
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 11px;
            color: var(--text-secondary);
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .shap-table .bar {{
            height: 16px;
            background: linear-gradient(90deg, var(--primary), #60a5fa);
            border-radius: 4px;
        }}
        
        .importance-bar {{
            width: 200px;
        }}
        
        .feature-name {{
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
        }}
        
        .table-container {{
            max-height: 500px;
            overflow-y: auto;
        }}
        
        @media (max-width: 768px) {{
            body {{ padding: 12px; }}
            .metric-value {{ font-size: 18px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {self.strategy_name} Backtest Report</h1>
        <p class="timestamp">Symbol: {self.symbol} | Generated: {timestamp}</p>
        
        <div class="card">
            <h2>📈 Performance Metrics</h2>
            {metrics_html}
        </div>
        
        <div class="card">
            <h2>📉 Price Chart</h2>
            {candlestick_html}
        </div>
        
        <div class="card">
            <h2>📋 Trade History</h2>
            <div class="table-container">
                {trades_table_html}
            </div>
        </div>
        
        <div class="card">
            {shap_html}
        </div>
    </div>
</body>
</html>
"""


def generate_backtest_report(
    ohlcv_path: str,
    trades_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    strategy_name: str = "Strategy",
    symbol: str = "BTCUSDT",
) -> str:
    """
    Convenience function to generate backtest report from files.

    Args:
        ohlcv_path: Path to OHLCV parquet file
        trades_path: Path to trades JSON file
        output_path: Path for output HTML report
        model_path: Path to ModelArtifact directory (optional)
        strategy_name: Strategy name for title
        symbol: Trading symbol

    Returns:
        Path to generated report
    """
    # Load data
    ohlcv_df = pd.read_parquet(ohlcv_path)

    with open(trades_path) as f:
        trades = json.load(f)

    # Load model if provided
    model_artifact = None
    if model_path:
        try:
            from src.time_series_model.strategies.models.model_artifact import (
                ModelArtifact,
            )

            model_artifact = ModelArtifact.load(Path(model_path))
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")

    # Generate report
    visualizer = BacktestVisualizer(
        ohlcv_df=ohlcv_df,
        trades=trades,
        model_artifact=model_artifact,
        strategy_name=strategy_name,
        symbol=symbol,
    )

    return visualizer.generate_report(output_path)
