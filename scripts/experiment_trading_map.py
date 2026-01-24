#!/usr/bin/env python3
"""
实验4：生成交易地图

从execution_log.jsonl提取交易信息，在K线图上标注交易事件，生成交互式图表和Grafana格式JSON。

使用方法:
    python scripts/experiment_trading_map.py \
        --exec-log results/pipeline_<run_id>/execution_log.jsonl \
        --feature-store-root feature_store \
        --feature-store-layer nnmh_highcap6_240T_2024_with_reflexivity \
        --data-path data/parquet_data \
        --out-dir results/experiments/trading_map
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available, will only generate JSON output")

from src.time_series_model.diagnostics.execution_log_aggregate import (
    aggregate_stage_logs,
)


def load_ohlc_data(
    symbol: str,
    feature_store_root: Path,
    feature_store_layer: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    data_path: Optional[Path] = None,
) -> pd.DataFrame:
    """加载OHLC数据"""
    # 优先从FeatureStore加载
    fs_path = feature_store_root / feature_store_layer / symbol / timeframe
    if fs_path.exists():
        frames = []
        for p in sorted(fs_path.glob("*.parquet")):
            df = pd.read_parquet(p)
            if "timestamp" not in df.columns:
                if df.index.name == "timestamp":
                    df = df.reset_index()
            if "open" in df.columns and "high" in df.columns:
                frames.append(
                    df[["timestamp", "open", "high", "low", "close", "volume"]]
                )

        if frames:
            ohlc_df = pd.concat(frames, ignore_index=True)
            ohlc_df["timestamp"] = pd.to_datetime(ohlc_df["timestamp"], utc=True)
            ohlc_df = ohlc_df.sort_values("timestamp")

            # 过滤日期范围
            start_ts = pd.to_datetime(start_date, utc=True)
            end_ts = pd.to_datetime(end_date, utc=True)
            ohlc_df = ohlc_df[
                (ohlc_df["timestamp"] >= start_ts) & (ohlc_df["timestamp"] <= end_ts)
            ]
            return ohlc_df

    # 如果FeatureStore没有，尝试从data_path加载
    if data_path:
        # 这里可以添加从parquet_data加载的逻辑
        pass

    raise ValueError(f"无法加载OHLC数据: {symbol}")


def extract_trades_from_logs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从execution logs提取交易信息"""
    trades = []
    current_positions = {}  # position_id -> position info

    for rec in records:
        execution = rec.get("execution") or {}
        gate = rec.get("gate") or {}
        returns_data = rec.get("returns") or {}

        if not execution.get("intent", False):
            continue

        symbol = rec.get("symbol")
        timestamp = rec.get("timestamp")
        archetype = gate.get("archetype") or execution.get("archetype")
        regime = rec.get("router", {}).get("mode") or "UNKNOWN"

        # 确定entry/exit
        # 这里简化处理，实际应该跟踪position状态
        # 假设每个execution intent是一个entry信号

        position_id = f"{symbol}_{timestamp}"

        # 获取return
        ret_mean = returns_data.get("ret_mean")
        ret_trend = returns_data.get("ret_trend")
        ret = ret_mean if ret_mean is not None else ret_trend

        # 确定side（简化，实际应该从preds或execution中获取）
        side = "LONG" if ret and ret > 0 else "SHORT"

        trade = {
            "position_id": position_id,
            "symbol": symbol,
            "entry_time": timestamp,
            "entry_price": None,  # 需要从OHLC数据中获取
            "exit_time": None,  # 需要跟踪exit
            "exit_price": None,
            "side": side,
            "archetype": archetype,
            "regime": regime,
            "entry_reason": "gate_allow" if not gate.get("blocked") else "gate_blocked",
            "exit_reason": None,
            "operations": [
                {
                    "type": "entry",
                    "time": timestamp,
                    "price": None,
                    "size": None,
                    "reason": f"{archetype}_{regime}",
                }
            ],
        }

        trades.append(trade)

    return trades


def generate_trading_map_json(
    trades: List[Dict[str, Any]],
    symbol: str,
    out_dir: Path,
) -> None:
    """生成Grafana格式的JSON"""
    grafana_data = {
        "symbol": symbol,
        "trades": trades,
    }

    json_path = out_dir / f"{symbol}_trading_map.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(grafana_data, f, indent=2, default=str)

    print(f"✅ Grafana JSON已生成: {json_path}")


def generate_plotly_chart(
    ohlc_df: pd.DataFrame,
    trades: List[Dict[str, Any]],
    symbol: str,
    out_dir: Path,
) -> None:
    """生成Plotly交互式图表"""
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly不可用，跳过图表生成")
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f"{symbol} - Trading Map", "Volume"),
        row_heights=[0.7, 0.3],
    )

    # K线图
    fig.add_trace(
        go.Candlestick(
            x=ohlc_df["timestamp"],
            open=ohlc_df["open"],
            high=ohlc_df["high"],
            low=ohlc_df["low"],
            close=ohlc_df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # 成交量
    colors = [
        "red" if c < o else "green" for c, o in zip(ohlc_df["close"], ohlc_df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=ohlc_df["timestamp"],
            y=ohlc_df["volume"],
            marker_color=colors,
            name="Volume",
            opacity=0.5,
        ),
        row=2,
        col=1,
    )

    # 添加交易标记
    for trade in trades:
        entry_time = pd.to_datetime(trade["entry_time"], utc=True)
        entry_price = trade.get("entry_price")

        if entry_price:
            # Entry marker
            entry_color = "green" if trade["side"] == "LONG" else "red"
            entry_symbol = "triangle-up" if trade["side"] == "LONG" else "triangle-down"

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
                    name=f"Entry {trade['side']}",
                    hovertemplate=f"<b>ENTRY {trade['side']}</b><br>"
                    f"Time: {entry_time}<br>"
                    f"Price: {entry_price:.2f}<br>"
                    f"Archetype: {trade['archetype']}<br>"
                    f"Regime: {trade['regime']}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        title=f"Trading Map: {symbol}",
        xaxis_rangeslider_visible=False,
        height=800,
    )

    html_path = out_dir / f"{symbol}_trading_map.html"
    fig.write_html(str(html_path))
    print(f"✅ Plotly图表已生成: {html_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成交易地图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exec-log",
        required=True,
        help="Execution log文件或目录",
    )
    parser.add_argument(
        "--feature-store-root",
        default="feature_store",
        help="FeatureStore根目录",
    )
    parser.add_argument(
        "--feature-store-layer",
        required=True,
        help="FeatureStore layer名称",
    )
    parser.add_argument(
        "--data-path",
        default="data/parquet_data",
        help="原始数据目录（备用）",
    )
    parser.add_argument(
        "--timeframe",
        default="240T",
        help="时间周期",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="开始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="输出目录",
    )

    args = parser.parse_args()

    exec_log_path = Path(args.exec_log)
    if not exec_log_path.exists():
        print(f"❌ Execution log不存在: {exec_log_path}")
        return 1

    print(f"📊 加载execution log: {exec_log_path}")
    if exec_log_path.is_dir():
        records = aggregate_stage_logs(exec_log_path)
    else:
        records = []
        with open(exec_log_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    print(f"✅ 加载了 {len(records)} 条记录")

    # 提取symbols
    symbols = sorted(set(r.get("symbol") for r in records if r.get("symbol")))
    print(f"📈 找到 {len(symbols)} 个symbols: {', '.join(symbols)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 为每个symbol生成交易地图
    for symbol in symbols:
        print(f"\n📊 处理 {symbol}...")

        # 提取该symbol的交易
        symbol_records = [r for r in records if r.get("symbol") == symbol]
        trades = extract_trades_from_logs(symbol_records)

        if len(trades) == 0:
            print(f"   ⚠️  {symbol} 没有交易")
            continue

        # 加载OHLC数据
        try:
            ohlc_df = load_ohlc_data(
                symbol=symbol,
                feature_store_root=Path(args.feature_store_root),
                feature_store_layer=args.feature_store_layer,
                timeframe=args.timeframe,
                start_date=args.start_date or "2024-01-01",
                end_date=args.end_date or "2024-12-31",
                data_path=Path(args.data_path) if args.data_path else None,
            )

            # 匹配交易价格
            ohlc_df["timestamp"] = pd.to_datetime(ohlc_df["timestamp"], utc=True)
            for trade in trades:
                entry_time = pd.to_datetime(trade["entry_time"], utc=True)
                # 找到最接近的K线
                closest_idx = ohlc_df["timestamp"].sub(entry_time).abs().idxmin()
                trade["entry_price"] = float(ohlc_df.loc[closest_idx, "close"])

            # 生成图表
            if PLOTLY_AVAILABLE:
                generate_plotly_chart(ohlc_df, trades, symbol, out_dir)

            # 生成JSON
            generate_trading_map_json(trades, symbol, out_dir)

        except Exception as e:
            print(f"   ❌ 处理 {symbol} 时出错: {e}")
            continue

    print(f"\n✅ 交易地图生成完成，输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
