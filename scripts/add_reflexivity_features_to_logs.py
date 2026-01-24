#!/usr/bin/env python3
"""
为现有的执行日志添加反身性特征

从现有的pipeline输出（preds, logs）中计算反身性特征，并添加到执行日志中。

使用方法:
    python scripts/add_reflexivity_features_to_logs.py \
        --preds results/pipeline_output/preds \
        --logs results/pipeline_output/logs_3action.parquet \
        --data-path data/parquet_data \
        --timeframe 240T \
        --output results/pipeline_with_reflexivity/exec_logs
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.time_series_model.diagnostics.execution_log import ExecutionStageLogWriter
from src.time_series_model.diagnostics.execution_log_canonical import (
    load_pipeline_inputs,
    build_stage_logs_from_pipeline,
)
from src.features.time_series.reflexivity_features import (
    compute_ofci_from_trades,
    compute_ofci_pct_from_series,
    compute_shd_from_ohlcv,
    compute_shd_pct_from_series,
)


def load_tick_data(
    data_path: Path, symbol: str, start_date: str, end_date: str
) -> Optional[pd.DataFrame]:
    """加载tick数据用于计算OFCI"""
    # 尝试从parquet数据目录加载tick数据
    tick_files = list((data_path / symbol).glob("*.parquet"))
    if not tick_files:
        return None

    # 加载并过滤日期范围
    dfs = []
    for f in tick_files:
        try:
            df = pd.read_parquet(f)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
                if len(df) > 0:
                    dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True).sort_values("timestamp")


def compute_reflexivity_features_for_symbol(
    symbol: str,
    logs_df: pd.DataFrame,
    data_path: Path,
    timeframe: str,
) -> pd.DataFrame:
    """
    为指定symbol计算反身性特征

    Returns:
        DataFrame with ofci_pct and shd_pct columns, indexed by timestamp
    """
    symbol_logs = logs_df[logs_df.get("symbol", logs_df.index) == symbol].copy()
    if len(symbol_logs) == 0:
        return pd.DataFrame()

    # 确保有timestamp列
    if "timestamp" not in symbol_logs.columns:
        if logs_df.index.name == "timestamp":
            symbol_logs["timestamp"] = symbol_logs.index
        else:
            return pd.DataFrame()

    # 统一时间戳类型为UTC，避免类型不匹配
    symbol_logs["timestamp"] = pd.to_datetime(symbol_logs["timestamp"], utc=True)
    # 确保index也是UTC时区
    symbol_logs = symbol_logs.set_index("timestamp")
    symbol_logs.index = (
        symbol_logs.index.tz_localize(None)
        if symbol_logs.index.tz is None
        else symbol_logs.index.tz_convert("UTC")
    )

    start_date = symbol_logs.index.min()
    end_date = symbol_logs.index.max()

    # 计算SHD（需要OHLCV和CVD数据）
    shd_pct_series = pd.Series(index=symbol_logs.index, dtype=float)

    if "close" in symbol_logs.columns and "cvd" in symbol_logs.columns:
        try:
            # 计算SHD
            shd_result = compute_shd_from_ohlcv(
                df=symbol_logs[["close", "cvd"]],
                window=60,
            )
            if "shd" in shd_result.columns:
                shd_series = shd_result["shd"]
                # 确保index时区一致
                if shd_series.index.tz is not None:
                    shd_series.index = shd_series.index.tz_convert("UTC")
                elif symbol_logs.index.tz is not None:
                    shd_series.index = shd_series.index.tz_localize("UTC")

                # 计算SHD percentile
                shd_pct_result = compute_shd_pct_from_series(
                    shd=shd_series,
                    window=288,
                    shift=1,
                )
                if "shd_pct" in shd_pct_result.columns:
                    shd_pct_series = shd_pct_result["shd_pct"]
                    # 确保index时区一致
                    if shd_pct_series.index.tz is not None:
                        shd_pct_series.index = shd_pct_series.index.tz_convert("UTC")
                    elif symbol_logs.index.tz is not None:
                        shd_pct_series.index = shd_pct_series.index.tz_localize("UTC")
        except Exception as e:
            print(f"Warning: Failed to compute SHD for {symbol}: {e}")

    # 计算OFCI（需要tick数据）
    ofci_pct_series = pd.Series(index=symbol_logs.index, dtype=float)

    tick_data = load_tick_data(
        data_path, symbol, str(start_date.date()), str(end_date.date())
    )
    if tick_data is not None and "side" in tick_data.columns:
        try:
            # 确保tick_data的timestamp也是UTC
            if "timestamp" in tick_data.columns:
                tick_data["timestamp"] = pd.to_datetime(
                    tick_data["timestamp"], utc=True
                )
                tick_data = tick_data.set_index("timestamp")
                if tick_data.index.tz is not None:
                    tick_data.index = tick_data.index.tz_convert("UTC")

            # 计算OFCI
            ofci_result = compute_ofci_from_trades(
                trades=tick_data,
                window=100,
            )
            if "ofci" in ofci_result.columns:
                ofci_series = ofci_result["ofci"]
                # 确保index时区一致
                if ofci_series.index.tz is not None:
                    ofci_series.index = ofci_series.index.tz_convert("UTC")
                elif symbol_logs.index.tz is not None:
                    ofci_series.index = ofci_series.index.tz_localize("UTC")

                # 计算OFCI percentile
                ofci_pct_result = compute_ofci_pct_from_series(
                    ofci=ofci_series,
                    window=288,
                    shift=1,
                )
                if "ofci_pct" in ofci_pct_result.columns:
                    ofci_pct_series = ofci_pct_result["ofci_pct"]
                    # 确保index时区一致
                    if ofci_pct_series.index.tz is not None:
                        ofci_pct_series.index = ofci_pct_series.index.tz_convert("UTC")
                    elif symbol_logs.index.tz is not None:
                        ofci_pct_series.index = ofci_pct_series.index.tz_localize("UTC")
        except Exception as e:
            print(f"Warning: Failed to compute OFCI for {symbol}: {e}")

    # 合并结果，确保所有index都是UTC时区
    result = pd.DataFrame(
        {
            "ofci_pct": ofci_pct_series,
            "shd_pct": shd_pct_series,
        },
        index=symbol_logs.index,
    )

    # 确保result的index时区一致
    if result.index.tz is None and symbol_logs.index.tz is not None:
        result.index = result.index.tz_localize("UTC")
    elif result.index.tz is not None and symbol_logs.index.tz is None:
        result.index = result.index.tz_localize(None)

    return result


def load_reflexivity_from_featurestore(
    logs_df: pd.DataFrame,
    feature_store_dir: str,
    feature_store_layer: str,
    timeframe: str,
) -> Dict[str, pd.DataFrame]:
    """从FeatureStore加载反身性特征"""
    from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec

    store = FeatureStore(feature_store_dir)
    all_reflexivity = {}

    symbols = logs_df["symbol"].unique() if "symbol" in logs_df.columns else []
    if len(symbols) == 0:
        return all_reflexivity

    for symbol in symbols:
        try:
            spec = FeatureStoreSpec(
                layer=feature_store_layer,
                symbol=str(symbol),
                timeframe=timeframe,
            )

            # 获取时间范围
            symbol_logs = (
                logs_df[logs_df["symbol"] == symbol]
                if "symbol" in logs_df.columns
                else logs_df
            )
            if len(symbol_logs) == 0:
                continue

            if "timestamp" in symbol_logs.columns:
                timestamps = pd.to_datetime(symbol_logs["timestamp"], utc=True)
            elif isinstance(symbol_logs.index, pd.DatetimeIndex):
                timestamps = symbol_logs.index
            else:
                continue

            start_ts = timestamps.min()
            end_ts = timestamps.max()

            # 从FeatureStore读取特征
            df_features = store.read_range(
                spec=spec,
                start=start_ts,
                end=end_ts,
            )

            if df_features is not None and len(df_features) > 0:
                # 提取反身性特征
                reflexivity_cols = ["ofci_pct", "shd_pct"]
                available_cols = [
                    c for c in reflexivity_cols if c in df_features.columns
                ]
                if available_cols:
                    reflexivity_df = df_features[available_cols].copy()
                    all_reflexivity[str(symbol)] = reflexivity_df
                    print(
                        f"✅ Loaded reflexivity features for {symbol}: {len(reflexivity_df)} rows"
                    )
        except Exception as e:
            print(f"⚠️ Failed to load reflexivity from FeatureStore for {symbol}: {e}")
            continue

    return all_reflexivity


def add_reflexivity_to_stage_logs(
    preds_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    data_path: Path,
    timeframe: str,
    out_dir: Path,
    run_id: Optional[str] = None,
    strategy_name: str = "pipeline",
    feature_store_dir: Optional[str] = None,
    feature_store_layer: Optional[str] = None,
) -> None:
    """为stage logs添加反身性特征"""
    # 优先从FeatureStore加载反身性特征，如果没有则重新计算
    all_reflexivity = {}
    if feature_store_dir and feature_store_layer:
        try:
            all_reflexivity = load_reflexivity_from_featurestore(
                logs_df=logs_df,
                feature_store_dir=feature_store_dir,
                feature_store_layer=feature_store_layer,
                timeframe=timeframe,
            )
            print(
                f"✅ Loaded reflexivity features from FeatureStore for {len(all_reflexivity)} symbols"
            )
        except Exception as e:
            print(f"⚠️ Failed to load from FeatureStore, will compute: {e}")
            all_reflexivity = {}

    # 如果FeatureStore加载失败，回退到重新计算
    if not all_reflexivity:
        # 获取所有symbols
        symbols = (
            logs_df.get("symbol", logs_df.index).unique()
            if hasattr(logs_df.get("symbol", logs_df.index), "unique")
            else [
                (
                    logs_df.get("symbol", "UNKNOWN").iloc[0]
                    if len(logs_df) > 0
                    else "UNKNOWN"
                )
            ]
        )

        if isinstance(symbols, (str, int)):
            symbols = [symbols]

        # 为每个symbol计算反身性特征
        for symbol in symbols:
            reflexivity_df = compute_reflexivity_features_for_symbol(
                symbol=str(symbol),
                logs_df=logs_df,
                data_path=data_path,
                timeframe=timeframe,
            )
            if len(reflexivity_df) > 0:
                all_reflexivity[str(symbol)] = reflexivity_df

    # 构建stage logs
    records = build_stage_logs_from_pipeline(
        preds_df=preds_df,
        mode_df=None,
        logs_df=logs_df,
        run_id=run_id,
        timeframe=timeframe,
        strategy_name=strategy_name,
    )

    # 为每个record添加反身性特征并创建features stage
    writers = {}
    from src.time_series_model.diagnostics.execution_log import build_stage_record

    for rec in records:
        stage = rec.get("stage")
        if not stage:
            continue

        # 写入原始stage log
        writer = writers.get(stage)
        if writer is None:
            writer = ExecutionStageLogWriter(base_dir=out_dir, stage=str(stage))
            writers[stage] = writer
        writer.write(rec, decision_ts_ns=int(rec.get("decision_ts_ns", 0)))

        # 为每个decision创建features stage记录
        symbol = rec.get("symbol", "")
        timestamp_str = rec.get("timestamp", "")
        decision_id = rec.get("decision_id", "")
        decision_ts_ns = rec.get("decision_ts_ns", 0)

        if symbol and timestamp_str:
            # 准备features数据
            features_data = {}

            # 添加反身性特征
            if symbol in all_reflexivity:
                try:
                    # 统一时间戳类型为UTC
                    timestamp = pd.to_datetime(timestamp_str, utc=True)
                    reflexivity_df = all_reflexivity[symbol]

                    # 确保reflexivity_df的index也是UTC时区
                    if reflexivity_df.index.tz is None:
                        reflexivity_df.index = reflexivity_df.index.tz_localize("UTC")
                    elif reflexivity_df.index.tz != timestamp.tz:
                        reflexivity_df.index = reflexivity_df.index.tz_convert("UTC")

                    # 找到最接近的时间戳
                    if timestamp in reflexivity_df.index:
                        ofci_pct = float(reflexivity_df.loc[timestamp, "ofci_pct"])
                        shd_pct = float(reflexivity_df.loc[timestamp, "shd_pct"])
                    else:
                        # 使用前向填充，但需要确保时区一致
                        try:
                            reflexivity_aligned = reflexivity_df.reindex(
                                [timestamp], method="ffill"
                            )
                            if (
                                len(reflexivity_aligned) > 0
                                and not reflexivity_aligned.isna().all().all()
                            ):
                                ofci_pct = (
                                    float(reflexivity_aligned.iloc[0]["ofci_pct"])
                                    if not pd.isna(
                                        reflexivity_aligned.iloc[0]["ofci_pct"]
                                    )
                                    else 0.0
                                )
                                shd_pct = (
                                    float(reflexivity_aligned.iloc[0]["shd_pct"])
                                    if not pd.isna(
                                        reflexivity_aligned.iloc[0]["shd_pct"]
                                    )
                                    else 0.0
                                )
                            else:
                                ofci_pct = 0.0
                                shd_pct = 0.0
                        except Exception:
                            # 如果reindex失败，尝试使用最近的值
                            try:
                                nearest_idx = reflexivity_df.index.get_indexer(
                                    [timestamp], method="nearest"
                                )[0]
                                if nearest_idx >= 0:
                                    ofci_pct = float(
                                        reflexivity_df.iloc[nearest_idx]["ofci_pct"]
                                    )
                                    shd_pct = float(
                                        reflexivity_df.iloc[nearest_idx]["shd_pct"]
                                    )
                                else:
                                    ofci_pct = 0.0
                                    shd_pct = 0.0
                            except Exception:
                                ofci_pct = 0.0
                                shd_pct = 0.0

                    features_data["ofci_pct"] = ofci_pct
                    features_data["shd_pct"] = shd_pct
                except Exception as e:
                    print(
                        f"Warning: Failed to add reflexivity to {symbol} at {timestamp_str}: {e}"
                    )
                    features_data["ofci_pct"] = 0.0
                    features_data["shd_pct"] = 0.0
            else:
                features_data["ofci_pct"] = 0.0
                features_data["shd_pct"] = 0.0

            # 创建features stage记录
            features_rec = build_stage_record(
                stage="features",
                decision_id=decision_id,
                decision_ts_ns=decision_ts_ns,
                source="pipeline",
                run_id=rec.get("run_id"),
                symbol=symbol,
                timeframe=rec.get("timeframe"),
                strategy_name=rec.get("strategy_name"),
                instrument_id=rec.get("instrument_id"),
                data=features_data if features_data else None,
            )

            # 写入features stage log
            features_writer = writers.get("features")
            if features_writer is None:
                features_writer = ExecutionStageLogWriter(
                    base_dir=out_dir, stage="features"
                )
                writers["features"] = features_writer
            features_writer.write(features_rec, decision_ts_ns=decision_ts_ns)


def main():
    parser = argparse.ArgumentParser(
        description="Add reflexivity features to existing pipeline logs"
    )
    parser.add_argument("--preds", required=True, help="preds file/dir")
    parser.add_argument("--logs", required=True, help="logs_3action file/dir")
    parser.add_argument(
        "--data-path",
        default="data/parquet_data",
        help="Raw data directory for tick data",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output base dir for stage logs"
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timeframe", default="240T")
    parser.add_argument("--strategy-name", default="pipeline")
    parser.add_argument(
        "--feature-store-dir",
        type=str,
        default=None,
        help="FeatureStore root directory (if provided, will load from FeatureStore instead of computing)",
    )
    parser.add_argument(
        "--feature-store-layer",
        type=str,
        default=None,
        help="FeatureStore layer name (required if --feature-store-dir is provided)",
    )

    args = parser.parse_args()

    # 加载pipeline输入
    preds_df, mode_df, logs_df = load_pipeline_inputs(
        Path(args.preds),
        None,  # mode not needed
        Path(args.logs),
    )

    print(f"Loaded {len(preds_df)} preds, {len(logs_df)} logs")

    # 添加反身性特征并生成stage logs
    add_reflexivity_to_stage_logs(
        preds_df=preds_df,
        logs_df=logs_df,
        data_path=Path(args.data_path),
        timeframe=args.timeframe,
        out_dir=Path(args.out_dir),
        run_id=args.run_id,
        strategy_name=args.strategy_name,
        feature_store_dir=args.feature_store_dir,
        feature_store_layer=args.feature_store_layer,
    )

    print(f"✅ Stage logs with reflexivity features saved to {args.out_dir}")


if __name__ == "__main__":
    main()
