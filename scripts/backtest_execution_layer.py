#!/usr/bin/env python3
"""
Execution Layer Backtest - 使用 archetypes/execution.yaml 参数计算 Sharpe

用法:
    python scripts/backtest_execution_layer.py \
        --logs results/train_final_xxx/bpc/predictions.parquet \
        --strategy bpc \
        --features-store-root feature_store \
        --features-store-layer bpc_highcap6_240T_v1 \
        --timeframe 240T

输出:
    - Sharpe Ratio (使用 execution.yaml 配置)
    - 与 backtest.yaml 配置的 Sharpe 对比
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore, FeatureStoreSpec


def load_execution_config(
    strategy: str, strategies_root: str = "config/strategies"
) -> Dict[str, Any]:
    """加载 archetypes/execution.yaml 配置"""
    exec_path = Path(strategies_root) / strategy / "archetypes" / "execution.yaml"
    if not exec_path.exists():
        raise FileNotFoundError(f"execution.yaml not found: {exec_path}")

    with open(exec_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gate_config(
    strategy: str, strategies_root: str = "config/strategies"
) -> Dict[str, Any]:
    """加载 archetypes/gate.yaml 配置"""
    gate_path = Path(strategies_root) / strategy / "archetypes" / "gate.yaml"
    if not gate_path.exists():
        return {}

    with open(gate_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulate_rr_execution(
    df: pd.DataFrame,
    exec_config: Dict[str, Any],
    atr_col: str = "atr",
) -> pd.Series:
    """
    使用 execution.yaml 的参数模拟 RR 回报

    支持的执行模式:
    - trailing stop (激活后逐步收紧)
    - fixed stop/take profit
    - time stop (超时平仓)

    策略:
    1. 如果有 forward_rr / ret_mean / bpc_impulse_return_atr 等实际收益列，用它们应用止损止盈逻辑
    2. 如果有 head_mfe_atr / head_mae_atr，用它们模拟
    3. 否则用 OHLC 简化模拟
    """
    stop_loss_cfg = exec_config.get("stop_loss", {})
    take_profit_cfg = exec_config.get("take_profit", {})
    holding_cfg = exec_config.get("holding", {})

    # 解析参数
    stop_type = stop_loss_cfg.get("type", "fixed")
    initial_r = float(stop_loss_cfg.get("initial_r", 2.0))

    trailing_cfg = stop_loss_cfg.get("trailing", {})
    activation_r = float(trailing_cfg.get("activation_r", 1.0))
    trail_r = float(trailing_cfg.get("trail_r", 1.5))
    step_r = float(trailing_cfg.get("step_r", 0.5))

    tp_enabled = take_profit_cfg.get("enabled", False)
    max_holding_bars = holding_cfg.get("max_holding_bars") or 50
    time_stop_bars = holding_cfg.get("time_stop_bars") or max_holding_bars

    # 检测可用的实际收益列 (优先级高 -> 低)
    actual_rr_col = None
    for col in ["forward_rr", "ret_mean", "ret_trend", "bpc_impulse_return_atr"]:
        if col in df.columns:
            actual_rr_col = col
            break

    # 检测可用的方向列
    dir_col = None
    for col in ["head_dir_score", "bpc_breakout_direction", "direction"]:
        if col in df.columns:
            dir_col = col
            break

    # 检测可用的 MFE/MAE 列
    mfe_col = "head_mfe_atr" if "head_mfe_atr" in df.columns else None
    mae_col = "head_mae_atr" if "head_mae_atr" in df.columns else None

    if actual_rr_col:
        print(f"   🎯 Using actual RR from: {actual_rr_col}")
        print(
            f"   📈 Applying execution constraints: stop_r={initial_r}, type={stop_type}"
        )

        # 使用实际收益，应用止损逻辑
        actual_rr = df[actual_rr_col].values
        results = []

        for i, rr in enumerate(actual_rr):
            if pd.isna(rr):
                results.append(np.nan)
                continue

            # 应用止损逻辑
            if rr <= -initial_r:
                # 触发止损
                realized_r = -initial_r
            elif stop_type == "trailing" and rr >= activation_r:
                # 激活移动止损，收紧到 trail_r
                realized_r = max(rr - trail_r, -initial_r)
            elif tp_enabled:
                tp_r = float(take_profit_cfg.get("target_r", 2.0))
                if rr >= tp_r:
                    realized_r = tp_r
                else:
                    realized_r = rr
            else:
                realized_r = rr

            results.append(float(realized_r))

        return pd.Series(results, index=df.index)

    # 备用方案: 使用 MFE/MAE 或 OHLC 模拟
    required = ["high", "low", "close", atr_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns for execution simulation: {missing}")
        return pd.Series(np.nan, index=df.index)

    if dir_col is None:
        print(f"⚠️  No direction column found. Available: {list(df.columns)[:20]}...")
        return pd.Series(np.nan, index=df.index)

    print(f"   Using direction from: {dir_col}")
    if mfe_col and mae_col:
        print(f"   Using MFE/MAE from: {mfe_col}, {mae_col}")
    else:
        print(f"   Simulating MFE/MAE from OHLC (simplified)")

    results = []

    for idx, row in df.iterrows():
        atr = row[atr_col]
        if pd.isna(atr) or atr <= 0:
            results.append(np.nan)
            continue

        # 方向判断
        dir_score = row.get(dir_col, 0)
        if pd.isna(dir_score):
            results.append(0.0)
            continue

        # 根据列类型判断方向
        if dir_col == "bpc_breakout_direction":
            sign = int(dir_score)
        else:
            sign = 1 if float(dir_score) > 0 else (-1 if float(dir_score) < 0 else 0)

        if sign == 0:
            results.append(0.0)
            continue

        entry_price = row["close"]

        # 模拟持仓期间的收益
        if mfe_col and mae_col:
            mfe = float(row.get(mfe_col, 0) or 0)
            mae = float(row.get(mae_col, 0) or 0)
        else:
            if sign > 0:
                mfe = (row["high"] - entry_price) / atr if atr > 0 else 0
                mae = (entry_price - row["low"]) / atr if atr > 0 else 0
            else:
                mfe = (entry_price - row["low"]) / atr if atr > 0 else 0
                mae = (row["high"] - entry_price) / atr if atr > 0 else 0

        stop_r = initial_r

        if stop_type == "trailing":
            if mfe >= activation_r:
                effective_profit = mfe - trail_r
                if effective_profit > 0:
                    realized_r = min(effective_profit, mfe)
                else:
                    realized_r = -stop_r if mae >= stop_r else effective_profit
            else:
                if mae >= stop_r:
                    realized_r = -stop_r
                else:
                    realized_r = mfe - mae if mfe > mae else -mae
        else:
            if mae >= stop_r:
                realized_r = -stop_r
            elif tp_enabled:
                tp_r = float(take_profit_cfg.get("target_r", 2.0))
                if mfe >= tp_r:
                    realized_r = tp_r
                else:
                    realized_r = mfe - mae
            else:
                realized_r = mfe - mae

        results.append(float(realized_r))

    return pd.Series(results, index=df.index)


def compute_sharpe(returns: pd.Series, annualize: bool = False) -> float:
    """计算 Sharpe Ratio"""
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    mean_r = returns.mean()
    std_r = returns.std(ddof=1)

    if std_r < 1e-8:
        return 0.0

    sharpe = mean_r / std_r

    if annualize:
        # 假设 4H 时间框架，每年约 6*365 = 2190 个 bar
        sharpe *= np.sqrt(2190)

    return float(sharpe)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Execution Layer Backtest with archetypes/execution.yaml"
    )
    p.add_argument(
        "--logs",
        required=True,
        help="Input logs file (predictions.parquet or logs_gated.parquet)",
    )
    p.add_argument("--strategy", required=True, help="Strategy name (e.g., bpc)")
    p.add_argument("--strategies-root", default="config/strategies")
    p.add_argument("--features-store-root", default="feature_store")
    p.add_argument("--features-store-layer", required=True)
    p.add_argument("--timeframe", default="240T")
    p.add_argument(
        "--filter-allowed",
        action="store_true",
        help="Only include gate_decision=allow rows",
    )
    args = p.parse_args()

    print("=" * 80)
    print("🎯 Execution Layer Backtest")
    print("=" * 80)

    # 加载 execution.yaml 配置
    try:
        exec_config = load_execution_config(args.strategy, args.strategies_root)
        print(f"\n📋 Loaded execution.yaml for '{args.strategy}':")
        stop_loss = exec_config.get("stop_loss", {})
        print(f"   Stop Loss Type: {stop_loss.get('type', 'fixed')}")
        print(f"   Initial R: {stop_loss.get('initial_r', 2.0)}")
        if stop_loss.get("type") == "trailing":
            trailing = stop_loss.get("trailing", {})
            print(f"   Trailing Activation: {trailing.get('activation_r', 1.0)}R")
            print(f"   Trail Distance: {trailing.get('trail_r', 1.5)}R")
    except Exception as e:
        print(f"❌ Failed to load execution.yaml: {e}")
        return 1

    # 读取 logs 文件
    logs_path = Path(args.logs)
    if not logs_path.exists():
        print(f"❌ Logs file not found: {logs_path}")
        return 1

    df = pd.read_parquet(logs_path)
    print(f"\n📂 Loaded logs: {len(df)} rows")

    # 处理列名兼容
    if "_symbol" in df.columns and "symbol" not in df.columns:
        df["symbol"] = df["_symbol"]

    # 过滤 gate 通过的样本
    if args.filter_allowed:
        if "gate_decision" in df.columns:
            df = df[df["gate_decision"] == "allow"]
            print(f"   After filtering gate=allow: {len(df)} rows")
        elif "gate_ok" in df.columns:
            df = df[df["gate_ok"] == True]
            print(f"   After filtering gate_ok=True: {len(df)} rows")

    if len(df) == 0:
        print("❌ No data after filtering")
        return 1

    # 读取 FeatureStore 获取 OHLC 和 ATR
    symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
    if not symbols:
        print("❌ No symbols found in logs")
        return 1

    print(f"\n📊 Symbols: {', '.join(symbols)}")

    # 读取 FeatureStore
    store = FeatureStore(args.features_store_root)
    parts = []
    for sym in symbols:
        spec = FeatureStoreSpec(
            layer=args.features_store_layer, symbol=sym, timeframe=args.timeframe
        )
        try:
            df_sym = store.read_range(
                spec, start=pd.Timestamp("1970-01-01"), end=pd.Timestamp("2100-01-01")
            )
            if not df_sym.empty:
                if "symbol" not in df_sym.columns:
                    df_sym = df_sym.copy()
                    df_sym["symbol"] = sym
                parts.append(df_sym)
        except Exception as e:
            print(f"   ⚠️  Failed to read {sym}: {e}")

    if not parts:
        print("❌ No FeatureStore data loaded")
        return 1

    feats = pd.concat(parts, axis=0, ignore_index=False)
    print(f"   Loaded FeatureStore: {len(feats)} rows")

    # 准备 timestamp 列 (处理 index 和 column 的冲突)
    feats = feats.copy()
    if "timestamp" in feats.columns and feats.index.name == "timestamp":
        feats = feats.reset_index(drop=True)  # 保留列，丢弃 index
    elif "timestamp" not in feats.columns:
        if isinstance(feats.index, pd.DatetimeIndex):
            feats["timestamp"] = feats.index
            feats = feats.reset_index(drop=True)
        elif feats.index.name == "timestamp":
            feats = feats.reset_index()

    df = df.copy()
    # 处理没有 timestamp 列的情况
    if "timestamp" in df.columns and df.index.name == "timestamp":
        df = df.reset_index(drop=True)  # 保留列，丢弃 index
    elif "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = df.index
            df = df.reset_index(drop=True)
        elif df.index.name == "timestamp":
            df = df.reset_index()
        else:
            # 尝试从 FeatureStore 数据通过 index 匹配
            print(
                f"   ⚠️ No timestamp column in logs. Will skip FeatureStore merge and use available columns."
            )
            # 设置一个虚拟的 timestamp
            df["timestamp"] = pd.NaT

    # Merge 获取 OHLC 和 ATR
    feats["symbol"] = feats["symbol"].astype(str)
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], errors="coerce")
    df["symbol"] = (
        df["symbol"].astype(str)
        if "symbol" in df.columns
        else df.get("_symbol", "UNKNOWN").astype(str)
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 检查 df 中是否已经有 OHLC 和 atr
    has_ohlc = all(c in df.columns for c in ["high", "low", "close", "atr"])

    if has_ohlc:
        # 已经有 OHLC，无需 merge
        merged = df
        print(f"\n🔄 Using existing OHLC/ATR from logs: {len(merged)} rows")
    elif df["timestamp"].isna().all():
        # 没有 timestamp，无法 merge，但如果 df 中有 OHLC 则直接使用
        print(
            f"\n⚠️ Cannot merge with FeatureStore (no timestamp). Using logs data directly."
        )
        merged = df
    else:
        merged = df.merge(
            feats[["symbol", "timestamp", "high", "low", "close", "atr"]],
            on=["symbol", "timestamp"],
            how="left",
            suffixes=("", "_fs"),
        )
        print(f"\n🔄 Merged data: {len(merged)} rows")

    # 使用 FeatureStore 的 OHLC（如果 logs 里没有）
    for col in ["high", "low", "close", "atr"]:
        fs_col = f"{col}_fs"
        if fs_col in merged.columns:
            if col not in merged.columns or merged[col].isna().all():
                merged[col] = merged[fs_col]
            merged = merged.drop(columns=[fs_col])

    print(f"\n🔄 Merged data: {len(merged)} rows")

    # 使用 execution.yaml 配置模拟 RR
    print("\n📈 Simulating with execution.yaml config...")
    exec_returns = simulate_rr_execution(merged, exec_config, atr_col="atr")

    valid_returns = exec_returns.dropna()
    if len(valid_returns) == 0:
        print("❌ No valid returns computed")
        return 1

    # 计算 Sharpe
    exec_sharpe = compute_sharpe(valid_returns, annualize=False)
    exec_sharpe_ann = compute_sharpe(valid_returns, annualize=True)

    print("\n" + "=" * 80)
    print("📊 EXECUTION LAYER BACKTEST RESULTS")
    print("=" * 80)
    print(f"\n   Trades: {len(valid_returns)}")
    print(f"   Mean R: {valid_returns.mean():.4f}")
    print(f"   Std R:  {valid_returns.std():.4f}")
    print(f"   Win Rate: {(valid_returns > 0).mean():.2%}")
    print(f"\n   Sharpe (raw): {exec_sharpe:.4f}")
    print(f"   Sharpe (annualized): {exec_sharpe_ann:.4f}")

    # 对比原始 RR 列（如果存在）
    for rr_col in ["forward_rr", "ret_mean", "ret_trend", "bpc_impulse_return_atr"]:
        if rr_col in merged.columns:
            orig_returns = merged[rr_col].dropna()
            if len(orig_returns) > 0:
                orig_sharpe = compute_sharpe(orig_returns, annualize=False)
                orig_mean = orig_returns.mean()
                print(f"\n   📌 Original {rr_col}:")
                print(f"      Mean: {orig_mean:.4f}")
                print(f"      Sharpe (raw): {orig_sharpe:.4f}")
                print(
                    f"      → Delta: {'+' if exec_sharpe > orig_sharpe else ''}{exec_sharpe - orig_sharpe:.4f}"
                )
                break

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
