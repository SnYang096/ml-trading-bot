"""Out-of-sample test using February 2025 data."""

import os
import json
import pickle
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ml_trading.data_tools.data_loader import MarketDataLoader
from ml_trading.data_tools.feature_engineering_wavelet import WaveletFeatureEngineer
from ml_trading.strategies.ml_strategy import MLTradingStrategy

# Paths
MODEL_PATH = os.path.join("models", "trained_model_january_2025.pkl")
SCALER_PATH = os.path.join("models", "feature_scalers_january_2025.pkl")
FEB_ZIP = r"D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-02.zip"
RESULTS_DIR = os.path.join("results", "february_2025_oos")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n" + "=" * 70)
print("📊 OOS Test: February 2025 (Out-of-Sample)")
print("=" * 70 + "\n")

# Load trained model
print("📦 Loading trained January 2025 model...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print("\n💡 Please train the model first:")
    print("   python scripts/train_january.py")
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

strategy = model_data["strategy"]
feature_engineer = model_data["feature_engineer"]
print("✅ Model loaded successfully")

# Load scalers (no refit)
print("📦 Loading feature scalers (frozen, no refit)...")
feature_engineer.load_scalers(SCALER_PATH)
print("✅ Scalers loaded")

# Extract and load February data
print(f"\n📦 Loading February 2025 data...")
if not os.path.exists(FEB_ZIP):
    print(f"❌ February data not found: {FEB_ZIP}")
    print("\n💡 Please download the data first using:")
    print("   .\\download_to_agg_data.ps1")
    sys.exit(1)

print(f"✅ Found: {FEB_ZIP}")
print(f"   File size: {os.path.getsize(FEB_ZIP) / (1024**3):.2f} GB")

temp_dir = os.path.join("data", "temp_extract_feb")
os.makedirs(temp_dir, exist_ok=True)

print("📦 Extracting February zip file...")
with zipfile.ZipFile(FEB_ZIP, "r") as z:
    z.extractall(temp_dir)

csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
if not csv_files:
    raise SystemExit("❌ No CSV found in February zip")

csv_path = os.path.join(temp_dir, csv_files[0])
print(f"✅ Extracted: {csv_path}")

# Load and prepare data
print("\n🔄 Processing February data...")
df = pd.read_csv(csv_path)

if "transact_time" in df.columns:
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
elif "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
else:
    raise SystemExit("❌ timestamp column not found")

df.set_index("timestamp", inplace=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df = df.dropna(subset=["price", "quantity"])

print(f"   Records: {len(df):,}")
print(f"   Date range: {df.index[0]} to {df.index[-1]}")

# Create 1-second OHLCV bars
print("🔄 Creating 1-second OHLCV bars...")
per_sec = df.groupby(pd.Grouper(freq="1s")).agg(
    {
        "price": ["first", "max", "min", "last"],
        "quantity": "sum",
        "is_buyer_maker": "mean",
    }
)
per_sec.columns = ["open", "high", "low", "close", "volume", "is_buyer_maker"]
per_sec = per_sec.dropna().ffill()

# Add microstructure features
print("📈 Computing microstructure features...")
per_sec["taker_buy"] = (~per_sec["is_buyer_maker"].round().astype(bool)).astype(int)
per_sec["buy_qty"] = per_sec["taker_buy"] * per_sec["volume"]
per_sec["sell_qty"] = (1 - per_sec["taker_buy"]) * per_sec["volume"]
per_sec["taker_buy_ratio"] = per_sec["buy_qty"] / (
    per_sec["buy_qty"] + per_sec["sell_qty"]
).replace(0, np.nan)
per_sec["taker_buy_ratio"] = per_sec["taker_buy_ratio"].fillna(0.5)
per_sec["cvd"] = (per_sec["buy_qty"] - per_sec["sell_qty"]).cumsum()

ohlc = per_sec[
    [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_qty",
        "sell_qty",
        "taker_buy_ratio",
        "cvd",
    ]
]
print(f"✅ Created {len(ohlc):,} 1-second bars")

# Create multi-timeframe data
print("\n🔄 Creating multi-timeframe data...")
mdl = MarketDataLoader()
mdl.raw_data = ohlc
mtf = mdl.get_multi_timeframe_data()
print(f"✅ Timeframes created: {list(mtf.keys())}")
for tf, data in mtf.items():
    print(f"   - {tf}: {len(data)} bars")

# Engineer features (no refit)
print("\n🔧 Engineering features with frozen scalers (no refit)...")
engineered_feb = feature_engineer.engineer_features(mtf, fit=False)
print("✅ Features engineered successfully")


def run_backtest(
    timeframe: str,
    stop_loss_pct: float,
    take_profit_pct: float,
    signal_threshold: float,
    min_gap_bars: int,
    initial_capital: float = 100000.0,
    risk_pct: float = 0.005,
    max_leverage: float = 2.0,
    max_adds: int = 2,
    add_risk_frac: float = 0.25,
    atr_trail_mult: float = 2.0,
    atr_stop_k: float = 1.5,
    max_concurrent: int = 1,
):
    """Run backtest with advanced risk management."""

    print(f"\n{'='*70}")
    print(f"🔍 Backtesting {timeframe}")
    print(f"{'='*70}")

    data_tf = engineered_feb[timeframe]
    feat_cols = [
        c
        for c in data_tf.columns
        if c not in ["open", "high", "low", "close", "volume"]
    ]
    X = data_tf[feat_cols].dropna()

    if X.empty:
        print(f"❌ [{timeframe}] No data available")
        return None

    # Get predictions
    s1 = strategy.pipeline.stage1_models[timeframe]
    s2 = strategy.pipeline.stage2_models[timeframe]
    p1 = s1.predict(X)
    p2 = s2.predict(X)

    sig = pd.DataFrame(
        {"stage1": p1, "stage2": p2, "close": data_tf.loc[X.index, "close"]},
        index=X.index,
    )

    # Generate signals
    sig["direction"] = 0
    long_th = 0.6 if timeframe == "15T" else 0.55
    short_th = 0.4 if timeframe == "15T" else 0.45
    sig.loc[sig["stage1"] > long_th, "direction"] = 1
    sig.loc[sig["stage1"] < short_th, "direction"] = -1
    sig["strength"] = np.abs(sig["stage1"] - 0.5)

    print(f"   Generated {len(sig)} signals")
    print(f"   Long signals: {(sig['direction'] == 1).sum()}")
    print(f"   Short signals: {(sig['direction'] == -1).sum()}")

    # Backtest simulation
    capital = initial_capital
    peak_equity = initial_capital
    max_drawdown = 0.0
    positions = []
    trades = []
    equity_curve = []
    last_signal_ts = None

    for ts, row in sig.iterrows():
        price = row["close"]

        # Update existing positions
        for pos in positions:
            if pos["status"] != "active":
                continue

            pnl = (
                (price - pos["entry"]) * pos["size"]
                if pos["side"] == "long"
                else (pos["entry"] - price) * pos["size"]
            )
            pos["pnl"] = pnl

            # Initialize stop loss
            if "stop" not in pos:
                atr = (
                    float(data_tf.loc[ts, "atr"]) if "atr" in data_tf.columns else None
                )
                if atr and atr > 0:
                    pos["stop"] = (
                        pos["entry"] - atr_stop_k * atr
                        if pos["side"] == "long"
                        else pos["entry"] + atr_stop_k * atr
                    )
                else:
                    pos["stop"] = (
                        pos["entry"] * (1.0 - stop_loss_pct)
                        if pos["side"] == "long"
                        else pos["entry"] * (1.0 + stop_loss_pct)
                    )

            # Check stop loss
            if pos["side"] == "long" and price <= pos["stop"]:
                close_pnl = (price - pos["entry"]) * pos["size"]
                trades.append(
                    {
                        **pos,
                        "exit": price,
                        "exit_ts": ts,
                        "reason": "stop",
                        "pnl": close_pnl,
                    }
                )
                capital += close_pnl
                pos["status"] = "closed"
            elif pos["side"] == "short" and price >= pos["stop"]:
                close_pnl = (pos["entry"] - price) * pos["size"]
                trades.append(
                    {
                        **pos,
                        "exit": price,
                        "exit_ts": ts,
                        "reason": "stop",
                        "pnl": close_pnl,
                    }
                )
                capital += close_pnl
                pos["status"] = "closed"

            # Check take profit
            reached_tp = (
                (price >= pos["entry"] * (1.0 + take_profit_pct))
                if pos["side"] == "long"
                else (price <= pos["entry"] * (1.0 - take_profit_pct))
            )
            if pos["status"] == "active" and reached_tp:
                close_pnl = (
                    (price - pos["entry"]) * pos["size"]
                    if pos["side"] == "long"
                    else (pos["entry"] - price) * pos["size"]
                )
                trades.append(
                    {
                        **pos,
                        "exit": price,
                        "exit_ts": ts,
                        "reason": "take_profit",
                        "pnl": close_pnl,
                    }
                )
                capital += close_pnl
                pos["status"] = "closed"

        # New signal
        if row["direction"] != 0 and row["strength"] > signal_threshold:
            minutes = (
                5
                if timeframe == "5T"
                else 15 if timeframe == "15T" else 60 if timeframe == "60T" else 240
            )
            active_count = sum(1 for p in positions if p["status"] == "active")

            if (
                last_signal_ts is None
                or (ts - last_signal_ts).total_seconds() / 60.0
                >= (min_gap_bars * minutes)
            ) and active_count < max_concurrent:
                entry = price
                atr = (
                    float(data_tf.loc[ts, "atr"]) if "atr" in data_tf.columns else None
                )

                if atr and atr > 0:
                    stop_price = (
                        (entry - atr_stop_k * atr)
                        if row["direction"] == 1
                        else (entry + atr_stop_k * atr)
                    )
                else:
                    stop_price = (
                        entry * (1.0 - stop_loss_pct)
                        if row["direction"] == 1
                        else entry * (1.0 + stop_loss_pct)
                    )

                stop_dist = abs(entry - stop_price)
                size = (capital * risk_pct) / stop_dist if stop_dist > 0 else 0.0

                if size > 0:
                    positions.append(
                        {
                            "side": "long" if row["direction"] == 1 else "short",
                            "size": size,
                            "entry": price,
                            "entry_ts": ts,
                            "status": "active",
                            "pnl": 0.0,
                        }
                    )
                    last_signal_ts = ts

        # Track equity
        open_pnl = sum(
            [p.get("pnl", 0.0) for p in positions if p["status"] == "active"]
        )
        equity = capital + open_pnl
        equity_curve.append(
            {
                "timestamp": ts,
                "equity": equity,
                "capital": capital,
                "open_pnl": open_pnl,
            }
        )

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd

    # Close remaining positions
    if len(sig) > 0:
        last_price = sig["close"].iloc[-1]
        last_ts = sig.index[-1]
        for pos in positions:
            if pos["status"] == "active":
                pnl = (
                    (last_price - pos["entry"]) * pos["size"]
                    if pos["side"] == "long"
                    else (pos["entry"] - last_price) * pos["size"]
                )
                trades.append(
                    {
                        **pos,
                        "exit": last_price,
                        "exit_ts": last_ts,
                        "reason": "end_of_data",
                        "pnl": pnl,
                    }
                )
                capital += pnl
                pos["status"] = "closed"

    # Calculate statistics
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    total_pnl = sum([t["pnl"] for t in trades])
    final_equity = capital

    results = {
        "timeframe": timeframe,
        "total_trades": len(trades),
        "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "total_pnl": float(total_pnl),
        "total_return": (final_equity - initial_capital) / initial_capital * 100.0,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": (
            (
                abs(sum([t["pnl"] for t in wins]) / sum([t["pnl"] for t in losses]))
                if losses and sum([t["pnl"] for t in losses]) != 0
                else float("inf")
            )
            if trades
            else 0.0
        ),
        "max_drawdown": max_drawdown * 100.0,
        "final_equity": float(final_equity),
        "initial_capital": initial_capital,
    }

    # Save results
    pd.DataFrame(trades).to_csv(
        os.path.join(RESULTS_DIR, f"{timeframe}_february_trades.csv"), index=False
    )
    pd.DataFrame(equity_curve).to_csv(
        os.path.join(RESULTS_DIR, f"{timeframe}_february_equity_curve.csv"), index=False
    )
    with open(
        os.path.join(RESULTS_DIR, f"{timeframe}_february_results.json"), "w"
    ) as f:
        json.dump(results, f, indent=2)

    # Print results
    print(f"\n📊 Results for {timeframe}:")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate']:.2f}%")
    print(f"   Total Return: {results['total_return']:.2f}%")
    print(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Final Equity: ${results['final_equity']:,.2f}")

    return results


# Run backtests for different timeframes
print("\n" + "=" * 70)
print("🚀 Running Backtests on February 2025 Data")
print("=" * 70)

all_results = {}

# 5-minute timeframe
res_5t = run_backtest(
    "5T",
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    signal_threshold=0.04,
    min_gap_bars=3,
    risk_pct=0.002,
    max_leverage=1.5,
)
if res_5t:
    all_results["5T"] = res_5t

# 15-minute timeframe
res_15t = run_backtest(
    "15T",
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    signal_threshold=0.04,
    min_gap_bars=3,
    risk_pct=0.002,
    max_leverage=1.5,
)
if res_15t:
    all_results["15T"] = res_15t

# 1-hour timeframe
res_60t = run_backtest(
    "60T",
    stop_loss_pct=0.03,
    take_profit_pct=0.06,
    signal_threshold=0.06,
    min_gap_bars=4,
    risk_pct=0.0025,
    max_leverage=2.0,
)
if res_60t:
    all_results["60T"] = res_60t

# Save summary
summary_path = os.path.join(RESULTS_DIR, "february_oos_summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 70)
print("✅ February 2025 OOS Test Completed!")
print("=" * 70)
print(f"\n📁 Results saved to: {RESULTS_DIR}")
print("\n📊 Summary:")
for tf, res in all_results.items():
    print(f"\n{tf}:")
    print(f"  Return: {res['total_return']:+.2f}%")
    print(f"  Max DD: {res['max_drawdown']:.2f}%")
    print(f"  Win Rate: {res['win_rate']:.2f}%")
    print(f"  Trades: {res['total_trades']}")

# Cleanup
import shutil

shutil.rmtree(temp_dir, ignore_errors=True)
print(f"\n🧹 Cleaned up temporary files")
print()
