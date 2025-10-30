import os, json, pickle, zipfile, shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from datetime import datetime
from ml_trading.data_tools.data_loader import MarketDataLoader
from ml_trading.strategies.ml_strategy import MLTradingStrategy

DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME",
                                    "trained_model_btcusdt_20250501_20250531")
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.environ.get("MODEL_DIR", "models"),
                 f"{DEFAULT_MODEL_NAME}.pkl"),
)
SCALER_PATH = os.environ.get(
    "SCALER_PATH",
    os.path.join(
        os.environ.get("MODEL_DIR", "models"),
        f"{DEFAULT_MODEL_NAME}_scalers.pkl",
    ),
)
JUNE_DATA = os.environ.get(
    "OOS_DATA",
    os.path.join("data", "parquet_data", "BTCUSDT-aggTrades-2025-06.parquet"),
)
RESULTS_DIR = os.path.join("results", "june_2025_oos")
REPORTS_DIR = "reports"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_trade_dataframe(
        data_path: str) -> tuple[pd.DataFrame, Optional[Path]]:
    path = Path(data_path)
    suffix = path.suffix.lower()
    temp_dir: Optional[Path] = None

    if suffix == ".zip":
        temp_dir = path.parent / f"temp_extract_{path.stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(path, "r") as z:
            z.extractall(temp_dir)

        csv_files = list(temp_dir.glob("*.csv"))
        if not csv_files:
            raise SystemExit("No CSV found in provided ZIP archive")

        df = pd.read_csv(csv_files[0])
        print(f"Extracted CSV file: {csv_files[0]}")
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise SystemExit(
            f"Unsupported file extension '{suffix}'. Expected .zip, .csv, or .parquet"
        )

    print(f"Raw OOS data shape: {df.shape}")
    return df, temp_dir


print(f"Loading trained model from {MODEL_PATH}...")
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)
strategy = model_data["strategy"]
feature_engineer = model_data["feature_engineer"]
print(f"Loading scalers from {SCALER_PATH} (no refit)...")
feature_engineer.load_scalers(SCALER_PATH)

print(f"Loading OOS data from {JUNE_DATA} and preparing OHLCV...")
if not os.path.exists(JUNE_DATA):
    raise SystemExit(f"Data file not found: {JUNE_DATA}")

raw_df, temp_dir = load_trade_dataframe(JUNE_DATA)

try:
    df = raw_df.copy()

    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise SystemExit("timestamp column not found")

    df.set_index("timestamp", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    if "is_buyer_maker" in df.columns:
        df["is_buyer_maker"] = pd.to_numeric(df["is_buyer_maker"],
                                             errors="coerce").fillna(0.5)
    else:
        df["is_buyer_maker"] = 0.5

    df = df.dropna(subset=["price", "quantity"])

    per_sec = df.groupby(pd.Grouper(freq="1s")).agg({
        "price": ["first", "max", "min", "last"],
        "quantity":
        "sum",
        "is_buyer_maker":
        "mean",
    })
    per_sec.columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_buyer_maker",
    ]
    per_sec = per_sec.dropna().ffill()
    per_sec["taker_buy"] = (
        ~per_sec["is_buyer_maker"].round().astype(bool)).astype(int)
    per_sec["buy_qty"] = per_sec["taker_buy"] * per_sec["volume"]
    per_sec["sell_qty"] = (1 - per_sec["taker_buy"]) * per_sec["volume"]
    per_sec["taker_buy_ratio"] = per_sec["buy_qty"] / (
        per_sec["buy_qty"] + per_sec["sell_qty"]).replace(0, np.nan)
    per_sec["taker_buy_ratio"] = per_sec["taker_buy_ratio"].fillna(0.5)
    per_sec["cvd"] = (per_sec["buy_qty"] - per_sec["sell_qty"]).cumsum()
    ohlc = per_sec[[
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_qty",
        "sell_qty",
        "taker_buy_ratio",
        "cvd",
    ]]
    print(
        f"June 1s bars: {len(ohlc)}, range: {ohlc.index[0]} -> {ohlc.index[-1]}"
    )
finally:
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("🧹 Cleaned up temporary files")

mdl = MarketDataLoader()
mdl.raw_data = ohlc
mtf = mdl.get_multi_timeframe_data()
print("Timeframes:", {k: len(v) for k, v in mtf.items()})

print("Engineering June features with existing scalers (no refit)...")
engineered_june = feature_engineer.engineer_features(mtf, fit=False)


def run_bt(
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
    data_tf = engineered_june[timeframe]
    feat_cols = [
        c for c in data_tf.columns
        if c not in ["open", "high", "low", "close", "volume"]
    ]
    X = data_tf[feat_cols].dropna()
    if X.empty:
        print(f"[{timeframe}] No data")
        return None
    s1 = strategy.pipeline.stage1_models[timeframe]
    s2 = strategy.pipeline.stage2_models[timeframe]
    p1 = s1.predict(X)
    p2 = s2.predict(X)
    sig = pd.DataFrame(
        {
            "stage1": p1,
            "stage2": p2,
            "close": data_tf.loc[X.index, "close"]
        },
        index=X.index,
    )
    sig["d"] = 0
    long_th = 0.6 if timeframe == "15T" else 0.55
    short_th = 0.4 if timeframe == "15T" else 0.45
    sig.loc[sig["stage1"] > long_th, "d"] = 1
    sig.loc[sig["stage1"] < short_th, "d"] = -1
    sig["strength"] = np.abs(sig["stage1"] - 0.5)

    capital = initial_capital
    peak_eq = initial_capital
    max_dd = 0.0
    positions = []
    trades = []
    equity = []
    last_sig_ts = None
    risk_pct = float(risk_pct)  # account risk per initial entry
    max_leverage = float(max_leverage)  # total notional cap multiple

    for ts, row in sig.iterrows():
        price = row["close"]
        for pos in positions:
            if pos["status"] != "active":
                continue
            pnl = ((price - pos["entry"]) *
                   pos["size"] if pos["side"] == "long" else
                   (pos["entry"] - price) * pos["size"])
            pos["pnl"] = pnl
            # initialize stop if needed
            if "stop" not in pos:
                # ATR*k stop (fallback to pct if ATR missing)
                atr = (float(data_tf.loc[ts, "atr"])
                       if "atr" in data_tf.columns else None)
                if atr and atr > 0:
                    pos["stop"] = (pos["entry"] - atr_stop_k * atr
                                   if pos["side"] == "long" else pos["entry"] +
                                   atr_stop_k * atr)
                else:
                    pos["stop"] = (pos["entry"] * (1.0 - stop_loss_pct)
                                   if pos["side"] == "long" else pos["entry"] *
                                   (1.0 + stop_loss_pct))
            # partial TP at 1R and move to breakeven
            reached_1r = ((price >= pos["entry"] * (1.0 + take_profit_pct))
                          if pos["side"] == "long" else
                          (price <= pos["entry"] * (1.0 - take_profit_pct)))
            reached_2r = ((price >= pos["entry"] * (1.0 + 2 * take_profit_pct))
                          if pos["side"] == "long" else
                          (price <= pos["entry"] *
                           (1.0 - 2 * take_profit_pct)))
            if (pos["status"] == "active" and not pos.get("tp1_done", False)
                    and reached_1r):
                close_size = pos["size"] * 0.5
                close_pnl = ((price - pos["entry"]) *
                             close_size if pos["side"] == "long" else
                             (pos["entry"] - price) * close_size)
                trades.append({
                    **pos,
                    "size": close_size,
                    "exit": price,
                    "exit_ts": ts,
                    "reason": "tp1_50pct",
                })
                capital += close_pnl
                pos["size"] -= close_size
                pos["tp1_done"] = True
                pos["stop"] = pos["entry"]
            # trailing after 1R using ATR if present
            atr = float(
                data_tf.loc[ts, "atr"]) if "atr" in data_tf.columns else None
            if (pos["status"] == "active" and pos.get("tp1_done", False)
                    and atr and atr > 0):
                if pos["side"] == "long":
                    pos["stop"] = max(pos["stop"],
                                      price - atr_trail_mult * atr)
                else:
                    pos["stop"] = min(pos["stop"],
                                      price + atr_trail_mult * atr)
            # stop check
            if pos["side"] == "long" and price <= pos["stop"]:
                close_pnl = (price - pos["entry"]) * pos["size"]
                trades.append({
                    **pos, "exit": price,
                    "exit_ts": ts,
                    "reason": "stop"
                })
                capital += close_pnl
                pos["status"] = "closed"
            elif pos["side"] == "short" and price >= pos["stop"]:
                close_pnl = (pos["entry"] - price) * pos["size"]
                trades.append({
                    **pos, "exit": price,
                    "exit_ts": ts,
                    "reason": "stop"
                })
                capital += close_pnl
                pos["status"] = "closed"
            # full TP at 2R
            if pos["status"] == "active" and reached_2r:
                close_pnl = ((price - pos["entry"]) *
                             pos["size"] if pos["side"] == "long" else
                             (pos["entry"] - price) * pos["size"])
                trades.append({
                    **pos, "exit": price,
                    "exit_ts": ts,
                    "reason": "tp2_full"
                })
                capital += close_pnl
                pos["status"] = "closed"
            # risk reduction at -0.5R
            adverse = ((price <= pos["entry"] * (1.0 - 0.5 * stop_loss_pct))
                       if pos["side"] == "long" else
                       (price >= pos["entry"] * (1.0 + 0.5 * stop_loss_pct)))
            if (pos["status"] == "active" and adverse
                    and not pos.get("reduced_once", False)):
                cut = pos["size"] * 0.3
                close_pnl = ((price - pos["entry"]) *
                             cut if pos["side"] == "long" else
                             (pos["entry"] - price) * cut)
                trades.append({
                    **pos,
                    "size": cut,
                    "exit": price,
                    "exit_ts": ts,
                    "reason": "risk_reduce_30pct",
                })
                capital += close_pnl
                pos["size"] -= cut
                pos["reduced_once"] = True
        if row["d"] != 0 and row["strength"] > signal_threshold:
            minutes = 5 if timeframe == "5T" else 15
            # limit concurrent active positions
            active_count = sum(1 for p in positions if p["status"] == "active")
            if (last_sig_ts is None or
                (ts - last_sig_ts).total_seconds() / 60.0 >=
                (min_gap_bars * minutes)) and active_count < max_concurrent:
                # risk-based position size: risk fraction of capital to ATR stop
                entry = price
                atr = (float(data_tf.loc[ts, "atr"])
                       if "atr" in data_tf.columns else None)
                stop_price = (
                    (entry - atr_stop_k * atr) if
                    (row["d"] == 1 and atr and atr > 0) else
                    (entry * (1.0 - stop_loss_pct) if row["d"] == 1 else None))
                if row["d"] != 1:
                    stop_price = ((entry + atr_stop_k * atr) if
                                  (atr and atr > 0) else
                                  (entry * (1.0 + stop_loss_pct)))
                stop_dist = abs(entry - stop_price)
                raw_size = (capital *
                            risk_pct) / stop_dist if stop_dist > 0 else 0.0
                # modulate by signal strength (0.5 at threshold -> ~0.0 add; at 1.0 -> full)
                strength_scale = max(
                    0.0,
                    min(
                        1.0,
                        (row["strength"] - signal_threshold) /
                        max(1e-6, (0.5 - signal_threshold)),
                    ),
                )
                size = raw_size * strength_scale
                # leverage cap
                exposure = sum([(p["size"] * price) for p in positions
                                if p["status"] == "active"])
                max_notional = capital * max_leverage
                remain_notional = max(0.0, max_notional - exposure)
                if remain_notional > 0:
                    size = min(size, remain_notional / price)
                if size > 0:
                    positions.append({
                        "side": "long" if row["d"] == 1 else "short",
                        "size": size,
                        "entry": price,
                        "entry_ts": ts,
                        "status": "active",
                        "adds": 0,
                    })
                    last_sig_ts = ts
        # pyramiding when progressed 0.5R/1.0R
        for pos in positions:
            if pos["status"] != "active":
                continue
            progressed_05r = ((price >= pos["entry"] *
                               (1.0 + 0.5 * take_profit_pct)) if pos["side"]
                              == "long" else (price <= pos["entry"] *
                                              (1.0 - 0.5 * take_profit_pct)))
            progressed_10r = ((price >= pos["entry"] *
                               (1.0 + 1.0 * take_profit_pct)) if pos["side"]
                              == "long" else (price <= pos["entry"] *
                                              (1.0 - 1.0 * take_profit_pct)))
            if pos["adds"] < max_adds and (progressed_05r or progressed_10r):
                # add with half risk
                curr_stop = pos.get(
                    "stop",
                    (pos["entry"] * (1.0 - stop_loss_pct) if pos["side"]
                     == "long" else pos["entry"] * (1.0 + stop_loss_pct)),
                )
                stop_dist_now = abs(price - curr_stop)
                add_raw = ((capital * (risk_pct * add_risk_frac)) /
                           stop_dist_now if stop_dist_now > 0 else 0.0)
                # leverage cap on add
                exposure = sum([(p["size"] * price) for p in positions
                                if p["status"] == "active"])
                max_notional = capital * max_leverage
                remain_notional = max(0.0, max_notional - exposure)
                add_size = (min(add_raw, remain_notional /
                                price) if remain_notional > 0 else 0.0)
                if add_size > 0:
                    pos["size"] += add_size
                    pos["adds"] += 1
        open_pnl = sum(
            [p.get("pnl", 0.0) for p in positions if p["status"] == "active"])
        eq = capital + open_pnl
        equity.append({
            "timestamp": ts,
            "equity": eq,
            "capital": capital,
            "open_pnl": open_pnl
        })
        if eq > peak_eq:
            peak_eq = eq
        dd = (peak_eq - eq) / peak_eq
        if dd > max_dd:
            max_dd = dd

    if len(sig) > 0:
        last_price = sig["close"].iloc[-1]
        last_ts = sig.index[-1]
        for pos in positions:
            if pos["status"] == "active":
                pnl = ((last_price - pos["entry"]) *
                       pos["size"] if pos["side"] == "long" else
                       (pos["entry"] - last_price) * pos["size"])
                trades.append({
                    **pos,
                    "exit": last_price,
                    "exit_ts": last_ts,
                    "reason": "end_of_data",
                })
                capital += pnl
                pos["status"] = "closed"
                pos["pnl"] = pnl

    wins = [
        t for t in trades if t.get(
            "pnl",
            (t["exit"] - t["entry"]) *
            (t["size"] if t["side"] == "long" else -t["size"]),
        ) > 0
    ]
    losses = [t for t in trades if t not in wins]
    avg_win = np.mean([t.get("pnl", 0) for t in wins]) if wins else 0
    avg_loss = np.mean([t.get("pnl", 0) for t in losses]) if losses else 0
    total_pnl = sum([t.get("pnl", 0) for t in trades])
    final_eq = capital
    res = {
        "timeframe":
        timeframe,
        "total_trades":
        len(trades),
        "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "total_pnl":
        float(total_pnl),
        "total_return": (final_eq - 100000.0) / 100000.0 * 100.0,
        "avg_win":
        float(avg_win) if wins else 0.0,
        "avg_loss":
        float(avg_loss) if losses else 0.0,
        "profit_factor":
        ((abs(avg_win * len(wins) /
              (avg_loss * len(losses))) if losses else float("inf"))
         if trades else 0.0),
        "max_drawdown":
        max_dd * 100.0,
        "final_equity":
        float(final_eq),
        "initial_capital":
        100000.0,
    }
    pd.DataFrame(trades).to_csv(os.path.join(
        RESULTS_DIR, f"wavelet_{timeframe}_june_trades.csv"),
                                index=False)
    pd.DataFrame(equity).to_csv(
        os.path.join(RESULTS_DIR,
                     f"wavelet_{timeframe}_june_equity_curve.csv"),
        index=False,
    )
    with open(
            os.path.join(RESULTS_DIR,
                         f"wavelet_{timeframe}_june_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{timeframe}] Results:", res)
    return res


# Optionally load tuned params if present
best_csv = os.path.join(RESULTS_DIR, "grid_tune_best.csv")
best_optuna_json = os.path.join(RESULTS_DIR, "optuna_best_params.json")
params = {
    "5T": {
        "sl": 0.02,
        "tp": 0.04,
        "sig": 0.04,
        "gap": 3
    },
    "15T": {
        "sl": 0.02,
        "tp": 0.04,
        "sig": 0.04,
        "gap": 3
    },
}
if os.path.exists(best_optuna_json):
    try:
        opt = json.load(open(best_optuna_json))
        for tf, p in opt.items():
            params[str(tf)] = {
                "sl": float(p["sl"]),
                "tp": float(p["tp"]),
                "sig": float(p["sig"]),
                "gap": int(p["gap"]),
            }
        print("Loaded Optuna params:", params)
    except Exception:
        pass
elif os.path.exists(best_csv):
    try:
        best = pd.read_csv(best_csv)
        for _, row in best.iterrows():
            tf = str(row["timeframe"])
            params[tf] = {
                "sl": float(row["sl"]),
                "tp": float(row["tp"]),
                "sig": float(row["sig_th"]),
                "gap": int(row["gap"]),
            }
        print("Loaded tuned params:", params)
    except Exception:
        pass

print("Running June OOS 5T...")
p5 = params["5T"]
# Conservative overrides to target <=10% DD
p5_sl = max(p5["sl"], 0.03)
p5_tp = max(p5["tp"], 0.05)
p5_sig = max(p5["sig"], 0.08)
p5_gap = max(p5["gap"], 8)
res_5t = run_bt(
    "5T",
    stop_loss_pct=p5_sl,
    take_profit_pct=p5_tp,
    signal_threshold=p5_sig,
    min_gap_bars=p5_gap,
    risk_pct=0.0015,
    max_leverage=1.2,
    max_adds=0,
    add_risk_frac=0.0,
    atr_trail_mult=2.5,
)
print("Running June OOS 15T...")
p15 = params["15T"]
p15_sl = max(p15["sl"], 0.025)
p15_tp = max(p15["tp"], 0.05)
p15_sig = max(p15["sig"], 0.06)
p15_gap = max(p15["gap"], 6)
res_15t = run_bt(
    "15T",
    stop_loss_pct=p15_sl,
    take_profit_pct=p15_tp,
    signal_threshold=p15_sig,
    min_gap_bars=p15_gap,
    risk_pct=0.002,
    max_leverage=1.2,
    max_adds=0,
    add_risk_frac=0.0,
    atr_trail_mult=2.5,
)

# Additionally run 60T (1h) and 240T (4h) with conservative overrides
print("Running June OOS 60T (1h)...")
res_60t = run_bt(
    "60T",
    stop_loss_pct=0.03,
    take_profit_pct=0.06,
    signal_threshold=0.06,
    min_gap_bars=6,
    risk_pct=0.002,
    max_leverage=2.0,
    max_adds=1,
    add_risk_frac=0.2,
    atr_trail_mult=2.0,
)
print("Running June OOS 240T (4h)...")
res_240t = run_bt(
    "240T",
    stop_loss_pct=0.03,
    take_profit_pct=0.06,
    signal_threshold=0.06,
    min_gap_bars=6,
    risk_pct=0.002,
    max_leverage=2.0,
    max_adds=1,
    add_risk_frac=0.2,
    atr_trail_mult=2.0,
)

import shutil

shutil.rmtree(temp_dir, ignore_errors=True)
print("Done.")
