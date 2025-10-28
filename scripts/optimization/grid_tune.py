import os, json, pickle, zipfile
import numpy as np
import pandas as pd
from itertools import product

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ml_trading.data_tools.data_loader import MarketDataLoader

MODEL_PATH = os.path.join("models", "trained_model_wavelet_may_2025.pkl")
SCALER_PATH = os.path.join("models", "feature_scalers_wavelet_may_2025.pkl")
JUNE_ZIP = os.path.join("data", "raw", "BTCUSDT-aggTrades-2025-06.zip")
OUT_DIR = os.path.join("results", "june_2025_oos")
os.makedirs(OUT_DIR, exist_ok=True)


def prepare_ohlc(zip_path: str) -> pd.DataFrame:
    tmp = os.path.join("data", "temp_grid_june")
    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp)
    csvs = [f for f in os.listdir(tmp) if f.endswith(".csv")]
    if not csvs:
        raise SystemExit("No CSV in zip")
    df = pd.read_csv(os.path.join(tmp, csvs[0]))
    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["is_buyer_maker"] = df.get("is_buyer_maker", False)
    per_sec = df.groupby(pd.Grouper(freq="1s")).agg(
        {
            "price": ["first", "max", "min", "last"],
            "quantity": "sum",
            "is_buyer_maker": "mean",
        }
    )
    per_sec.columns = ["open", "high", "low", "close", "volume", "is_buyer_maker"]
    per_sec = per_sec.dropna().ffill()
    per_sec["taker_buy"] = (~per_sec["is_buyer_maker"].round().astype(bool)).astype(int)
    per_sec["buy_qty"] = per_sec["taker_buy"] * per_sec["volume"]
    per_sec["sell_qty"] = (1 - per_sec["taker_buy"]) * per_sec["volume"]
    per_sec["taker_buy_ratio"] = per_sec["buy_qty"] / (
        per_sec["buy_qty"] + per_sec["sell_qty"]
    ).replace(0, np.nan)
    per_sec["taker_buy_ratio"] = per_sec["taker_buy_ratio"].fillna(0.5)
    per_sec["cvd"] = (per_sec["buy_qty"] - per_sec["sell_qty"]).cumsum()
    return per_sec[
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


def run_once(strategy, feature_engineer, mtf, timeframe, sl, tp, sig_th, min_gap_bars):
    data_tf = feature_engineer.engineer_features(mtf, fit=False)[timeframe]
    feat_cols = [
        c
        for c in data_tf.columns
        if c not in ["open", "high", "low", "close", "volume"]
    ]
    X = data_tf[feat_cols].dropna()
    if X.empty:
        return None
    s1 = strategy.pipeline.stage1_models[timeframe]
    s2 = strategy.pipeline.stage2_models[timeframe]
    p1 = s1.predict(X)
    p2 = s2.predict(X)
    sig = pd.DataFrame(
        {"stage1": p1, "stage2": p2, "close": data_tf.loc[X.index, "close"]},
        index=X.index,
    )
    sig["d"] = 0
    sig.loc[sig["stage1"] > (0.5 + sig_th), "d"] = 1
    sig.loc[sig["stage1"] < (0.5 - sig_th), "d"] = -1
    sig["strength"] = np.abs(sig["stage1"] - 0.5)
    capital = 100000.0
    positions = []
    trades = []
    last_sig_ts = None
    minutes = 5 if timeframe == "5T" else 15
    for ts, row in sig.iterrows():
        price = row["close"]
        # manage open positions
        for pos in positions:
            if pos["status"] != "active":
                continue
            pnl = (
                (price - pos["entry"]) * pos["size"]
                if pos["side"] == "long"
                else (pos["entry"] - price) * pos["size"]
            )
            pos["pnl"] = pnl
            if pos["side"] == "long" and (
                price <= pos["entry"] * (1 - sl) or price >= pos["entry"] * (1 + tp)
            ):
                trades.append({**pos, "exit": price, "exit_ts": ts})
                capital += pnl
                pos["status"] = "closed"
            if pos["side"] == "short" and (
                price >= pos["entry"] * (1 + sl) or price <= pos["entry"] * (1 - tp)
            ):
                trades.append({**pos, "exit": price, "exit_ts": ts})
                capital += pnl
                pos["status"] = "closed"
        # entries
        if row["d"] != 0 and row["strength"] > sig_th:
            if last_sig_ts is None or (ts - last_sig_ts).total_seconds() / 60.0 >= (
                min_gap_bars * minutes
            ):
                size = (capital * (0.1 if timeframe == "5T" else 0.15) / price) * row[
                    "strength"
                ]
                positions.append(
                    {
                        "side": "long" if row["d"] == 1 else "short",
                        "size": size,
                        "entry": price,
                        "entry_ts": ts,
                        "status": "active",
                    }
                )
                last_sig_ts = ts
    # close remaining
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
                trades.append({**pos, "exit": last_price, "exit_ts": last_ts})
                capital += pnl
                pos["status"] = "closed"
                pos["pnl"] = pnl
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    pf = (
        (
            sum(t.get("pnl", 0) for t in wins)
            / abs(sum(t.get("pnl", 0) for t in losses))
            if losses
            else float("inf")
        )
        if trades
        else 0.0
    )
    return {
        "trades": len(trades),
        "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "profit_factor": pf,
        "final_equity": capital,
        "return_pct": (capital - 100000.0) / 100000.0 * 100.0,
    }


def main():
    print("Loading model...")
    model = pickle.load(open(MODEL_PATH, "rb"))
    strategy = model["strategy"]
    fe = model["feature_engineer"]
    # ensure scalers are up to date
    fe.load_scalers(SCALER_PATH)
    print("Preparing June data...")
    ohlc = prepare_ohlc(JUNE_ZIP)
    mdl = MarketDataLoader()
    mdl.raw_data = ohlc
    mtf = mdl.get_multi_timeframe_data()
    print({k: len(v) for k, v in mtf.items()})
    # grid
    timeframes = ["5T", "15T"]
    sl_grid = [0.01, 0.015, 0.02]
    tp_grid = [0.02, 0.03, 0.04]
    sig_grid = [0.04, 0.06, 0.08]
    gap_grid = [3, 5]
    results = []
    for tf, sl, tp, sg, gap in product(
        timeframes, sl_grid, tp_grid, sig_grid, gap_grid
    ):
        res = run_once(strategy, fe, {tf: mtf[tf]}, tf, sl, tp, sg, gap)
        if res is None:
            continue
        res.update({"timeframe": tf, "sl": sl, "tp": tp, "sig_th": sg, "gap": gap})
        results.append(res)
        print(tf, sl, tp, sg, gap, "=>", res["return_pct"], "ret")
    df = pd.DataFrame(results).sort_values(
        ["timeframe", "return_pct", "profit_factor"], ascending=[True, False, False]
    )
    df.to_csv(os.path.join(OUT_DIR, "grid_tune_results.csv"), index=False)
    best = df.groupby("timeframe").head(1)
    best.to_csv(os.path.join(OUT_DIR, "grid_tune_best.csv"), index=False)
    print("Best params:")
    print(best)


if __name__ == "__main__":
    main()
