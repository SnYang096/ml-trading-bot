import os, json, pickle, zipfile
import numpy as np
import pandas as pd
import optuna
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ml_trading.data_tools.data_loader import MarketDataLoader

MODEL_PATH = os.path.join("models", "trained_model_wavelet_may_2025.pkl")
SCALER_PATH = os.path.join("models", "feature_scalers_wavelet_may_2025.pkl")
JUNE_ZIP = os.path.join("data", "raw", "BTCUSDT-aggTrades-2025-06.zip")
OUT_DIR = os.path.join("results", "june_2025_oos")
os.makedirs(OUT_DIR, exist_ok=True)


def prepare_data():
    tmp = os.path.join("data", "temp_optuna")
    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(JUNE_ZIP, "r") as z:
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
    ps = df.groupby(pd.Grouper(freq="1s")).agg(
        {
            "price": ["first", "max", "min", "last"],
            "quantity": "sum",
            "is_buyer_maker": "mean",
        }
    )
    ps.columns = ["open", "high", "low", "close", "volume", "is_buyer_maker"]
    ps = ps.dropna().ffill()
    ps["taker_buy"] = (~ps["is_buyer_maker"].round().astype(bool)).astype(int)
    ps["buy_qty"] = ps["taker_buy"] * ps["volume"]
    ps["sell_qty"] = (1 - ps["taker_buy"]) * ps["volume"]
    ps["taker_buy_ratio"] = (
        ps["buy_qty"] / (ps["buy_qty"] + ps["sell_qty"]).replace(0, np.nan)
    ).fillna(0.5)
    ps["cvd"] = (ps["buy_qty"] - ps["sell_qty"]).cumsum()
    mdl = MarketDataLoader()
    mdl.raw_data = ps
    return mdl.get_multi_timeframe_data()


def load_components():
    m = pickle.load(open(MODEL_PATH, "rb"))
    fe = m["feature_engineer"]
    fe.load_scalers(SCALER_PATH)
    strat = m["strategy"]
    return strat, fe


def run_once(strat, fe, mtf, timeframe, params):
    # import run_bt from oos_june
    import os, sys

    code = open("scripts/oos_june.py", "r").read()
    ns = {"__file__": os.path.abspath("scripts/oos_june.py"), "os": os, "sys": sys}
    exec(compile(code, "scripts/oos_june.py", "exec"), ns, ns)
    run_bt = ns["run_bt"]
    return run_bt(
        timeframe,
        stop_loss_pct=params["sl"],
        take_profit_pct=params["tp"],
        signal_threshold=params["sig"],
        min_gap_bars=params["gap"],
        initial_capital=100000.0,
        risk_pct=params["risk"],
        max_leverage=params["lev"],
        max_adds=params["adds"],
        add_risk_frac=params["add_frac"],
        atr_trail_mult=params["atr_trail"],
        atr_stop_k=params["atr_k"],
        max_concurrent=params["max_cc"],
    )


def score(res):
    # Objective: penalize high DD; prefer higher return with DD cap 10%
    dd = res["max_drawdown"]
    ret = res["total_return"]
    if dd > 10.0:
        return -dd  # infeasible region
    return ret - 0.5 * dd


def objective(trial, timeframe, mtf, strat, fe):
    params = {
        "sl": trial.suggest_float("sl", 0.01, 0.05),
        "tp": trial.suggest_float("tp", 0.02, 0.08),
        "sig": trial.suggest_float("sig", 0.04, 0.12),
        "gap": trial.suggest_int("gap", 3, 12),
        "risk": trial.suggest_float("risk", 0.0005, 0.005, log=True),
        "lev": trial.suggest_float("lev", 1.0, 3.0),
        "adds": trial.suggest_int("adds", 0, 2),
        "add_frac": trial.suggest_float("add_frac", 0.1, 0.5),
        "atr_trail": trial.suggest_float("atr_trail", 1.5, 3.0),
        "atr_k": trial.suggest_float("atr_k", 1.0, 3.0),
        "max_cc": trial.suggest_int("max_cc", 1, 2),
    }
    res = run_once(strat, fe, {timeframe: mtf[timeframe]}, timeframe, params)
    return score(res)


def main():
    mtf = prepare_data()
    strat, fe = load_components()
    study5 = optuna.create_study(direction="maximize")
    study5.optimize(lambda t: objective(t, "5T", mtf, strat, fe), n_trials=20)
    study15 = optuna.create_study(direction="maximize")
    study15.optimize(lambda t: objective(t, "15T", mtf, strat, fe), n_trials=20)
    best = {"5T": study5.best_trial.params, "15T": study15.best_trial.params}
    with open(os.path.join(OUT_DIR, "optuna_best_params.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("Best found:", best)


if __name__ == "__main__":
    main()
