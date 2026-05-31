"""DEPRECATED — SRB fake_break_reverse 语义已移除。

保留以便读取旧实验产物（trades_csv 列 is_reverse 现在恒为 False）。
反手逻辑归 FBF（docs/design/fbf/ 与 config/strategies/fbf/）与未来策略 X
（docs/design/strategy_x_hub_rebound.md）。本脚本仅用于回看旧 run，
新的回测不会再产生反手交易。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

# Allow direct `python scripts/...` execution from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feature_store.layer_naming import detect_layer_for_strategy
from src.time_series_model.live.srb_regime import swing_sr_levels


@dataclass
class MonthBundle:
    month: str
    run_root: Path
    trades_csv: Path
    event_json: Path
    exec_yaml: Path


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_exec_policy(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("fake_break_reverse") or {}


def _detect_srb_layer(repo_root: Path) -> str:
    layer = detect_layer_for_strategy(
        strategy="srb",
        features_store_root=str(repo_root / "feature_store"),
        timeframe="120T",
    )
    if not layer:
        raise FileNotFoundError("Could not detect SRB feature store layer")
    return str(layer)


def _load_symbol_features(
    repo_root: Path,
    layer: str,
    symbol: str,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if symbol in cache:
        return cache[symbol]
    base = repo_root / "feature_store" / layer / symbol / "120T"
    parts = sorted(base.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files found for {symbol} in {base}")
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, axis=0).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex for {symbol} feature store rows")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    cache[symbol] = df
    return df


def _consecutive_confirm_streak(
    closes: pd.Series,
    threshold: float | None,
    side: str,
    confirm_k: int,
) -> tuple[int, int | None, bool]:
    if threshold is None or closes.empty:
        return 0, None, False
    best = 0
    curr = 0
    first_hit_bar: int | None = None
    want_long = side.upper() in {"LONG", "BUY"}
    for i, px in enumerate(closes.astype(float).tolist(), start=1):
        cond = px > threshold if want_long else px < threshold
        if cond:
            curr += 1
            if first_hit_bar is None:
                first_hit_bar = i
            if curr > best:
                best = curr
        else:
            curr = 0
    return best, first_hit_bar, best >= confirm_k


def _stop_hunt_extreme(
    bars: pd.DataFrame,
    side: str,
    sample_window: int = 3,
) -> float | None:
    if bars.empty:
        return None
    sub = bars.head(max(1, sample_window))
    want_long = side.upper() in {"LONG", "BUY"}
    if want_long:
        return float(sub["low"].astype(float).min())
    return float(sub["high"].astype(float).max())


def _bars_after_stop(
    df: pd.DataFrame, stop_ts: pd.Timestamp, lookahead: int
) -> pd.DataFrame:
    out = df.loc[df.index > stop_ts].head(lookahead).copy()
    return out


def _bars_before_or_at(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    return df.loc[:ts].copy()


def _candidate_rows(
    repo_root: Path,
    run_dir: Path,
    layer: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = _read_jsonl(run_dir / "monthly_ledger.jsonl")
    feature_cache: dict[str, pd.DataFrame] = {}
    candidates: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    for item in ledger:
        run_root = Path(item["run_root"])
        month = str(item["month"])
        bundle = MonthBundle(
            month=month,
            run_root=run_root,
            trades_csv=run_root / "srb" / "event_trades_srb.csv",
            event_json=run_root / "srb" / "event_backtest_srb.json",
            exec_yaml=run_root
            / "strategies_calibrated"
            / "srb"
            / "archetypes"
            / "execution.yaml",
        )
        event_obj = _read_json(bundle.event_json)
        policy = _load_exec_policy(bundle.exec_yaml)
        expired_count = int(
            (event_obj.get("funnel") or {}).get("srb_reverse_expired", 0) or 0
        )
        confirm_k = int(policy.get("confirm_k", 3) or 3)
        lookahead = int(policy.get("fake_lookahead", 10) or 10)
        cooldown_bars = int(policy.get("cooldown_bars", 10) or 10)
        max_reverse_per_level = int(policy.get("max_reverse_per_level", 1) or 1)
        sr_lookback = 20

        trades = pd.read_csv(bundle.trades_csv)
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        source = trades[
            (trades["exit_reason"] == "sl")
            & (~trades["is_add_position"].astype(bool))
            & (~trades["is_reverse"].astype(bool))
        ].sort_values("exit_time")
        expired = source.head(expired_count).copy()
        pending_src = source.iloc[expired_count:].copy()

        for _, row in pending_src.iterrows():
            pending.append(
                {
                    "month": month,
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "entry_time": row["entry_time"],
                    "exit_time": row["exit_time"],
                    "entry_price": float(row["entry_price"]),
                    "exit_price": float(row["exit_price"]),
                }
            )

        for _, row in expired.iterrows():
            symbol = str(row["symbol"])
            side = str(row["side"]).upper()
            df = _load_symbol_features(repo_root, layer, symbol, feature_cache)
            future = _bars_after_stop(df, row["exit_time"], lookahead)
            closes = (
                future["close"] if "close" in future.columns else pd.Series(dtype=float)
            )

            hist_to_entry = _bars_before_or_at(df, row["entry_time"])
            sup, res = swing_sr_levels(hist_to_entry, row["entry_time"], sr_lookback)
            true_sr_level = res if side in {"LONG", "BUY"} else sup
            stop_hunt_level = _stop_hunt_extreme(
                future, side, sample_window=min(3, lookahead)
            )

            entry_best, entry_first, entry_would = _consecutive_confirm_streak(
                closes, float(row["entry_price"]), side, confirm_k
            )
            true_best, true_first, true_would = _consecutive_confirm_streak(
                closes, true_sr_level, side, confirm_k
            )
            hunt_best, hunt_first, hunt_would = _consecutive_confirm_streak(
                closes, stop_hunt_level, side, confirm_k
            )

            candidates.append(
                {
                    "month": month,
                    "symbol": symbol,
                    "side": side,
                    "entry_time": row["entry_time"],
                    "exit_time": row["exit_time"],
                    "entry_price": float(row["entry_price"]),
                    "exit_price": float(row["exit_price"]),
                    "atr": float(row["atr"]),
                    "confirm_k": confirm_k,
                    "fake_lookahead": lookahead,
                    "cooldown_bars": cooldown_bars,
                    "max_reverse_per_level": max_reverse_per_level,
                    "bars_observed": int(len(future)),
                    "true_sr_level": true_sr_level,
                    "stop_hunt_extreme": stop_hunt_level,
                    "would_trigger_under_entry_price": bool(entry_would),
                    "would_trigger_under_true_sr": bool(true_would),
                    "would_trigger_under_stop_hunt_extreme": bool(hunt_would),
                    "best_confirm_streak_entry_price": int(entry_best),
                    "best_confirm_streak_true_sr": int(true_best),
                    "best_confirm_streak_stop_hunt_extreme": int(hunt_best),
                    "bars_to_first_reclaim_entry_price": entry_first,
                    "bars_to_first_reclaim_true_sr": true_first,
                    "bars_to_first_reclaim_stop_hunt_extreme": hunt_first,
                    "first_future_close": (
                        float(closes.iloc[0]) if not closes.empty else None
                    ),
                    "last_future_close": (
                        float(closes.iloc[-1]) if not closes.empty else None
                    ),
                    "sr_strength_max_at_stop_plus1": (
                        float(future["sr_strength_max"].iloc[0])
                        if not future.empty and "sr_strength_max" in future.columns
                        else None
                    ),
                    "bpc_volume_compression_pct_at_stop_plus1": (
                        float(future["bpc_volume_compression_pct"].iloc[0])
                        if not future.empty
                        and "bpc_volume_compression_pct" in future.columns
                        else None
                    ),
                }
            )

    cand_df = pd.DataFrame(candidates).sort_values(["month", "exit_time", "symbol"])
    pending_df = pd.DataFrame(pending).sort_values(["month", "exit_time", "symbol"])
    return cand_df, pending_df


def _summary_markdown(
    run_dir: Path,
    layer: str,
    cand_df: pd.DataFrame,
    pending_df: pd.DataFrame,
) -> str:
    total = len(cand_df)
    entry_hits = int(cand_df["would_trigger_under_entry_price"].sum())
    true_hits = int(cand_df["would_trigger_under_true_sr"].sum())
    hunt_hits = int(cand_df["would_trigger_under_stop_hunt_extreme"].sum())
    both_true_not_entry = cand_df[
        (~cand_df["would_trigger_under_entry_price"])
        & (cand_df["would_trigger_under_true_sr"])
    ]
    hunt_not_entry = cand_df[
        (~cand_df["would_trigger_under_entry_price"])
        & (cand_df["would_trigger_under_stop_hunt_extreme"])
    ]
    month_table = (
        cand_df.groupby("month", as_index=False)
        .agg(
            expired_candidates=("symbol", "size"),
            entry_hits=("would_trigger_under_entry_price", "sum"),
            true_sr_hits=("would_trigger_under_true_sr", "sum"),
            hunt_hits=("would_trigger_under_stop_hunt_extreme", "sum"),
        )
        .sort_values("month")
    )
    top_examples = cand_df[
        (~cand_df["would_trigger_under_entry_price"])
        & (
            cand_df["would_trigger_under_true_sr"]
            | cand_df["would_trigger_under_stop_hunt_extreme"]
        )
    ].copy()
    top_examples["rank_score"] = (
        top_examples["best_confirm_streak_true_sr"]
        + top_examples["best_confirm_streak_stop_hunt_extreme"]
    )
    top_examples = top_examples.sort_values(
        ["rank_score", "month", "symbol"], ascending=[False, True, True]
    ).head(10)

    lines = [
        "# SRB Reverse Shadow Report",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Feature layer: `{layer}`",
        f"- Expired candidates reconstructed: **{total}**",
        f"- Pending (not expired) source stop-outs: **{len(pending_df)}**",
        "",
        "## Headline",
        "",
        f"- `entry_price` semantic would trigger: **{entry_hits}/{total}**",
        f"- `true_sr_level` semantic would trigger: **{true_hits}/{total}**",
        f"- `stop_hunt_extreme` semantic would trigger: **{hunt_hits}/{total}**",
        "",
        "## Interpretation",
        "",
        f"- Cases where `entry_price` missed but `true_sr_level` would trigger: **{len(both_true_not_entry)}**",
        f"- Cases where `entry_price` missed but `stop_hunt_extreme` would trigger: **{len(hunt_not_entry)}**",
        "",
        "## Monthly Breakdown",
        "",
        month_table.to_markdown(index=False),
        "",
        "## Representative Cases",
        "",
    ]
    if top_examples.empty:
        lines.append("- No examples where alternate anchors beat `entry_price`.")
    else:
        keep = [
            "month",
            "symbol",
            "side",
            "entry_price",
            "true_sr_level",
            "stop_hunt_extreme",
            "best_confirm_streak_entry_price",
            "best_confirm_streak_true_sr",
            "best_confirm_streak_stop_hunt_extreme",
            "would_trigger_under_entry_price",
            "would_trigger_under_true_sr",
            "would_trigger_under_stop_hunt_extreme",
        ]
        lines.append(top_examples[keep].to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Pending Source Stop-outs",
            "",
            pending_df.to_markdown(index=False) if not pending_df.empty else "- None",
            "",
            "## Recommendation",
            "",
        ]
    )
    if true_hits > entry_hits:
        lines.append(
            "- `entry_price` looks stricter than structural SR. Next implementation should record the breakout-side `true_sr_level` at candidate creation and use it as the primary reclaim anchor."
        )
    else:
        lines.append(
            "- `entry_price` does not look materially stricter than structural SR in this run. Focus next on reclaim timing / confirm semantics rather than anchor replacement."
        )
    if hunt_hits > true_hits:
        lines.append(
            "- `stop_hunt_extreme` is very permissive in this shadow run. Treat it as a secondary diagnostic or auxiliary condition, not the only trigger anchor."
        )
    else:
        lines.append(
            "- `stop_hunt_extreme` is useful as a diagnostic field, but structural SR remains the cleaner primary semantic anchor."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="SRB reverse shadow report")
    parser.add_argument("--run-dir", required=True, help="rolling_sim run root")
    parser.add_argument(
        "--feature-layer",
        default="",
        help="optional feature store layer; default auto-detect latest SRB 120T layer",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    repo_root = run_dir.parents[4]
    layer = args.feature_layer.strip() or _detect_srb_layer(repo_root)

    cand_df, pending_df = _candidate_rows(repo_root, run_dir, layer)
    out_csv = run_dir / "srb_reverse_shadow_candidates.csv"
    out_pending = run_dir / "srb_reverse_shadow_pending.csv"
    out_md = run_dir / "srb_reverse_shadow_summary.md"

    cand_df.to_csv(out_csv, index=False)
    pending_df.to_csv(out_pending, index=False)
    out_md.write_text(
        _summary_markdown(run_dir, layer, cand_df, pending_df), encoding="utf-8"
    )

    print(f"wrote: {out_csv}")
    print(f"wrote: {out_pending}")
    print(f"wrote: {out_md}")
    print(
        json.dumps(
            {
                "expired_candidates": int(len(cand_df)),
                "pending_candidates": int(len(pending_df)),
                "entry_hits": int(cand_df["would_trigger_under_entry_price"].sum()),
                "true_sr_hits": int(cand_df["would_trigger_under_true_sr"].sum()),
                "stop_hunt_hits": int(
                    cand_df["would_trigger_under_stop_hunt_extreme"].sum()
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
