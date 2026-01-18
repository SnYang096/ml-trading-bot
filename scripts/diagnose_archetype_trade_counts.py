#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _count_entries(
    df: pd.DataFrame,
    *,
    symbol_col: str,
    timestamp_col: str,
    mode_col: str,
    archetype_col: str,
    gate_decision_col: str,
) -> pd.DataFrame:
    work = df.copy()
    work[symbol_col] = work[symbol_col].astype(str)
    work[mode_col] = work[mode_col].astype(str).str.upper()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col]).sort_values([symbol_col, timestamp_col])

    counts: List[Dict[str, object]] = []
    for sym, g in work.groupby(symbol_col, sort=True):
        g = g.reset_index(drop=True)
        allow_g = g[gate_decision_col].fillna("").astype(str).str.lower() == "allow"
        tradable_g = allow_g & (g[mode_col] != "NO_TRADE")
        mode_g = g[mode_col].to_numpy()
        arch_g = g[archetype_col].fillna("").astype(str).to_numpy()
        prev_in = False
        for i in range(len(g)):
            in_trade = bool(tradable_g[i])
            if in_trade and not prev_in:
                arch = arch_g[i] if arch_g[i] else "UNKNOWN"
                counts.append(
                    {
                        "symbol": sym,
                        "archetype": arch,
                        "entry_ts": g.loc[i, timestamp_col],
                        "mode": mode_g[i],
                    }
                )
            prev_in = in_trade

    return pd.DataFrame(counts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Count archetype trade entries from gated mode outputs."
    )
    ap.add_argument("--mode", required=True, help="mode_3action_gate parquet/csv")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--symbol-col", default="symbol")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--mode-col", default="mode")
    ap.add_argument("--archetype-col", default="gate_archetype")
    ap.add_argument("--gate-decision-col", default="gate_decision")
    args = ap.parse_args()

    mode_df = _read_any(Path(args.mode))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = _count_entries(
        mode_df,
        symbol_col=args.symbol_col,
        timestamp_col=args.timestamp_col,
        mode_col=args.mode_col,
        archetype_col=args.archetype_col,
        gate_decision_col=args.gate_decision_col,
    )

    if entries.empty:
        summary = {
            "total_entries": 0,
            "by_archetype": {},
            "by_symbol": {},
        }
    else:
        by_arch = entries["archetype"].value_counts().to_dict()
        by_sym = entries["symbol"].value_counts().to_dict()
        summary = {
            "total_entries": int(len(entries)),
            "by_archetype": {str(k): int(v) for k, v in by_arch.items()},
            "by_symbol": {str(k): int(v) for k, v in by_sym.items()},
        }

    entries.to_csv(out_dir / "entries.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        "# Archetype Trade Counts\n\n"
        + f"Total entries: **{summary.get('total_entries', 0)}**\n\n"
        + "## By archetype\n\n"
        + "\n".join(
            [f"- {k}: {v}" for k, v in (summary.get("by_archetype") or {}).items()]
        )
        + "\n\n## By symbol\n\n"
        + "\n".join(
            [f"- {k}: {v}" for k, v in (summary.get("by_symbol") or {}).items()]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"✅ Wrote: {out_dir / 'entries.csv'}")
    print(f"✅ Wrote: {out_dir / 'summary.json'}")
    print(f"✅ Wrote: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
