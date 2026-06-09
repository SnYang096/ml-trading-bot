#!/usr/bin/env python3
"""List features_labeled.parquet candidates with monitor-relevant metadata."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MONITOR_COLS = (
    "forward_rr",
    "ema_1200_position",
    "vol_persistence",
    "vol_leverage_asymmetry",
)


def _infer_strategy(path: Path) -> str:
    parts = path.parts
    if "train_final" in parts:
        idx = parts.index("train_final")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    for i, part in enumerate(parts):
        if part == "strategies" and i + 1 < len(parts):
            return parts[i + 1]
    if path.parent.name not in ("tpc", "bpc", "me", "srb", "fast_scalp", "chop_grid"):
        return path.parent.name
    return path.parent.name


def _timestamp_col(columns: set[str]) -> Optional[str]:
    if "timestamp" in columns:
        return "timestamp"
    if "datetime" in columns:
        return "datetime"
    return None


def summarize_parquet(path: Path) -> Dict[str, Any]:
    import pandas as pd

    stat = path.stat()
    try:
        rel = str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        rel = str(path.resolve())
    out: Dict[str, Any] = {
        "path": rel,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "strategy_guess": _infer_strategy(path),
    }

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        schema_names = set(pf.schema_arrow.names)
        out["rows"] = int(pf.metadata.num_rows)
        out["columns"] = len(schema_names)
    except Exception as exc:
        out["error"] = f"parquet open: {exc}"
        return out

    ts_col = _timestamp_col(schema_names)
    out["timestamp_col"] = ts_col
    for col in _MONITOR_COLS:
        out[f"has_{col}"] = col in schema_names

    read_cols: List[str] = []
    if ts_col:
        read_cols.append(ts_col)
    if "symbol" in schema_names:
        read_cols.append("symbol")

    if read_cols:
        try:
            df = pd.read_parquet(path, columns=read_cols)
            if ts_col:
                ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                out["time_start"] = ts.min().isoformat() if ts.notna().any() else None
                out["time_end"] = ts.max().isoformat() if ts.notna().any() else None
            if "symbol" in df.columns:
                syms = sorted(
                    {str(s).upper() for s in df["symbol"].dropna().unique().tolist()}
                )
                out["symbols"] = syms
                out["n_symbols"] = len(syms)
            else:
                out["symbols"] = None
                out["n_symbols"] = None
        except Exception as exc:
            out["sample_error"] = str(exc)

    return out


def discover_parquets(
    root: Path,
    *,
    strategy: str = "",
    name: str = "features_labeled.parquet",
    limit: int = 30,
) -> List[Path]:
    if not root.is_dir():
        return []
    matches: List[Path] = []
    for path in root.rglob(name):
        if not path.is_file():
            continue
        if strategy:
            guess = _infer_strategy(path)
            slug = strategy.strip().lower()
            if slug not in path.as_posix().lower() and guess.lower() != slug:
                continue
        matches.append(path)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if limit > 0:
        matches = matches[:limit]
    return matches


def format_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No features_labeled.parquet files found."
    headers = [
        "mtime",
        "rows",
        "symbols",
        "time_start",
        "time_end",
        "fwd_rr",
        "ema1200",
        "path",
    ]
    lines = [
        "  ".join(
            h.ljust(12 if i < len(headers) - 1 else 0) for i, h in enumerate(headers)
        )
    ]
    lines.append("-" * 120)
    for row in rows:
        if row.get("error"):
            lines.append(f"ERROR {row.get('path')}: {row['error']}")
            continue
        syms = row.get("symbols")
        if isinstance(syms, list):
            sym_s = ",".join(syms[:4])
            if len(syms) > 4:
                sym_s += f"+{len(syms)-4}"
        elif syms is None:
            sym_s = "(none)"
        else:
            sym_s = str(syms)
        mtime = str(row.get("mtime", ""))[:10]
        t0 = str(row.get("time_start") or "")[:10]
        t1 = str(row.get("time_end") or "")[:10]
        lines.append(
            "  ".join(
                [
                    mtime.ljust(12),
                    str(row.get("rows", "?")).ljust(12),
                    sym_s[:24].ljust(24),
                    t0.ljust(12),
                    t1.ljust(12),
                    ("Y" if row.get("has_forward_rr") else "n").ljust(8),
                    ("Y" if row.get("has_ema_1200_position") else "n").ljust(8),
                    str(row.get("path", "")),
                ]
            )
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Catalog features_labeled.parquet files for local monitor replay"
    )
    p.add_argument(
        "--root",
        default="results",
        help="Search root (default: results/)",
    )
    p.add_argument(
        "--path",
        default="",
        help="Inspect a single parquet (skip discovery)",
    )
    p.add_argument(
        "--strategy",
        default="",
        help="Filter paths containing strategy slug (e.g. tpc, bpc)",
    )
    p.add_argument(
        "--name",
        default="features_labeled.parquet",
        help="Filename to match (default: features_labeled.parquet)",
    )
    p.add_argument("--limit", type=int, default=30, help="Max files (default: 30)")
    p.add_argument("--json", action="store_true", help="JSON output")
    args = p.parse_args()

    if str(args.path).strip():
        path = Path(str(args.path).strip())
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            return 3
        rows = [summarize_parquet(path)]
    else:
        root = Path(str(args.root).strip())
        if not root.is_absolute():
            root = (PROJECT_ROOT / root).resolve()
        paths = discover_parquets(
            root,
            strategy=str(args.strategy).strip(),
            name=str(args.name).strip(),
            limit=int(args.limit),
        )
        rows = [summarize_parquet(path) for path in paths]

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(format_table(rows))
        print(f"\n({len(rows)} file(s); use --json for full metadata)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
