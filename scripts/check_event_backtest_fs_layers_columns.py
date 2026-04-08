#!/usr/bin/env python3
"""
Align FeatureStore Parquet column sets with what event_backtest.run() would use.

Mirrors:
  - timeframe from {strategies_root}/{strategy}/meta.yaml (same as event_backtest)
  - detect_layer_for_strategy(strategy, features_store_root, timeframe=_get_timeframe(s))
  - per-tf merge layer pick: first layer whose name splits contain the tf token (else first layer)

Usage (from repo root):
  python scripts/check_event_backtest_fs_layers_columns.py --strategy me --symbol BTCUSDT --month 2024-01
  python scripts/check_event_backtest_fs_layers_columns.py --strategy bpc,fer,me-long --symbol ETHUSDT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover
    print("pyarrow is required:", e, file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent

_TF_FALLBACK_WARNED: set[tuple[str, str]] = set()


def _resolve_strategies_root(raw: str) -> Path:
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _timeframe_from_strategy_meta(
    strategy: str, strategies_root: Path
) -> Optional[str]:
    meta_path = strategies_root / strategy / "meta.yaml"
    if not meta_path.is_file():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        tf = meta.get("timeframe")
        if isinstance(tf, str) and tf.strip():
            return tf.strip()
        st = meta.get("strategy")
        if isinstance(st, dict):
            tf = st.get("timeframe")
            if isinstance(tf, str) and tf.strip():
                return tf.strip()
    except Exception:
        return None
    return None


def _get_timeframe(strategy: str, *, strategies_root: Path) -> str:
    meta_tf = _timeframe_from_strategy_meta(strategy, strategies_root)
    if meta_tf:
        return meta_tf
    key = (str(strategies_root), strategy)
    if key not in _TF_FALLBACK_WARNED:
        _TF_FALLBACK_WARNED.add(key)
        print(
            f"warning: no timeframe in {strategies_root}/{strategy}/meta.yaml — using 240T",
            file=sys.stderr,
        )
    return "240T"


def _pick_merge_layer(
    fs_layers: Dict[str, str], tf: str
) -> Tuple[Optional[str], Optional[str]]:
    """Same as EventBacktester.run merge block."""
    for s, ln in fs_layers.items():
        if tf in ln.split("_"):
            return s, ln
    if fs_layers:
        s0 = next(iter(fs_layers.keys()))
        return s0, fs_layers[s0]
    return None, None


def _parquet_column_names(parquet_path: Path) -> List[str]:
    pf = pq.ParquetFile(parquet_path)
    return list(pf.schema_arrow.names)


def _layer_meta(fs_root: Path, layer: str) -> Optional[dict]:
    p = fs_root / f"{layer}.meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        default="me",
        help="Comma-separated strategy ids (same as event_backtest --strategy)",
    )
    parser.add_argument(
        "--strategies-root",
        default="config/strategies",
        help="Strategy configs root (default config/strategies), same as event_backtest",
    )
    parser.add_argument(
        "--feature-store-root",
        default="feature_store",
        help="FeatureStore root directory",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol folder under layer")
    parser.add_argument(
        "--month",
        default="2024-01",
        help="YYYY-MM parquet stem to inspect",
    )
    parser.add_argument(
        "--substr",
        action="append",
        default=["me_accel_5k"],
        help="Substring to grep in column names (repeatable). Default: me_accel_5k",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from src.feature_store.layer_naming import detect_layer_for_strategy

    strategies_root = _resolve_strategies_root(args.strategies_root)
    fs_root = (ROOT / args.feature_store_root).resolve()
    if not fs_root.is_dir():
        print(f"Feature store root missing: {fs_root}", file=sys.stderr)
        return 1

    strategies = [s.strip() for s in args.strategy.split(",") if s.strip()]
    _fs_layers: Dict[str, str] = {}
    print(
        "=== per-strategy detect_layer_for_strategy (same args as event_backtest.run) ==="
    )
    for s in strategies:
        tf = _get_timeframe(s, strategies_root=strategies_root)
        det = detect_layer_for_strategy(
            strategy=s,
            features_store_root=str(fs_root),
            timeframe=tf,
        )
        print(f"  strategy={s!r}  _get_timeframe={tf!r}  ->  layer={det!r}")
        if det:
            _fs_layers[s] = det
        else:
            loose = detect_layer_for_strategy(
                strategy=s,
                features_store_root=str(fs_root),
                timeframe=None,
            )
            if loose:
                meta = _layer_meta(fs_root, loose)
                mtf = (meta or {}).get("timeframe", "")
                print(
                    f"    (hint: with timeframe=None, latest match is {loose!r}, "
                    f"meta timeframe={mtf!r} — mismatch explains empty FS merge)"
                )

    print("\n=== event_backtest-style _fs_layers (non-empty entries only) ===")
    if not _fs_layers:
        print("(empty — event_backtest would skip FeatureStore merge)")
        return 0

    for s, ln in _fs_layers.items():
        meta = _layer_meta(fs_root, ln)
        meta_tf = (meta or {}).get("timeframe", "")
        meta_cfg = (meta or {}).get("config_dir", "")
        print(
            f"  {s!r} -> {ln!r}  (meta timeframe={meta_tf!r}, config_dir={meta_cfg!r})"
        )

    unique_tfs = sorted(
        set(_get_timeframe(s, strategies_root=strategies_root) for s in strategies)
    )
    print(f"\n=== unique timeframes from strategies: {unique_tfs} ===")

    for tf in unique_tfs:
        strat_pick, layer_pick = _pick_merge_layer(_fs_layers, tf)
        pq_path = fs_root / layer_pick / args.symbol / tf / f"{args.month}.parquet"
        print(
            f"\n--- tf={tf!r} merge would use layer={layer_pick!r} (first match strat={strat_pick!r}) ---"
        )
        print(f"    parquet: {pq_path}")
        if not pq_path.is_file():
            print("    (file missing)")
            continue
        cols = _parquet_column_names(pq_path)
        print(f"    n_columns (parquet schema): {len(cols)}")
        for sub in args.substr:
            hits = [c for c in cols if sub in c]
            print(f"    columns containing {sub!r}: {len(hits)}")
            for c in sorted(hits)[:50]:
                print(f"      {c}")
            if len(hits) > 50:
                print(f"      ... and {len(hits) - 50} more")

    print(
        "\nNote: IFC row has ~N keys after merge only if those columns are missing on IFC "
        "and successfully joined from FS (see event_backtest 'FeatureStore merged M cols')."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
