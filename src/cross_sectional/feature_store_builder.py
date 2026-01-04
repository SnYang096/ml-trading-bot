from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import yaml

from src.features.registry import get_feature_func


@dataclass(frozen=True)
class CSFeatureStoreBuildConfig:
    data_path: str = "data/parquet_data"
    features_store_root: str = "feature_store"
    features_store_layer: str = ""
    timeframe: str = "240T"
    start_date: str = ""
    end_date: str = ""
    warmup_bars: int = 600
    include_ohlcv: bool = True
    overwrite: bool = False


def compute_layer_name(
    *, factors: Sequence[str], timeframe: str, warmup_bars: int
) -> str:
    key = json.dumps(
        {
            "factors": list(factors),
            "timeframe": timeframe,
            "warmup_bars": int(warmup_bars),
        },
        sort_keys=True,
    )
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"cs_features_{h}"


def _month_key(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    start = pd.Timestamp(start).normalize().replace(day=1)
    end = pd.Timestamp(end).normalize().replace(day=1)
    return list(pd.date_range(start=start, end=end, freq="MS"))


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tz is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _store_month_path(
    *,
    root: Path,
    layer: str,
    symbol: str,
    timeframe: str,
    month_key: str,
) -> Path:
    return root / layer / symbol / timeframe / f"{month_key}.parquet"


def load_factor_set(*, factor_set_yaml: str, factor_set: str) -> List[str]:
    obj = yaml.safe_load(Path(factor_set_yaml).read_text(encoding="utf-8")) or {}
    sets = (obj.get("factor_sets") or {}) if isinstance(obj, dict) else {}
    if factor_set not in sets:
        raise KeyError(f"factor_set '{factor_set}' not found in {factor_set_yaml}")
    cols = [str(x).strip() for x in (sets.get(factor_set) or []) if str(x).strip()]
    if not cols:
        raise ValueError(f"factor_set '{factor_set}' is empty in {factor_set_yaml}")
    return cols


def _build_output_to_feature_key_map(feature_deps_path: str) -> Dict[str, str]:
    obj = yaml.safe_load(Path(feature_deps_path).read_text(encoding="utf-8")) or {}
    feats = (obj.get("features") or {}) if isinstance(obj, dict) else {}
    out: Dict[str, str] = {}
    for feature_key, meta in feats.items():
        for c in meta.get("output_columns") or []:
            c = str(c)
            # If collision exists, keep first (file order). This is fine for curated sets.
            out.setdefault(c, str(feature_key))
    return out


def _resolve_feature_keys_for_outputs(
    *,
    desired_output_cols: Sequence[str],
    feature_deps_path: str,
) -> List[str]:
    out2key = _build_output_to_feature_key_map(feature_deps_path)
    missing = [c for c in desired_output_cols if c not in out2key]
    if missing:
        raise KeyError(
            f"Unknown output columns (not found in feature_dependencies): {missing[:20]}"
        )
    # de-dup preserve order
    keys = []
    seen: Set[str] = set()
    for c in desired_output_cols:
        k = out2key[c]
        if k not in seen:
            keys.append(k)
            seen.add(k)
    return keys


def _load_feature_nodes(feature_deps_path: str) -> Dict[str, dict]:
    obj = yaml.safe_load(Path(feature_deps_path).read_text(encoding="utf-8")) or {}
    feats = (obj.get("features") or {}) if isinstance(obj, dict) else {}
    return {str(k): (v or {}) for k, v in feats.items()}


def _toposort_features(
    feature_keys: Sequence[str],
    nodes: Dict[str, dict],
) -> List[str]:
    order: List[str] = []
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def dfs(k: str) -> None:
        if k in visited:
            return
        if k in visiting:
            raise ValueError(f"Cycle detected in feature dependencies at: {k}")
        visiting.add(k)
        deps = nodes.get(k, {}).get("dependencies") or []
        for d in deps:
            d = str(d)
            if d not in nodes:
                raise KeyError(f"Unknown dependency '{d}' for feature '{k}'")
            dfs(d)
        visiting.remove(k)
        visited.add(k)
        order.append(k)

    for k in feature_keys:
        if k not in nodes:
            raise KeyError(f"Unknown feature key in dependencies YAML: {k}")
        dfs(k)
    return order


def _apply_feature_node(
    df: pd.DataFrame,
    *,
    feature_key: str,
    node: dict,
) -> pd.DataFrame:
    compute_func_name = str(node.get("compute_func"))
    if not compute_func_name:
        raise ValueError(f"{feature_key}: missing compute_func")
    func = get_feature_func(compute_func_name)

    pass_full_df = bool(node.get("pass_full_df", False))
    compute_params = node.get("compute_params") or {}
    mappings = node.get("column_mappings") or {}

    if pass_full_df:
        out = func(df, **compute_params)
        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"{feature_key}: compute_func returned {type(out)} (expected DataFrame)"
            )
        return out

    # narrow-io: build kwargs from column_mappings -> df columns
    kwargs = dict(compute_params)
    for arg_name, col_name in mappings.items():
        col_name = str(col_name)
        if col_name not in df.columns:
            raise KeyError(
                f"{feature_key}: missing required column '{col_name}' for arg '{arg_name}'"
            )
        kwargs[str(arg_name)] = df[col_name]

    out = func(**kwargs)
    if isinstance(out, pd.Series):
        return out.to_frame()
    if not isinstance(out, pd.DataFrame):
        raise TypeError(
            f"{feature_key}: compute_func returned {type(out)} (expected DataFrame/Series)"
        )
    return out


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = (
            df.index.tz_localize("UTC")
            if df.index.tz is None
            else df.index.tz_convert("UTC")
        )
        df = df.copy()
        df.index = idx
        return df
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        out = df.copy()
        out["timestamp"] = ts
        out = out.dropna(subset=["timestamp"]).set_index("timestamp")
        return out
    raise ValueError("Expected DatetimeIndex or a 'timestamp' column")


def build_symbol_month(
    *,
    df_raw: pd.DataFrame,
    symbol: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    nodes: Dict[str, dict],
    compute_order: Sequence[str],
    desired_output_cols: Sequence[str],
    include_ohlcv: bool,
) -> pd.DataFrame:
    """
    Compute desired features on a warmup+month slice, then return only rows in [month_start, month_end).
    """
    df = _ensure_datetime_index(df_raw)
    df = df.sort_index()

    # Work slice (includes warmup in df_raw already)
    mask = (df.index >= month_start) & (df.index < month_end)
    if not bool(mask.any()):
        return pd.DataFrame()

    work = df.copy()
    # Some pass_full_df features expect _symbol/symbol
    if "_symbol" not in work.columns:
        work["_symbol"] = str(symbol)
    if "symbol" not in work.columns:
        work["symbol"] = str(symbol)

    computed: Dict[str, pd.DataFrame] = {}
    for fk in compute_order:
        out = _apply_feature_node(work, feature_key=fk, node=nodes[fk])
        # align index
        out = _ensure_datetime_index(out)
        computed[fk] = out
        # merge into work for downstream deps
        for c in out.columns:
            work[c] = out[c]

    cols: List[str] = []
    if include_ohlcv:
        for c in ["open", "high", "low", "close", "volume"]:
            if c in work.columns:
                cols.append(c)
        # include common orderflow columns if present (still no ticks required)
        for c in ["cvd", "taker_buy_ratio", "buy_qty", "sell_qty", "delta"]:
            if c in work.columns:
                cols.append(c)

    # final factor outputs
    for c in desired_output_cols:
        if c in work.columns:
            cols.append(c)

    cols = list(dict.fromkeys(cols))  # de-dup preserve order
    out_month = work.loc[mask, cols].copy()
    out_month["symbol"] = str(symbol)
    return out_month


def build_feature_store_for_symbols(
    *,
    symbols: Sequence[str],
    desired_output_cols: Sequence[str],
    feature_deps_path: str,
    cfg: CSFeatureStoreBuildConfig,
) -> str:
    """
    Build monthly-parquet FeatureStore partitions for given symbols.
    Returns the resolved layer name.
    """
    if not cfg.features_store_layer:
        layer = compute_layer_name(
            factors=list(desired_output_cols),
            timeframe=str(cfg.timeframe),
            warmup_bars=int(cfg.warmup_bars),
        )
    else:
        layer = str(cfg.features_store_layer)

    root = Path(cfg.features_store_root)
    # Special mode: Alpha101 original cross-sectional-rank factors are computed
    # from multi-asset wide tables, not from feature_dependencies.yaml nodes.
    alpha_cs_cols = [
        c for c in desired_output_cols if str(c).startswith("alpha101_cs_")
    ]
    alpha_cs_ids: List[int] = []
    if alpha_cs_cols:
        for c in alpha_cs_cols:
            s = str(c).replace("alpha101_cs_", "").strip()
            try:
                alpha_cs_ids.append(int(s))
            except Exception:
                continue
        alpha_cs_ids = sorted(set(alpha_cs_ids))
    else:
        nodes = _load_feature_nodes(feature_deps_path)
        feature_keys = _resolve_feature_keys_for_outputs(
            desired_output_cols=list(desired_output_cols),
            feature_deps_path=feature_deps_path,
        )
        compute_order = _toposort_features(feature_keys, nodes)

    start_ts = pd.Timestamp(cfg.start_date, tz="UTC")
    end_ts = pd.Timestamp(cfg.end_date, tz="UTC") + pd.Timedelta(days=1)
    months = _month_starts(start_ts, end_ts)

    # Import here to avoid import cost for callers that only need layer name utilities
    from src.data_tools.data_utils import load_raw_data

    for sym in symbols:
        sym = str(sym).strip().upper()
        if not sym:
            continue
        if alpha_cs_ids:
            # alpha101_cs_* requires cross-sectional computation; handle below after loading all symbols.
            continue
        for ms in months:
            me = (ms + pd.offsets.MonthBegin(1)).to_pydatetime()
            month_start = _ensure_utc(ms)
            month_end = _ensure_utc(me)
            month_key = _month_key(month_start)
            out_path = _store_month_path(
                root=root,
                layer=layer,
                symbol=sym,
                timeframe=str(cfg.timeframe),
                month_key=month_key,
            )
            if out_path.exists() and not cfg.overwrite:
                continue

            # Warmup window: load extra history before month_start
            warmup = int(cfg.warmup_bars)
            # Estimate warmup time span from timeframe minutes if possible; fallback to bars as rows
            try:
                offset = pd.tseries.frequencies.to_offset(str(cfg.timeframe))
                seconds = getattr(offset, "delta", None)
                if seconds is not None:
                    warmup_td = pd.Timedelta(
                        seconds=int(seconds.total_seconds()) * warmup
                    )
                else:
                    warmup_td = pd.Timedelta(days=90)
            except Exception:
                warmup_td = pd.Timedelta(days=90)

            load_start = (month_start - warmup_td).strftime("%Y-%m-%d")
            load_end = (month_end - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d")

            try:
                df_raw = load_raw_data(
                    data_path=str(cfg.data_path),
                    symbol=sym,
                    start_date=str(load_start),
                    end_date=str(load_end),
                    timeframe=str(cfg.timeframe),
                )
            except Exception:
                # Missing symbol data is common in large universes; skip.
                continue
            if df_raw is None or df_raw.empty:
                continue

            out_month = build_symbol_month(
                df_raw=df_raw,
                symbol=sym,
                month_start=month_start,
                month_end=month_end,
                nodes=nodes,
                compute_order=compute_order,
                desired_output_cols=desired_output_cols,
                include_ohlcv=bool(cfg.include_ohlcv),
            )
            if out_month.empty:
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_month.to_parquet(out_path, index=True)

    if alpha_cs_ids:
        # Cross-sectional Alpha101 compute per month: load all symbols, compute once, then split.
        from src.cross_sectional.alpha101_cs_rank import compute_alpha101_cs_rank_panel

        # Warmup window shared across symbols
        warmup = int(cfg.warmup_bars)
        try:
            offset = pd.tseries.frequencies.to_offset(str(cfg.timeframe))
            seconds = getattr(offset, "delta", None)
            if seconds is not None:
                warmup_td = pd.Timedelta(seconds=int(seconds.total_seconds()) * warmup)
            else:
                warmup_td = pd.Timedelta(days=90)
        except Exception:
            warmup_td = pd.Timedelta(days=90)

        for ms in months:
            me = (ms + pd.offsets.MonthBegin(1)).to_pydatetime()
            month_start = _ensure_utc(ms)
            month_end = _ensure_utc(me)
            month_key = _month_key(month_start)

            # If all symbols already exist and overwrite disabled, skip month.
            if not cfg.overwrite:
                all_exist = True
                for sym in [str(s).strip().upper() for s in symbols]:
                    p = _store_month_path(
                        root=root,
                        layer=layer,
                        symbol=sym,
                        timeframe=str(cfg.timeframe),
                        month_key=month_key,
                    )
                    if not p.exists():
                        all_exist = False
                        break
                if all_exist:
                    continue

            load_start = (month_start - warmup_td).strftime("%Y-%m-%d")
            load_end = (month_end - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d")

            frames: Dict[str, pd.DataFrame] = {}
            for sym in [str(s).strip().upper() for s in symbols]:
                try:
                    df_raw = load_raw_data(
                        data_path=str(cfg.data_path),
                        symbol=sym,
                        start_date=str(load_start),
                        end_date=str(load_end),
                        timeframe=str(cfg.timeframe),
                    )
                except Exception:
                    # Missing symbol data is common in large universes; skip.
                    continue
                if df_raw is None or df_raw.empty:
                    continue
                frames[sym] = df_raw

            if not frames:
                continue

            panel = compute_alpha101_cs_rank_panel(frames, alpha_ids=alpha_cs_ids)
            if panel is None or panel.empty:
                continue

            # Keep only month rows (exclude warmup)
            ts = pd.to_datetime(
                panel.index.get_level_values("timestamp"), utc=True, errors="coerce"
            )
            mask = (ts >= month_start) & (ts < month_end)
            panel = panel.loc[mask].copy()
            if panel.empty:
                continue

            # Split per symbol and write
            available_syms = set(panel.index.get_level_values("symbol").astype(str))
            for sym in [str(s).strip().upper() for s in symbols]:
                if sym not in available_syms:
                    continue
                out_path = _store_month_path(
                    root=root,
                    layer=layer,
                    symbol=sym,
                    timeframe=str(cfg.timeframe),
                    month_key=month_key,
                )
                if out_path.exists() and not cfg.overwrite:
                    continue
                sub = panel.xs(sym, level="symbol", drop_level=False).copy()
                if cfg.include_ohlcv and sym in frames:
                    # attach OHLCV for the month (from raw)
                    raw = frames[sym]
                    raw = raw.copy()
                    if "timestamp" in raw.columns:
                        raw = raw.set_index("timestamp")
                    if not isinstance(raw.index, pd.DatetimeIndex):
                        pass
                    else:
                        idx = (
                            raw.index.tz_localize("UTC")
                            if raw.index.tz is None
                            else raw.index.tz_convert("UTC")
                        )
                        raw.index = idx
                    raw_month = raw.loc[
                        (raw.index >= month_start) & (raw.index < month_end), :
                    ]
                    # flatten to align by timestamp
                    raw_month = raw_month.copy()
                    raw_month["timestamp"] = raw_month.index
                    raw_month["symbol"] = sym
                    raw_month = raw_month.set_index(["timestamp", "symbol"])
                    for c in ["open", "high", "low", "close", "volume"]:
                        if c in raw_month.columns:
                            sub[c] = raw_month[c]
                # store as flat index=timestamp for consistency with existing rank loader
                df_out = sub.reset_index().set_index("timestamp").sort_index()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df_out.to_parquet(out_path, index=True)

    return layer
