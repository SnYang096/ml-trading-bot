"""Live A/B/C regime snapshots from feature bus + strategy yaml."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from mlbot_console.services.regime_ops import _load_regime_yaml, _resolve_regime_source
from src.monitoring.regime_health import (
    regime_shares_from_window,
    resolve_baseline_regime_shares,
)
from time_series_model.archetype.loader import RegimeConfig

_A_HINT = "A 只在深熊吸筹；牛市应少 deploy、多持币。"
_B_HINT = "B 吃 swing alpha；bull=结构退出，bear/neutral=trailing 保护。"
_C_HINT = "C 是短周期状态机；chop 高 ≠ 全局熊市。"


def _features_120t_path(feature_bus_root: Path, symbol: str) -> Path:
    return feature_bus_root / "features" / "120T" / f"{symbol.upper()}.parquet"


def load_features_120t_df(
    feature_bus_root: Path,
    symbol: str,
    *,
    window_days: int = 7,
) -> pd.DataFrame:
    path = _features_120t_path(feature_bus_root, symbol)
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    if window_days > 0:
        end = df["timestamp"].max()
        start = end - pd.Timedelta(days=window_days)
        df = df[df["timestamp"] >= start]
    return df.reset_index(drop=True)


def _row_features(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in row.index:
        val = row[col]
        if pd.isna(val):
            continue
        try:
            out[str(col)] = float(val) if isinstance(val, (int, float)) else val
        except (TypeError, ValueError):
            out[str(col)] = val
    return out


def _features_120t_latest_meta(
    feature_bus_root: Path, symbol: str
) -> Optional[Dict[str, Any]]:
    """Freshness for regime cockpit: 120T features only (not 1min bars)."""
    sym = symbol.upper()
    json_path = feature_bus_root / "latest" / "features" / "120T" / f"{sym}.json"
    if json_path.is_file():
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            ts = raw.get("timestamp")
            if ts:
                return {
                    "timestamp": str(ts),
                    "path": raw.get("path")
                    or str(_features_120t_path(feature_bus_root, sym)),
                    "kind": "features_120T",
                }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    feat_path = _features_120t_path(feature_bus_root, sym)
    if feat_path.is_file():
        try:
            df = pd.read_parquet(feat_path, columns=["timestamp"])
            if not df.empty:
                ts = pd.to_datetime(df["timestamp"], utc=True).max()
                return {
                    "timestamp": ts.isoformat(),
                    "path": str(feat_path),
                    "kind": "features_120T",
                }
        except Exception:
            pass
    return None


def _feature_bus_meta(
    feature_bus_root: Path, symbol: str, *, stale_minutes: int
) -> Dict[str, Any]:
    meta = _features_120t_latest_meta(feature_bus_root, symbol)
    age_minutes: Optional[float] = None
    stale = True
    as_of: Optional[str] = None
    if meta and meta.get("timestamp"):
        as_of = str(meta["timestamp"])
        try:
            ts = pd.Timestamp(as_of)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            age_minutes = (
                datetime.now(timezone.utc) - ts.to_pydatetime()
            ).total_seconds() / 60.0
            stale = age_minutes > float(stale_minutes)
        except (TypeError, ValueError):
            pass
    return {
        "path": str(_features_120t_path(feature_bus_root, symbol)),
        "as_of": as_of,
        "age_minutes": age_minutes,
        "stale": stale,
    }


def _macro_score_label(score: Optional[float]) -> str:
    if score is None:
        return "未知"
    if score >= 4:
        return "risk-on 成熟"
    if score >= 3:
        return "转换期"
    if score <= 2:
        return "risk-off"
    return "中性"


def _a_spot_layer(
    strategies_root: Path,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    _, data, source_label, present = _resolve_regime_source(
        strategies_root, "spot_accum_simple"
    )
    weekly = features.get("weekly_ema_200_position")
    try:
        weekly_f = float(weekly) if weekly is not None else None
    except (TypeError, ValueError):
        weekly_f = None

    deploy_allowed = False
    deploy_state = "UNKNOWN"
    if weekly_f is not None:
        if weekly_f < 0.0:
            deploy_state = "DEEP_BEAR"
            deploy_allowed = True
        else:
            deploy_state = "ABOVE_EMA200"
            deploy_allowed = False

    macro_raw = features.get("abc_macro_regime_score")
    try:
        macro_score = float(macro_raw) if macro_raw is not None else None
    except (TypeError, ValueError):
        macro_score = None

    return {
        "strategy": "spot_accum_simple",
        "regime_source": source_label,
        "present": present,
        "weekly_ema_200_position": weekly_f,
        "deploy_state": deploy_state,
        "deploy_allowed": deploy_allowed,
        "abc_macro_regime_score": macro_score,
        "macro_label": _macro_score_label(macro_score),
        "hint": _A_HINT,
    }


def _b_trend_layer(
    strategies_root: Path,
    project_root: Path,
    *,
    symbol: str,
    features: Dict[str, Any],
    window_df: pd.DataFrame,
    secondary_features: Optional[Dict[str, Any]] = None,
    secondary_symbol: Optional[str] = None,
) -> Dict[str, Any]:
    path, data, source_label, present = _resolve_regime_source(strategies_root, "tpc")
    cfg = RegimeConfig.from_mapping(data) if present else RegimeConfig()
    label = cfg.classify(features) if features else "neutral"
    sec_label: Optional[str] = None
    if secondary_features:
        sec_label = cfg.classify(secondary_features)

    bull_share_7d = 0.0
    if not window_df.empty:
        shares = regime_shares_from_window(window_df, data if present else {})
        bull_share_7d = float(shares.get("bull") or 0.0)

    baseline_entry: Dict[str, Any] = {}
    baseline_path = project_root / "config" / "monitoring" / "regime_watchdog_baseline.json"
    if baseline_path.is_file():
        try:
            doc = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_entry = doc.get("tpc") or {}
        except json.JSONDecodeError:
            pass
    baseline_shares = resolve_baseline_regime_shares(
        regime_yaml=data if present else {},
        baseline_entry=baseline_entry,
    )
    baseline_bull = float((baseline_shares or {}).get("bull") or 0.0)
    drift_alert = abs(bull_share_7d - baseline_bull) >= 0.10

    descriptions = {}
    ar = data.get("allowed_regimes") if isinstance(data.get("allowed_regimes"), dict) else {}
    for k, v in ar.items():
        if isinstance(v, dict) and v.get("description"):
            descriptions[str(k)] = str(v["description"])

    return {
        "strategy": "tpc",
        "regime_source": source_label,
        "present": present,
        "symbol": symbol.upper(),
        "current_label": label,
        "features": {
            k: features.get(k)
            for k in ("adx_50", "ema_1200_position")
            if features.get(k) is not None
        },
        "bull_share_7d": bull_share_7d,
        "baseline_bull_share": baseline_bull,
        "drift_alert": drift_alert,
        "label_descriptions": descriptions,
        "divergence": (
            {
                "symbol": secondary_symbol,
                "label": sec_label,
            }
            if sec_label is not None and sec_label != label
            else None
        ),
        "hint": _B_HINT,
    }


def _multileg_state(
    *,
    feature_value: Optional[float],
    entry_min: float,
    exit_below: float,
    prefix: str,
) -> str:
    if feature_value is None:
        return f"{prefix}_UNKNOWN"
    if feature_value >= entry_min:
        return f"{prefix}_ENTRY"
    if feature_value < exit_below:
        return f"{prefix}_EXIT"
    return f"{prefix}_HOLD"


def _c_multileg_layer(
    strategies_root: Path,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    chop_path, chop_data, _, chop_present = _resolve_regime_source(
        strategies_root, "chop_grid"
    )
    scalp_path, scalp_data, _, scalp_present = _resolve_regime_source(
        strategies_root, "trend_scalp"
    )

    chop_ml = (
        (chop_data.get("extensions") or {}).get("multileg") or {}
        if chop_present
        else {}
    )
    scalp_ml = (
        (scalp_data.get("extensions") or {}).get("multileg") or {}
        if scalp_present
        else {}
    )

    chop_feat = features.get(str(chop_ml.get("entry_feature") or "bpc_semantic_chop"))
    try:
        chop_val = float(chop_feat) if chop_feat is not None else None
    except (TypeError, ValueError):
        chop_val = None

    scalp_feat_name = str(scalp_ml.get("entry_feature") or "trend_confidence")
    scalp_feat = features.get(scalp_feat_name)
    try:
        scalp_val = float(scalp_feat) if scalp_feat is not None else None
    except (TypeError, ValueError):
        scalp_val = None

    chop_entry = float(chop_ml.get("entry_min") or 0.52)
    chop_exit = float(chop_ml.get("exit_below") or 0.33)
    scalp_entry = float(scalp_ml.get("entry_min") or 0.7)
    scalp_cap = float(scalp_ml.get("cap_entry") or 0.25)

    chop_state = _multileg_state(
        feature_value=chop_val,
        entry_min=chop_entry,
        exit_below=chop_exit,
        prefix="CHOP",
    )
    scalp_state = "BELOW_ENTRY"
    if scalp_val is not None:
        if scalp_val >= scalp_entry and (chop_val is None or chop_val <= scalp_cap):
            scalp_state = "MOMENTUM_ENTRY"
        elif scalp_val >= scalp_entry:
            scalp_state = "BLOCKED_BY_CHOP"
        else:
            scalp_state = "BELOW_ENTRY"

    router_hint = "neutral"
    if chop_state == "CHOP_ENTRY":
        router_hint = "chop_favored"
    elif scalp_state == "MOMENTUM_ENTRY":
        router_hint = "momentum_favored"
    elif chop_state == "CHOP_HOLD" and scalp_state == "BELOW_ENTRY":
        router_hint = "chop_neutral"

    return {
        "chop_grid": {
            "present": chop_present,
            "feature": str(chop_ml.get("entry_feature") or "bpc_semantic_chop"),
            "value": chop_val,
            "entry_min": chop_entry,
            "exit_below": chop_exit,
            "state": chop_state,
        },
        "trend_scalp": {
            "present": scalp_present,
            "feature": scalp_feat_name,
            "value": scalp_val,
            "entry_min": scalp_entry,
            "cap_chop": scalp_cap,
            "state": scalp_state,
        },
        "router_hint": router_hint,
        "hint": _C_HINT,
    }


def build_live_layers(
    *,
    strategies_root: Path,
    project_root: Path,
    feature_bus_root: Path,
    symbol: str = "BTCUSDT",
    window_days: int = 7,
    stale_minutes: int = 240,
    secondary_symbol: str = "ETHUSDT",
) -> Dict[str, Any]:
    sym = symbol.upper()
    bus_meta = _feature_bus_meta(feature_bus_root, sym, stale_minutes=stale_minutes)
    window_df = load_features_120t_df(feature_bus_root, sym, window_days=window_days)
    latest_features: Dict[str, Any] = {}
    if not window_df.empty:
        latest_features = _row_features(window_df.iloc[-1])

    sec_features: Optional[Dict[str, Any]] = None
    sec_df = load_features_120t_df(feature_bus_root, secondary_symbol, window_days=1)
    if not sec_df.empty:
        sec_features = _row_features(sec_df.iloc[-1])

    return {
        "symbol": sym,
        "feature_bus": bus_meta,
        "layers": {
            "a_spot": _a_spot_layer(strategies_root, latest_features),
            "b_trend": _b_trend_layer(
                strategies_root,
                project_root,
                symbol=sym,
                features=latest_features,
                window_df=window_df,
                secondary_features=sec_features,
                secondary_symbol=secondary_symbol.upper(),
            ),
            "c_multileg": _c_multileg_layer(strategies_root, latest_features),
        },
        "composite_context": _composite_context_from_layers(
            latest_features, window_df, strategies_root, project_root, sym
        ),
    }


def _composite_context_from_layers(
    features: Dict[str, Any],
    window_df: pd.DataFrame,
    strategies_root: Path,
    project_root: Path,
    symbol: str,
) -> Dict[str, Any]:
    a = _a_spot_layer(strategies_root, features)
    _, tpc_data, _, tpc_present = _resolve_regime_source(strategies_root, "tpc")
    cfg = RegimeConfig.from_mapping(tpc_data) if tpc_present else RegimeConfig()
    bull_label = cfg.classify(features) if features else "neutral"
    bull_share = 0.0
    if not window_df.empty and tpc_present:
        bull_share = float(
            regime_shares_from_window(window_df, tpc_data).get("bull") or 0.0
        )
    c = _c_multileg_layer(strategies_root, features)
    chop_val = c["chop_grid"].get("value")
    scalp_val = c["trend_scalp"].get("value")
    macro = a.get("abc_macro_regime_score")
    weekly = a.get("weekly_ema_200_position")
    return {
        "abc_macro_regime_score": macro,
        "weekly_ema_200_position": weekly,
        "tpc_bull_share_7d": bull_share,
        "tpc_bull_label": bull_label,
        "chop_semantic": chop_val,
        "trend_confidence": scalp_val,
        "symbol": symbol,
    }
