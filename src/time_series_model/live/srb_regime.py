"""
SRB 实验辅助：ADX、路径效率、SR 摆动位 — 供事件回测注入 features 与执行层分档。

- ADX / ER 仅使用已收盘 primary bar 的 OHLC 序列（因果）。
- SR 支撑/阻力：入场 bar 前 lookback 根 K 的 low 最小 / high 最大（方案 3b：事件层独立算法）。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def path_efficiency_last(close: np.ndarray, window: int) -> float:
    """Kaufman 风格路径效率：|C_t - C_{t-w}| / sum(|ΔC|)，取最后一窗。"""
    if close is None or len(close) < window + 1:
        return float("nan")
    seg = close[-(window + 1) :]
    net = abs(float(seg[-1] - seg[0]))
    path = float(np.sum(np.abs(np.diff(seg.astype(float)))))
    if path <= 1e-12:
        return float("nan")
    return float(net / path)


def adx14_last(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> float:
    """Wilder ADX，返回最后一根的值。"""
    n = len(close)
    if n < period + 2 or high is None or low is None:
        return float("nan")
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([np.abs(h - l), np.abs(h - prev_c), np.abs(l - prev_c)])
    up = h - np.roll(h, 1)
    down = -(l - np.roll(l, 1))
    up[0] = down[0] = 0.0
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    alpha = 1.0 / float(period)
    tr14 = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
    pdi = (
        100.0
        * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
        / tr14.clip(lower=1e-12)
    )
    mdi = (
        100.0
        * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
        / tr14.clip(lower=1e-12)
    )
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).clip(lower=1e-12)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    v = float(adx.iloc[-1])
    return v if np.isfinite(v) else float("nan")


def swing_sr_levels(
    df: pd.DataFrame,
    ts: pd.Timestamp,
    lookback: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    在 ts 及之前已收盘的 primary bar 上，用最近 lookback 根 K 计算：
    - 支撑：min(low)
    - 阻力：max(high)
    若缺 high/low，则退回 close。
    """
    if df is None or df.empty or lookback < 1:
        return None, None
    try:
        sub = df.loc[:ts].tail(lookback)
    except Exception:
        return None, None
    if sub.empty or len(sub) < max(3, min(lookback, len(sub))):
        return None, None
    if "low" in sub.columns:
        sup = float(sub["low"].astype(float).min())
    elif "close" in sub.columns:
        sup = float(sub["close"].astype(float).min())
    else:
        sup = None
    if "high" in sub.columns:
        res = float(sub["high"].astype(float).max())
    elif "close" in sub.columns:
        res = float(sub["close"].astype(float).max())
    else:
        res = None
    if sup is not None and res is not None and sup > res:
        sup, res = res, sup
    return sup, res


def should_reject_srb_wide_entry(
    side: str,
    close: float,
    atr: float,
    support_wide: Optional[float],
    resistance_wide: Optional[float],
    min_distance_atr: float,
) -> bool:
    """
    宽窗 SR 入场屏蔽（与 ``scripts/event_backtest.py`` PCM 路径一致）。

    LONG: 若上方宽窗阻力距现价 < min_distance_atr × ATR → True（拒绝）。
    SHORT: 若下方宽窗支撑距现价 < min_distance_atr × ATR → True。

    min_distance_atr≤0 / 无效价格或 ATR 时不拦截。
    """
    try:
        mn = float(min_distance_atr)
        px = float(close)
        a = float(atr)
    except (TypeError, ValueError):
        return False
    if mn <= 0 or not (px > 0) or not (a > 0):
        return False
    su = support_wide
    re = resistance_wide
    try:
        sw = float(su) if su is not None and su == su else 0.0
        rw = float(re) if re is not None and re == re else 0.0
    except (TypeError, ValueError):
        sw = rw = 0.0

    u = str(side or "").upper()
    if u in ("LONG", "BUY"):
        # 阻力在价格上方且过近
        if rw > px and (rw - px) < mn * a:
            return True
    elif u in ("SHORT", "SELL"):
        # 支撑在价格下方且过近
        if 0 < sw < px and (px - sw) < mn * a:
            return True
    return False


def pick_srb_true_sr_level(
    side: str,
    entry_px: float,
    atr: float,
    *,
    narrow_support: Optional[float],
    narrow_resistance: Optional[float],
    wide_support: Optional[float],
    wide_resistance: Optional[float],
    fallback_atr: float,
) -> float:
    """
    fake_break_reverse Stage 2 锚点：默认用窄窗 SR（突破侧）；若窄窗距入场过近则用宽窗。

    与 ``event_backtest.open_position`` 后写入 ``_srb_true_sr_level`` 的逻辑一致。
    """
    try:
        ep = float(entry_px)
        ae = float(atr)
        fb = float(fallback_atr)
    except (TypeError, ValueError):
        return float(entry_px or 0)

    def _f(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            v = float(x)
            return v if v == v and np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    ns = _f(narrow_support)
    nr = _f(narrow_resistance)
    ws = _f(wide_support)
    wr = _f(wide_resistance)

    u = str(side or "").upper()
    if u in ("LONG", "BUY"):
        narrow = ns
        wide = ws
    elif u in ("SHORT", "SELL"):
        narrow = nr
        wide = wr
    else:
        return ep

    pick = float(narrow) if narrow is not None else ep
    if (
        fb > 0
        and narrow is not None
        and ae > 0
        and wide is not None
        and abs(ep - narrow) < fb * ae
    ):
        pick = float(wide)
    return float(pick)


def resolve_regime_bucket(
    adx: float,
    er: float,
    th: Dict[str, float],
) -> str:
    """返回 bucket key: high_adx_low_er / high_adx_high_er / low_adx_low_er / low_adx_high_er / unknown"""
    adx_hi = th.get("adx_high", 40.0)
    er_hi = th.get("er_high", 0.36)
    if not (adx == adx) or not (er == er):
        return "unknown"
    ah = adx >= adx_hi
    eh = er >= er_hi
    if ah and not eh:
        return "high_adx_low_er"
    if ah and eh:
        return "high_adx_high_er"
    if not ah and not eh:
        return "low_adx_low_er"
    return "low_adx_high_er"


def maybe_inject_srb_experiment_features(
    *,
    df: pd.DataFrame,
    ts: pd.Timestamp,
    exec_raw: Dict[str, Any],
    out: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将 SRB 实验用字段合并到 out（通常为 primary_features）。
    exec_raw: srb archetypes/execution.yaml 的 raw dict。
    """
    re_cfg = (exec_raw or {}).get("regime_execution") or {}
    sr_cfg = (exec_raw or {}).get("sr_structural_exit") or {}
    add_pol = (exec_raw or {}).get("srb_add_position_policy") or {}
    rev_cfg = (exec_raw or {}).get("fake_break_reverse") or {}
    sr_inj = (exec_raw or {}).get("sr_feature_injection") or {}
    wide_guard = (exec_raw or {}).get("sr_wide_entry_guard") or {}
    need_regime = bool(re_cfg.get("enabled")) or (
        bool(add_pol.get("enabled"))
        and isinstance(add_pol.get("allow_regime_buckets"), list)
        and len(add_pol.get("allow_regime_buckets") or []) > 0
    )
    wide_lb = int(sr_inj.get("swing_lookback_wide_bars") or 0)
    need_sr = (
        bool(sr_cfg.get("enabled"))
        or bool(rev_cfg.get("enabled"))
        or wide_lb > 0
        or bool(wide_guard.get("enabled"))
    )
    if not need_regime and not need_sr:
        return out

    er_w = int(
        re_cfg.get("er_window_bars") or add_pol.get("regime_er_window_bars") or 20
    )
    adx_p = int(re_cfg.get("adx_period") or add_pol.get("regime_adx_period") or 14)
    lb = int(sr_cfg.get("lookback_bars", 20) or 20)

    if "close" not in df.columns:
        return out
    try:
        hist = df.loc[:ts]
    except Exception:
        return out
    if hist.empty:
        return out

    close = hist["close"].astype(float).values
    if need_regime:
        need = max(er_w + 2, adx_p * 2 + 2)
        if len(close) >= need:
            high = (
                hist["high"].astype(float).values if "high" in hist.columns else close
            )
            low = hist["low"].astype(float).values if "low" in hist.columns else close
            adx_v = adx14_last(high, low, close, period=adx_p)
            er_v = path_efficiency_last(close, er_w)
            if adx_v == adx_v:
                out["srb_regime_adx14"] = float(adx_v)
            if er_v == er_v:
                out["srb_regime_er20"] = float(er_v)
            th = dict(re_cfg.get("thresholds") or {})
            if not th:
                th = dict(add_pol.get("regime_thresholds") or {})
            out["srb_regime_bucket"] = resolve_regime_bucket(
                float(adx_v) if adx_v == adx_v else float("nan"),
                float(er_v) if er_v == er_v else float("nan"),
                {
                    "adx_high": float(th.get("adx_high", 40.0)),
                    "er_high": float(th.get("er_high", 0.36)),
                },
            )

    if need_sr:
        sup, res = swing_sr_levels(df, ts, max(3, lb))
        if sup is not None:
            out["srb_sr_support"] = float(sup)
        if res is not None:
            out["srb_sr_resistance"] = float(res)
        wide_bars = wide_lb
        if wide_bars <= 0 and bool(wide_guard.get("enabled")):
            wide_bars = 96  # 与 archetypes 默认 swing_lookback_wide_bars 对齐
        if wide_bars > 0:
            sup_w, res_w = swing_sr_levels(df, ts, max(3, wide_bars))
            if sup_w is not None:
                out["srb_sr_support_wide"] = float(sup_w)
            if res_w is not None:
                out["srb_sr_resistance_wide"] = float(res_w)

    return out


def srb_add_position_allowed(
    features: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> Tuple[bool, str]:
    """
    SRB 加仓门控（事件回测 signal-add 与 float_r_ladder 共用）。

    policy (execution.yaml → srb_add_position_policy):
      - enabled: 为 true 时才检查；否则恒通过。
      - allow_regime_buckets: 非空 list 时，仅当 ``srb_regime_bucket`` 在此集合内才允许加仓。
      - max_volume_compression_pct: 当前 ``volume_compression_feature`` **大于** 该阈值则禁止加仓。
      - volume_compression_feature: 默认 ``bpc_volume_compression_pct``。
    """
    if not policy or not bool(policy.get("enabled")):
        return True, ""
    allow = policy.get("allow_regime_buckets")
    if isinstance(allow, list) and len(allow) > 0:
        allowed_set = {str(x).strip() for x in allow}
        bucket = str(features.get("srb_regime_bucket", "unknown"))
        if bucket not in allowed_set:
            return False, "srb_policy_regime_bucket"
    max_pct = policy.get("max_volume_compression_pct")
    if max_pct is not None:
        feat_key = str(
            policy.get("volume_compression_feature") or "bpc_volume_compression_pct"
        )
        raw = features.get(feat_key)
        if raw is not None:
            try:
                fv = float(raw)
                if fv == fv and fv > float(max_pct):
                    return False, "srb_policy_volume_compression"
            except (TypeError, ValueError):
                pass
    return True, ""
