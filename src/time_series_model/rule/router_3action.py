from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule3ActionConfig:
    """
    Pure rule router based only on nnmultihead heads (no extra strategy features).

    It maps each timestamp into one of:
      - NO_TRADE
      - MEAN
      - TREND

    Assumptions:
    - pred_mfe_atr/pred_mae_atr/pred_t_to_mfe are in "training space" by default (log1p),
      unless preds_in_log1p=False.
    """

    # Input columns
    pred_dir_prob_col: str = "pred_dir_prob"
    pred_mfe_col: str = "pred_mfe_atr"
    pred_mae_col: str = "pred_mae_atr"
    pred_ttm_col: str = "pred_t_to_mfe"

    # Decision thresholds (in ATR units after inverse-transform)
    mfe_min: float = 0.4
    eff_min: float = 1.05  # mfe/(mae+eps) minimum to trade

    # Trend conditions
    dir_conf_trend_min: float = 0.25  # abs(p-0.5)*2
    mfe_trend_min: float = 0.8
    ttm_trend_min: float = 8.0
    # Trend confirm mode:
    # - "and" (legacy): dir_conf AND mfe AND ttm
    # - "or": dir_conf AND (mfe OR ttm)
    trend_confirm_mode: str = "and"

    # Mean conditions
    eff_mean_min: float = 1.15
    ttm_mean_max: float = 12.0

    eps: float = 1e-9


@dataclass(frozen=True)
class RegimeRuleConfig:
    """
    Rule-based regime qualifier. Uses feature columns, not model heads.

    Regime priority: TC > TE > MEAN > NO_TRADE
    """

    # Column names
    adx_col: str = "adx"
    adx_slope_col: Optional[str] = None
    sma_200_position_col: str = "sma_200_position"
    sma_200_slope_col: Optional[str] = "sma_200_slope"
    sr_distance_col: str = "sr_distance_normalized"
    sqs_col: str = "sqs"
    atr_percentile_col: str = "atr_percentile"

    # Thresholds
    trend_adx_min: float = 25.0
    trend_ma200_pos_min: float = 0.0

    mean_adx_max: float = 25.0
    mean_sr_max: float = 0.4
    mean_sqs_min: float = 0.2

    te_adx_min: float = 15.0
    te_adx_slope_min: float = 0.0
    te_use_ma200_cross: bool = True

    # Soft regime scoring (optional)
    use_soft_scores: bool = False
    min_regime_score: float = 0.2
    soft_profile_name: str = "plateau_open_khalf_v1"
    tc_score_floor: Optional[float] = None
    te_score_floor: Optional[float] = None
    mean_score_floor: Optional[float] = None

    tc_adx_center: Optional[float] = None
    tc_adx_k: float = 0.1
    tc_ma200_center: Optional[float] = None
    tc_ma200_k: float = 4.0

    te_adx_slope_center: Optional[float] = None
    te_adx_slope_k: float = 25.0
    te_ma200_center: Optional[float] = None
    te_ma200_k: float = 4.0

    mean_adx_center: Optional[float] = None
    mean_adx_k: float = 0.1
    mean_sr_center: Optional[float] = None
    mean_sr_k: float = 2.5
    mean_sqs_center: Optional[float] = None
    mean_sqs_k: float = 1.0

    # Extreme market veto (set to None to disable)
    extreme_atr_percentile_max: Optional[float] = 0.9

    # Missing handling
    missing_default_false: bool = True


@dataclass(frozen=True)
class QualityScoreConfig:
    """
    Trade quality score for Regime->Execution gating.
    """

    quality_trend_min: float = 0.6
    quality_mean_min: float = 0.8
    quality_te_min: float = 0.45

    # If True, use eff * dir_conf. Otherwise use eff only.
    use_dir_conf: bool = True


def _dir_conf(p: np.ndarray) -> np.ndarray:
    # p in [0,1] -> confidence in [0,1]
    return np.clip(np.abs(p - 0.5) * 2.0, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _maybe_expm1(x: np.ndarray, *, preds_in_log1p: bool) -> np.ndarray:
    if not preds_in_log1p:
        return x
    # model heads are strictly positive; expm1 keeps 0->0
    # Clip to avoid overflow when preds are extreme (e.g. bad model weights / untrained run).
    # expm1(15) ~= 3.27e6 which is already "effectively infinite" for our thresholds.
    return np.expm1(np.clip(x, 0.0, 15.0))


def _apply_temperature_scaling(
    p: np.ndarray, *, temperature: float, clip: tuple[float, float]
) -> np.ndarray:
    temp = float(temperature) if temperature else 1.0
    lo, hi = clip
    p = np.clip(p, lo, hi)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-(logit / temp)))


def _apply_isotonic_scaling(
    p: np.ndarray, *, clip: tuple[float, float], config: Dict[str, Any]
) -> np.ndarray:
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "isotonic calibration requires scikit-learn. "
            "Install scikit-learn or use temperature scaling."
        ) from exc
    lo, hi = clip
    p = np.clip(p, lo, hi)
    x = np.array(config.get("x", []), dtype=float)
    y = np.array(config.get("y", []), dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        raise ValueError("isotonic calibration requires matching x/y arrays.")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    return np.clip(iso.transform(p), lo, hi)


def _apply_linear_calibration(
    x: np.ndarray, *, scale: float, bias: float, clip: tuple[float, float]
) -> np.ndarray:
    lo, hi = clip
    return np.clip(x * float(scale) + float(bias), lo, hi)


def _apply_router_calibration(
    p: np.ndarray,
    mfe_p: np.ndarray,
    mae_p: np.ndarray,
    ttm_p: np.ndarray,
    *,
    calibration: Dict[str, Any],
    preds_in_log1p: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not calibration:
        return p, mfe_p, mae_p, ttm_p
    space = str(calibration.get("space", "log1p")).strip().lower()
    if space == "log1p" and not preds_in_log1p:
        raise ValueError(
            "router calibration expects log1p preds, but preds_in_log1p=False."
        )
    # dir prob calibration
    dir_cfg = calibration.get("dir_prob") or {}
    method = str(dir_cfg.get("method", "temperature")).lower()
    clip = tuple(dir_cfg.get("clip", [1e-6, 1 - 1e-6]))
    if method == "temperature":
        p = _apply_temperature_scaling(
            p, temperature=float(dir_cfg.get("temperature", 1.0)), clip=clip
        )
    elif method == "isotonic":
        p = _apply_isotonic_scaling(p, clip=clip, config=dir_cfg)
    else:
        raise ValueError(f"Unknown dir_prob calibration method: {method}")

    # linear bias correction in log space for regression heads
    for key, arr in (
        ("mfe_atr", mfe_p),
        ("mae_atr", mae_p),
        ("t_to_mfe", ttm_p),
    ):
        cfg = calibration.get(key) or {}
        if not cfg:
            continue
        arr[:] = _apply_linear_calibration(
            arr,
            scale=float(cfg.get("scale", 1.0)),
            bias=float(cfg.get("bias", 0.0)),
            clip=tuple(cfg.get("clip", [0.0, 15.0])),
        )
    return p, mfe_p, mae_p, ttm_p


def _slope(arr: np.ndarray, *, window: int = 3) -> np.ndarray:
    if window <= 0:
        return np.zeros_like(arr, dtype=float)
    prev = np.roll(arr, window)
    prev[:window] = np.nan
    return arr - prev


def compute_regime_rules(
    df: pd.DataFrame,
    *,
    cfg: RegimeRuleConfig = RegimeRuleConfig(),
) -> pd.DataFrame:
    """
    Compute rule-based regime: NO_TRADE / MEAN / TE / TREND.
    """
    missing = [
        c
        for c in [
            cfg.adx_col,
            cfg.sma_200_position_col,
            cfg.sr_distance_col,
            cfg.sqs_col,
        ]
        if c not in df.columns
    ]
    if missing and not cfg.missing_default_false:
        raise KeyError(f"Missing required regime columns: {missing}")

    def _col_or_nan(col: str) -> pd.Series:
        if col in df.columns:
            return df[col]
        return pd.Series(np.nan, index=df.index)

    adx = pd.to_numeric(_col_or_nan(cfg.adx_col), errors="coerce").to_numpy(dtype=float)
    sma_pos = pd.to_numeric(
        _col_or_nan(cfg.sma_200_position_col), errors="coerce"
    ).to_numpy(dtype=float)
    sr_dist = pd.to_numeric(_col_or_nan(cfg.sr_distance_col), errors="coerce").to_numpy(
        dtype=float
    )
    sqs = pd.to_numeric(_col_or_nan(cfg.sqs_col), errors="coerce").to_numpy(dtype=float)
    atr_pct = pd.to_numeric(
        _col_or_nan(cfg.atr_percentile_col), errors="coerce"
    ).to_numpy(dtype=float)

    if cfg.adx_slope_col and cfg.adx_slope_col in df.columns:
        adx_slope = pd.to_numeric(
            _col_or_nan(cfg.adx_slope_col), errors="coerce"
        ).to_numpy(dtype=float)
    else:
        adx_slope = _slope(adx, window=3)

    if cfg.sma_200_slope_col and cfg.sma_200_slope_col in df.columns:
        sma_slope = pd.to_numeric(
            _col_or_nan(cfg.sma_200_slope_col), errors="coerce"
        ).to_numpy(dtype=float)
    else:
        sma_slope = _slope(sma_pos, window=3)

    # Handle missing
    if cfg.missing_default_false:
        adx = np.where(np.isfinite(adx), adx, -np.inf)
        sma_pos = np.where(np.isfinite(sma_pos), sma_pos, -np.inf)
        sr_dist = np.where(np.isfinite(sr_dist), sr_dist, np.inf)
        sqs = np.where(np.isfinite(sqs), sqs, -np.inf)
        atr_pct = np.where(np.isfinite(atr_pct), atr_pct, -np.inf)
        adx_slope = np.where(np.isfinite(adx_slope), adx_slope, -np.inf)
        sma_slope = np.where(np.isfinite(sma_slope), sma_slope, -np.inf)

    if cfg.use_soft_scores:
        # Soft regime scoring
        tc_adx_center = (
            float(cfg.tc_adx_center)
            if cfg.tc_adx_center is not None
            else float(cfg.trend_adx_min)
        )
        tc_ma_center = (
            float(cfg.tc_ma200_center)
            if cfg.tc_ma200_center is not None
            else float(cfg.trend_ma200_pos_min)
        )
        te_slope_center = (
            float(cfg.te_adx_slope_center)
            if cfg.te_adx_slope_center is not None
            else float(cfg.te_adx_slope_min)
        )
        te_ma_center = (
            float(cfg.te_ma200_center)
            if cfg.te_ma200_center is not None
            else float(cfg.trend_ma200_pos_min)
        )
        mean_adx_center = (
            float(cfg.mean_adx_center)
            if cfg.mean_adx_center is not None
            else float(cfg.mean_adx_max)
        )
        mean_sr_center = (
            float(cfg.mean_sr_center)
            if cfg.mean_sr_center is not None
            else float(cfg.mean_sr_max)
        )
        mean_sqs_center = (
            float(cfg.mean_sqs_center)
            if cfg.mean_sqs_center is not None
            else float(cfg.mean_sqs_min)
        )

        tc_score = _sigmoid((adx - tc_adx_center) * float(cfg.tc_adx_k)) * _sigmoid(
            (sma_pos - tc_ma_center) * float(cfg.tc_ma200_k)
        )
        te_score = _sigmoid(
            (adx_slope - te_slope_center) * float(cfg.te_adx_slope_k)
        ) * _sigmoid((sma_pos - te_ma_center) * float(cfg.te_ma200_k))
        mean_score = (
            _sigmoid((mean_adx_center - adx) * float(cfg.mean_adx_k))
            * _sigmoid((mean_sr_center - sr_dist) * float(cfg.mean_sr_k))
            * _sigmoid((sqs - mean_sqs_center) * float(cfg.mean_sqs_k))
        )

        if cfg.te_use_ma200_cross:
            prev_pos = np.roll(sma_pos, 1)
            prev_pos[0] = np.nan
            ma200_cross = (sma_pos > 0.0) & (prev_pos <= 0.0)
        else:
            ma200_cross = sma_pos > 0.0
        te_score = te_score * (ma200_cross | (sma_slope > 0.0)).astype(float)

        # Extreme veto: in extreme volatility, force NO_TRADE
        if cfg.extreme_atr_percentile_max is not None:
            extreme = atr_pct >= float(cfg.extreme_atr_percentile_max)
            tc_score = np.where(extreme, 0.0, tc_score)
            te_score = np.where(extreme, 0.0, te_score)
            mean_score = np.where(extreme, 0.0, mean_score)

        scores = np.stack([tc_score, te_score, mean_score], axis=1)
        max_score = np.nanmax(scores, axis=1)
        best_idx = np.nanargmax(scores, axis=1)
        labels = np.array(["TC", "TE", "MEAN"], dtype=object)
        regime = np.where(
            max_score >= float(cfg.min_regime_score),
            labels[best_idx],
            "NO_TRADE",
        )

        # Weak guardrail: per-regime score floors (optional)
        if cfg.tc_score_floor is not None:
            regime = np.where(
                (regime == "TC") & (tc_score < float(cfg.tc_score_floor)),
                "NO_TRADE",
                regime,
            )
        if cfg.te_score_floor is not None:
            regime = np.where(
                (regime == "TE") & (te_score < float(cfg.te_score_floor)),
                "NO_TRADE",
                regime,
            )
        if cfg.mean_score_floor is not None:
            regime = np.where(
                (regime == "MEAN") & (mean_score < float(cfg.mean_score_floor)),
                "NO_TRADE",
                regime,
            )
    else:
        # Regime rules (hard thresholds)
        # TC (Trend Continuation): strong trend, price above MA200
        tc = (adx >= float(cfg.trend_adx_min)) & (
            sma_pos >= float(cfg.trend_ma200_pos_min)
        )

        mean = (
            (adx <= float(cfg.mean_adx_max))
            & (sr_dist <= float(cfg.mean_sr_max))
            & (sqs >= float(cfg.mean_sqs_min))
        )

        if cfg.te_use_ma200_cross:
            prev_pos = np.roll(sma_pos, 1)
            prev_pos[0] = np.nan
            ma200_cross = (sma_pos > 0.0) & (prev_pos <= 0.0)
        else:
            ma200_cross = sma_pos > 0.0

        te = (
            (adx >= float(cfg.te_adx_min))
            & (adx_slope >= float(cfg.te_adx_slope_min))
            & (ma200_cross | (sma_slope > 0.0))
        )

        # Extreme veto: in extreme volatility, force NO_TRADE
        if cfg.extreme_atr_percentile_max is not None:
            extreme = atr_pct >= float(cfg.extreme_atr_percentile_max)
            tc = tc & ~extreme
            te = te & ~extreme
            mean = mean & ~extreme

        # Priority: TC > TE > MEAN
        regime = np.full(len(df), "NO_TRADE", dtype=object)
        regime[mean] = "MEAN"
        regime[te] = "TE"
        regime[tc] = "TC"

    out = pd.DataFrame(index=df.index)
    out["regime"] = regime.astype(str)
    out["adx"] = adx
    out["adx_slope"] = adx_slope
    out["sma_200_position"] = sma_pos
    out["sma_200_slope"] = sma_slope
    out["sr_distance_normalized"] = sr_dist
    out["sqs"] = sqs
    out["atr_percentile"] = atr_pct
    if cfg.use_soft_scores:
        out["tc_score"] = tc_score
        out["te_score"] = te_score
        out["mean_score"] = mean_score
        out["regime_score"] = max_score
        out["regime_soft_profile"] = cfg.soft_profile_name
    return out


def compute_trade_quality(
    df: pd.DataFrame,
    *,
    cfg: Rule3ActionConfig = Rule3ActionConfig(),
    score_cfg: QualityScoreConfig = QualityScoreConfig(),
    preds_in_log1p: bool = True,
    calibration: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    missing = [
        c
        for c in [
            cfg.pred_dir_prob_col,
            cfg.pred_mfe_col,
            cfg.pred_mae_col,
            cfg.pred_ttm_col,
        ]
        if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required pred columns for quality: {missing}")

    p = pd.to_numeric(df[cfg.pred_dir_prob_col], errors="coerce").to_numpy(dtype=float)
    mfe_p = pd.to_numeric(df[cfg.pred_mfe_col], errors="coerce").to_numpy(dtype=float)
    mae_p = pd.to_numeric(df[cfg.pred_mae_col], errors="coerce").to_numpy(dtype=float)
    ttm_p = pd.to_numeric(df[cfg.pred_ttm_col], errors="coerce").to_numpy(dtype=float)
    if calibration:
        p, mfe_p, mae_p, ttm_p = _apply_router_calibration(
            p,
            mfe_p,
            mae_p,
            ttm_p,
            calibration=calibration,
            preds_in_log1p=preds_in_log1p,
        )

    mfe = _maybe_expm1(mfe_p, preds_in_log1p=preds_in_log1p)
    mae = _maybe_expm1(mae_p, preds_in_log1p=preds_in_log1p)
    ttm = _maybe_expm1(ttm_p, preds_in_log1p=preds_in_log1p)
    dconf = _dir_conf(np.clip(p, 0.0, 1.0))

    eff = np.where(
        np.isfinite(mfe) & np.isfinite(mae),
        mfe / (mae + float(cfg.eps)),
        0.0,
    )

    if score_cfg.use_dir_conf:
        quality = eff * dconf
    else:
        quality = eff

    out = pd.DataFrame(index=df.index)
    out["trade_quality"] = quality.astype(float)
    out["mfe_atr"] = mfe
    out["mae_atr"] = mae
    out["t_to_mfe"] = ttm
    out["eff"] = eff
    out["dir_conf"] = dconf
    return out


def compute_mode_3action_regime_quality(
    df: pd.DataFrame,
    *,
    rule_cfg: Rule3ActionConfig = Rule3ActionConfig(),
    regime_cfg: RegimeRuleConfig = RegimeRuleConfig(),
    score_cfg: QualityScoreConfig = QualityScoreConfig(),
    preds_in_log1p: bool = True,
    calibration: Optional[Dict[str, Any]] = None,
    out_col: str = "mode",
) -> pd.DataFrame:
    """
    Regime-qualified router (B/C):
      - Regime from rule features
      - Trade quality from model heads
      - Mode from (regime + quality threshold)
    """
    regime_df = compute_regime_rules(df, cfg=regime_cfg)
    quality_df = compute_trade_quality(
        df,
        cfg=rule_cfg,
        score_cfg=score_cfg,
        preds_in_log1p=preds_in_log1p,
        calibration=calibration,
    )

    regime = regime_df["regime"].astype(str).values
    quality = quality_df["trade_quality"].to_numpy(dtype=float)

    mode = np.full(len(df), "NO_TRADE", dtype=object)
    action = np.zeros(len(df), dtype=int)

    trend_mask = regime == "TC"
    te_mask = regime == "TE"
    mean_mask = regime == "MEAN"

    trend_ok = trend_mask & (quality >= float(score_cfg.quality_trend_min))
    te_ok = te_mask & (quality >= float(score_cfg.quality_te_min))
    mean_ok = mean_mask & (quality >= float(score_cfg.quality_mean_min))

    mode[mean_ok] = "MEAN"
    action[mean_ok] = 1
    mode[trend_ok | te_ok] = "TREND"
    action[trend_ok | te_ok] = 2

    out = pd.DataFrame(index=df.index)
    out[out_col] = mode.astype(str)
    out["mode_action"] = action.astype(int)
    out = out.join(quality_df).join(regime_df)
    return out


def compute_mode_3action_regime_only(
    df: pd.DataFrame,
    *,
    rule_cfg: Rule3ActionConfig = Rule3ActionConfig(),
    regime_cfg: RegimeRuleConfig = RegimeRuleConfig(),
    score_cfg: QualityScoreConfig = QualityScoreConfig(),
    preds_in_log1p: bool = True,
    calibration: Optional[Dict[str, Any]] = None,
    out_col: str = "mode",
) -> pd.DataFrame:
    """
    C-mode: Regime decides mode; NN provides trade_quality only.
    TE is mapped to TREND for execution compatibility.
    """
    regime_df = compute_regime_rules(df, cfg=regime_cfg)
    quality_df = compute_trade_quality(
        df,
        cfg=rule_cfg,
        score_cfg=score_cfg,
        preds_in_log1p=preds_in_log1p,
        calibration=calibration,
    )

    regime = regime_df["regime"].astype(str).values
    mode = np.full(len(df), "NO_TRADE", dtype=object)
    action = np.zeros(len(df), dtype=int)

    mean_mask = regime == "MEAN"
    trend_mask = (regime == "TC") | (regime == "TE")

    mode[mean_mask] = "MEAN"
    action[mean_mask] = 1
    mode[trend_mask] = "TREND"
    action[trend_mask] = 2

    out = pd.DataFrame(index=df.index)
    out[out_col] = mode.astype(str)
    out["mode_action"] = action.astype(int)
    out = out.join(quality_df).join(regime_df)
    return out


def compute_mode_3action(
    df: pd.DataFrame,
    *,
    cfg: Rule3ActionConfig = Rule3ActionConfig(),
    preds_in_log1p: bool = True,
    calibration: Optional[Dict[str, Any]] = None,
    out_col: str = "mode",
) -> pd.DataFrame:
    """
    Return a dataframe with columns:
      - mode (str): NO_TRADE/MEAN/TREND
      - mode_action (int): 0/1/2
      - optional derived diagnostics: mfe, mae, ttm, eff, dir_conf
    """
    missing = [
        c
        for c in [
            cfg.pred_dir_prob_col,
            cfg.pred_mfe_col,
            cfg.pred_mae_col,
            cfg.pred_ttm_col,
        ]
        if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required pred columns for rule router: {missing}")

    p = pd.to_numeric(df[cfg.pred_dir_prob_col], errors="coerce").to_numpy(dtype=float)
    mfe_p = pd.to_numeric(df[cfg.pred_mfe_col], errors="coerce").to_numpy(dtype=float)
    mae_p = pd.to_numeric(df[cfg.pred_mae_col], errors="coerce").to_numpy(dtype=float)
    ttm_p = pd.to_numeric(df[cfg.pred_ttm_col], errors="coerce").to_numpy(dtype=float)
    if calibration:
        p, mfe_p, mae_p, ttm_p = _apply_router_calibration(
            p,
            mfe_p,
            mae_p,
            ttm_p,
            calibration=calibration,
            preds_in_log1p=preds_in_log1p,
        )

    mfe = _maybe_expm1(mfe_p, preds_in_log1p=preds_in_log1p)
    mae = _maybe_expm1(mae_p, preds_in_log1p=preds_in_log1p)
    ttm = _maybe_expm1(ttm_p, preds_in_log1p=preds_in_log1p)

    dconf = _dir_conf(np.clip(p, 0.0, 1.0))
    # Avoid invalid divisions from NaN/Inf.
    eff = np.where(
        np.isfinite(mfe) & np.isfinite(mae),
        mfe / (mae + float(cfg.eps)),
        0.0,
    )

    # Default: NO_TRADE
    mode = np.full(len(df), "NO_TRADE", dtype=object)
    action = np.zeros(len(df), dtype=int)

    tradable = (
        np.isfinite(mfe)
        & np.isfinite(mae)
        & (mfe >= float(cfg.mfe_min))
        & (eff >= float(cfg.eff_min))
    )

    # Trend: strong directional confidence, plus confirmation
    if str(cfg.trend_confirm_mode).lower() == "or":
        trend = (
            tradable
            & (dconf >= float(cfg.dir_conf_trend_min))
            & ((mfe >= float(cfg.mfe_trend_min)) | (ttm >= float(cfg.ttm_trend_min)))
        )
    else:
        trend = (
            tradable
            & (dconf >= float(cfg.dir_conf_trend_min))
            & (mfe >= float(cfg.mfe_trend_min))
            & (ttm >= float(cfg.ttm_trend_min))
        )

    # Mean: better efficiency and quicker ttm (wins fast)
    mean = (
        tradable
        & ~trend
        & (eff >= float(cfg.eff_mean_min))
        & (ttm <= float(cfg.ttm_mean_max))
    )

    mode[mean] = "MEAN"
    action[mean] = 1
    mode[trend] = "TREND"
    action[trend] = 2

    out = pd.DataFrame(index=df.index)
    out[out_col] = mode.astype(str)
    out["mode_action"] = action.astype(int)
    out["mfe_atr"] = mfe
    out["mae_atr"] = mae
    out["t_to_mfe"] = ttm
    out["eff"] = eff
    out["dir_conf"] = dconf
    return out
