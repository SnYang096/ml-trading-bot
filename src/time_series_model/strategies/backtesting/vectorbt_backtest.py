"""
VectorBT 通用回测类

将现有的 run_vectorbt_backtest 封装为类，继承 BaseBacktest。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_backtest import BaseBacktest
from src.time_series_model.pipeline.training.label_utils import simulate_rr_exits


class VectorBTBacktest(BaseBacktest):
    """使用 vectorbt 的通用回测实现。"""

    @staticmethod
    def _quantile_threshold_series(
        series: pd.Series,
        *,
        q: float,
        mode: str = "expanding",
        window: int | None = None,
        min_periods: int = 50,
        train_threshold: float | None = None,
    ) -> pd.Series:
        """
        Build a (potentially time-varying) quantile threshold series without lookahead.

        - full: single global quantile over the full series (LOOKAHEAD; kept for compatibility)
        - expanding: expanding quantile using only history up to t-1 (no lookahead)
        - rolling: rolling quantile using only last `window` points up to t-1 (no lookahead)
        - train: fixed threshold provided by training calibration (no lookahead)
        """
        s = series.astype(float)
        qv = min(max(float(q), 0.0), 1.0)
        m = str(mode or "expanding").lower().strip()

        if m in ("train", "train_calibrated", "train_fixed"):
            if train_threshold is None:
                raise ValueError("quantile_mode=train requires train_threshold")
            return pd.Series(float(train_threshold), index=s.index)

        if m in ("full", "global", "leaky"):
            thr = (
                float(s.dropna().quantile(qv)) if s.dropna().shape[0] else float("nan")
            )
            return pd.Series(thr, index=s.index)

        mp = int(min_periods) if min_periods is not None else 50
        mp = max(mp, 1)

        if m in ("expanding", "causal", "expanding_causal"):
            return s.expanding(min_periods=mp).quantile(qv).shift(1)

        if m in ("rolling", "rolling_causal"):
            w = int(window) if window is not None else 200
            w = max(w, 1)
            return s.rolling(window=w, min_periods=mp).quantile(qv).shift(1)

        raise ValueError(f"Unknown quantile_mode: {mode}")

    @staticmethod
    def _multiclass_entries_from_proba(
        *,
        proba: np.ndarray,
        index: pd.Index,
        long_class: int,
        short_class: int,
        neutral_class: int,
        entry_mode: str = "argmax",
        entry_threshold: float | None = None,
        long_entry_threshold: float | None = None,
        short_entry_threshold: float | None = None,
        entry_quantile: float | None = None,
        long_entry_quantile: float | None = None,
        short_entry_quantile: float | None = None,
        quantile_mode: str = "expanding",
        quantile_window: int | None = None,
        quantile_min_periods: int = 50,
        long_train_threshold: float | None = None,
        short_train_threshold: float | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Build entries/exits for multiclass models.

        Modes:
        - argmax (default): enter when argmax == long/short, exit when argmax == neutral.
        - prob_threshold: enter when P(long/short) >= entry_threshold. Exits are set to False here,
          because RR exits (when enabled) should handle exits consistently.
        - prob_quantile: enter when P(long/short) is in the top quantile (e.g. q90) of its own distribution
          over the backtest window. Exits are set to False here (RR exits should handle exits).
        """
        if proba.ndim != 2:
            raise ValueError("proba must be 2D (n_samples, n_classes)")
        n_classes = int(proba.shape[1])
        for c in (long_class, short_class, neutral_class):
            if not (0 <= int(c) < n_classes):
                raise ValueError(
                    f"class index out of range: {c} (n_classes={n_classes})"
                )

        mode = str(entry_mode or "argmax").lower().strip()
        if mode == "argmax":
            class_preds = np.argmax(proba, axis=1)
            long_entries = pd.Series(class_preds == int(long_class), index=index)
            short_entries = pd.Series(class_preds == int(short_class), index=index)
            long_exits = pd.Series(class_preds == int(neutral_class), index=index)
            short_exits = pd.Series(class_preds == int(neutral_class), index=index)
            return long_entries, short_entries, long_exits, short_exits

        if mode in ("prob_threshold", "threshold", "proba_threshold"):
            thr_long = (
                entry_threshold
                if long_entry_threshold is None
                else long_entry_threshold
            )
            thr_short = (
                entry_threshold
                if short_entry_threshold is None
                else short_entry_threshold
            )
            thr_long = 0.5 if thr_long is None else float(thr_long)
            thr_short = 0.5 if thr_short is None else float(thr_short)
            thr_long = min(max(thr_long, 0.0), 1.0)
            thr_short = min(max(thr_short, 0.0), 1.0)
            long_p = pd.Series(proba[:, int(long_class)], index=index)
            short_p = pd.Series(proba[:, int(short_class)], index=index)
            long_entries = long_p >= thr_long
            short_entries = short_p >= thr_short
            both = long_entries & short_entries
            if bool(both.any()):
                pick_long = long_p >= short_p
                long_entries = long_entries & (~both | pick_long)
                short_entries = short_entries & (~both | ~pick_long)
            long_exits = pd.Series(False, index=index)
            short_exits = pd.Series(False, index=index)
            return long_entries, short_entries, long_exits, short_exits

        if mode in ("prob_quantile", "quantile", "proba_quantile"):
            long_p = pd.Series(proba[:, int(long_class)], index=index)
            short_p = pd.Series(proba[:, int(short_class)], index=index)
            q_long = (
                entry_quantile if long_entry_quantile is None else long_entry_quantile
            )
            q_short = (
                entry_quantile if short_entry_quantile is None else short_entry_quantile
            )
            if q_long is None or q_short is None:
                raise ValueError(
                    "prob_quantile requires entry_quantile (or both long_entry_quantile and short_entry_quantile)"
                )
            q_long = min(max(float(q_long), 0.0), 1.0)
            q_short = min(max(float(q_short), 0.0), 1.0)
            thr_long_s = VectorBTBacktest._quantile_threshold_series(
                long_p,
                q=q_long,
                mode=str(quantile_mode),
                window=quantile_window,
                min_periods=int(quantile_min_periods),
                train_threshold=long_train_threshold,
            )
            thr_short_s = VectorBTBacktest._quantile_threshold_series(
                short_p,
                q=q_short,
                mode=str(quantile_mode),
                window=quantile_window,
                min_periods=int(quantile_min_periods),
                train_threshold=short_train_threshold,
            )
            long_entries = (long_p >= thr_long_s).fillna(False)
            short_entries = (short_p >= thr_short_s).fillna(False)
            both = long_entries & short_entries
            if bool(both.any()):
                pick_long = long_p >= short_p
                long_entries = long_entries & (~both | pick_long)
                short_entries = short_entries & (~both | ~pick_long)
            long_exits = pd.Series(False, index=index)
            short_exits = pd.Series(False, index=index)
            return long_entries, short_entries, long_exits, short_exits

        raise ValueError(f"Unknown multiclass entry_mode: {entry_mode}")

    def _apply_max_holding_bars(
        self,
        *,
        index: pd.Index,
        long_entries: pd.Series,
        short_entries: pd.Series,
        long_exits: pd.Series,
        short_exits: pd.Series,
        max_holding_bars: int,
        allow_flip: bool = False,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Enforce a single-position state machine with:
        - max_holding_bars time exit
        - optional flip (exit current + enter opposite) when allow_flip=True
        """
        if max_holding_bars <= 0:
            return long_entries, short_entries, long_exits, short_exits

        le = long_entries.fillna(False).astype(bool).copy()
        se = short_entries.fillna(False).astype(bool).copy()
        lx = long_exits.fillna(False).astype(bool).copy()
        sx = short_exits.fillna(False).astype(bool).copy()

        in_long = False
        in_short = False
        entry_i_long = -1
        entry_i_short = -1

        for i in range(len(index)):
            # entries only when flat (unless allow_flip)
            if not in_long and not in_short:
                if bool(le.iloc[i]):
                    in_long = True
                    entry_i_long = i
                    lx.iloc[i] = False  # never exit on entry bar
                elif bool(se.iloc[i]):
                    in_short = True
                    entry_i_short = i
                    sx.iloc[i] = False

            # optional flip on opposite signal
            if allow_flip:
                if in_long and bool(se.iloc[i]) and i - entry_i_long >= 1:
                    lx.iloc[i] = True
                    in_long = False
                    entry_i_long = -1
                    # enter opposite on next bar to avoid same-bar exit/entry ambiguity
                    if i + 1 < len(index):
                        se.iloc[i + 1] = True
                if in_short and bool(le.iloc[i]) and i - entry_i_short >= 1:
                    sx.iloc[i] = True
                    in_short = False
                    entry_i_short = -1
                    if i + 1 < len(index):
                        le.iloc[i + 1] = True

            # time exit
            if in_long:
                held = i - entry_i_long
                if held >= 1 and (bool(lx.iloc[i]) or held >= max_holding_bars):
                    lx.iloc[i] = True
                    in_long = False
                    entry_i_long = -1
            if in_short:
                held = i - entry_i_short
                if held >= 1 and (bool(sx.iloc[i]) or held >= max_holding_bars):
                    sx.iloc[i] = True
                    in_short = False
                    entry_i_short = -1

        # no same-bar exit
        lx = lx & (~le)
        sx = sx & (~se)
        return le, se, lx, sx

    def _apply_sr_fuse(
        self,
        *,
        df: pd.DataFrame,
        price: pd.Series,
        long_entries: pd.Series,
        short_entries: pd.Series,
        params: Dict[str, Any],
    ) -> tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """
        Optional SR safety fuse (distance gate):
        - If enabled: block entries when price is too far from nearest SR.
        - dist_col is assumed to be a relative distance (pct of price) by default,
          aligned with dist_to_nearest_sr semantics in our feature pipeline.
        - Normalization: (abs(dist_pct) * price) / atr <= max_dist_atr
        """
        cfg = (
            (params.get("sr_fuse") or {})
            if isinstance(params.get("sr_fuse"), dict)
            else {}
        )
        enabled = bool(cfg.get("enabled", False))
        if not enabled:
            return long_entries, short_entries, None

        dist_col = str(cfg.get("dist_col", "dist_to_nearest_sr"))
        atr_col = str(cfg.get("atr_col", "atr"))
        max_dist_atr = float(cfg.get("max_dist_atr", 1.5))
        on_missing = str(cfg.get("on_missing", "skip")).lower()  # skip|block

        # If required columns are missing:
        # - on_missing=skip: ignore sr_fuse (do not block entries)
        # - on_missing=block: block entries (fail closed)
        if dist_col not in df.columns:
            if on_missing in {"skip", "ignore"}:
                print(
                    f"   ⚠️  sr_fuse enabled but dist_col='{dist_col}' missing -> skipping fuse"
                )
                return long_entries, short_entries, None
            print(
                f"   ⚠️  sr_fuse enabled but dist_col='{dist_col}' missing -> blocking entries"
            )
            return (
                long_entries & False,
                short_entries & False,
                pd.Series(False, index=df.index),
            )

        # ATR: must exist or be derived
        if atr_col in df.columns:
            atr = pd.to_numeric(df[atr_col], errors="coerce").astype(float)
        else:
            # Derive ATR quickly if possible; otherwise fallback to 1.0
            high_col = params.get("high_col", "high")
            low_col = params.get("low_col", "low")
            close_col = params.get("price_col", "close")
            if all(c in df.columns for c in (high_col, low_col, close_col)):
                high = pd.to_numeric(df[high_col], errors="coerce").astype(float)
                low = pd.to_numeric(df[low_col], errors="coerce").astype(float)
                close = pd.to_numeric(df[close_col], errors="coerce").astype(float)
                tr = pd.concat(
                    [
                        high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr_window = int((params.get("rr") or {}).get("atr_window", 14))
                atr = tr.rolling(window=atr_window, min_periods=1).mean()
            else:
                atr = pd.Series(1.0, index=df.index)

        dist_raw = pd.to_numeric(df[dist_col], errors="coerce").abs().astype(float)
        price_s = pd.to_numeric(price, errors="coerce").astype(float)

        # dist_to_nearest_sr in our feature pipeline is a relative distance (pct of price).
        # Avoid "auto-detect" based on full-window quantiles (would be a form of lookahead).
        dist_is_pct = bool(cfg.get("dist_is_pct", True))

        if dist_is_pct:
            abs_dist = dist_raw * price_s
        else:
            abs_dist = dist_raw

        norm_dist_atr = abs_dist / (atr + 1e-8)
        fuse_ok = (norm_dist_atr <= max_dist_atr).fillna(False).astype(bool)

        long_entries = long_entries & fuse_ok
        short_entries = short_entries & fuse_ok
        return long_entries, short_entries, fuse_ok

    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        task_type: str = "binary",
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        params = kwargs or {}
        enabled = params.get("enabled", True)
        if not enabled:
            return None

        try:
            import vectorbt as vbt  # type: ignore
        except ImportError:
            print("   ⚠️  vectorbt not installed. Skipping backtest.")
            return None

        price_col = params.get("price_col", "close")
        if price_col not in df.columns:
            print(f"   ⚠️  Price column '{price_col}' not found. Skipping backtest.")
            return None

        price = df[price_col].astype(float)
        fee = params.get("fee", 0.0004)
        slippage = params.get("slippage", 0.0)
        init_cash = params.get("initial_cash", 10000.0)

        index = df.index

        debug = bool(params.get("debug", False))
        use_signal_direction = bool(params.get("use_signal_direction", False))
        signal_col = params.get("signal_col", "signal")
        use_rr_exit = bool(params.get("use_rr_exit", False))

        # 确定策略方向：从配置或策略名称推断
        strategy_direction = params.get(
            "strategy_direction", None
        )  # long_only, short_only, both
        if strategy_direction is None:
            # 从策略名称推断（如果通过 params 传入）
            strategy_name = params.get("strategy_name", "").lower()
            if "_long" in strategy_name or strategy_name.endswith("_long"):
                strategy_direction = "long_only"
            elif "_short" in strategy_name or strategy_name.endswith("_short"):
                strategy_direction = "short_only"
            else:
                strategy_direction = "both"  # 默认双向

        if task_type == "multiclass" and predictions.ndim == 2:
            multi_cfg = params.get("multiclass", {})
            long_class = multi_cfg.get("long_class", 2)
            short_class = multi_cfg.get("short_class", 0)
            neutral_class = multi_cfg.get("neutral_class", 1)
            entry_mode = multi_cfg.get("entry_mode", "argmax")
            entry_threshold = multi_cfg.get("entry_threshold", None)
            long_entry_threshold = multi_cfg.get("long_entry_threshold", None)
            short_entry_threshold = multi_cfg.get("short_entry_threshold", None)
            entry_quantile = multi_cfg.get("entry_quantile", None)
            long_entry_quantile = multi_cfg.get("long_entry_quantile", None)
            short_entry_quantile = multi_cfg.get("short_entry_quantile", None)
            quantile_mode = multi_cfg.get("quantile_mode", "expanding")
            quantile_window = multi_cfg.get("quantile_window", None)
            quantile_min_periods = multi_cfg.get("quantile_min_periods", 50)
            long_train_threshold = multi_cfg.get("long_train_threshold", None)
            short_train_threshold = multi_cfg.get("short_train_threshold", None)
            long_entries, short_entries, long_exits, short_exits = (
                self._multiclass_entries_from_proba(
                    proba=predictions,
                    index=index,
                    long_class=int(long_class),
                    short_class=int(short_class),
                    neutral_class=int(neutral_class),
                    entry_mode=str(entry_mode),
                    entry_threshold=entry_threshold,
                    long_entry_threshold=long_entry_threshold,
                    short_entry_threshold=short_entry_threshold,
                    entry_quantile=entry_quantile,
                    long_entry_quantile=long_entry_quantile,
                    short_entry_quantile=short_entry_quantile,
                    quantile_mode=str(quantile_mode),
                    quantile_window=quantile_window,
                    quantile_min_periods=int(quantile_min_periods),
                    long_train_threshold=long_train_threshold,
                    short_train_threshold=short_train_threshold,
                )
            )

            # Structural fix: enforce time-exit so "no neutral predicted" won't hold forever.
            # Default to 50 bars if not specified (can be overridden in backtest.yaml).
            max_holding_bars = int(params.get("max_holding_bars", 50))
            allow_flip = bool(params.get("allow_flip", False))
            long_entries, short_entries, long_exits, short_exits = (
                self._apply_max_holding_bars(
                    index=index,
                    long_entries=long_entries,
                    short_entries=short_entries,
                    long_exits=long_exits,
                    short_exits=short_exits,
                    max_holding_bars=max_holding_bars,
                    allow_flip=allow_flip,
                )
            )
        elif str(task_type).lower() == "regression":
            # Regression: use top-quantile gating on predicted values.
            preds_series = pd.Series(predictions, index=index)

            top_quantile = float(params.get("top_quantile", 0.1))
            top_quantile = min(max(top_quantile, 0.0), 1.0)
            entry_mode = str(params.get("entry_mode", "level")).lower()  # level|cross
            quantile_mode = str(params.get("quantile_mode", "expanding"))
            quantile_window = params.get("quantile_window", None)
            quantile_min_periods = int(params.get("quantile_min_periods", 50))

            # Long entries: top N% predicted values.
            thr_s = self._quantile_threshold_series(
                preds_series,
                q=1.0 - top_quantile,
                mode=quantile_mode,
                window=quantile_window,
                min_periods=quantile_min_periods,
                train_threshold=params.get("train_entry_threshold", None),
            )
            entry_raw = (preds_series >= thr_s).fillna(False)
            if entry_mode == "cross":
                prev = entry_raw.shift(1).fillna(False)
                long_entries = entry_raw & (~prev)
            else:
                long_entries = entry_raw

            # Default: regression is direction-fixed by config; keep shorts disabled unless explicitly both.
            if strategy_direction == "both":
                # Optional: allow bottom-quantile shorts for bi-directional regression strategies.
                bottom_quantile = float(params.get("bottom_quantile", top_quantile))
                bottom_quantile = min(max(bottom_quantile, 0.0), 1.0)
                short_thr_s = self._quantile_threshold_series(
                    preds_series,
                    q=bottom_quantile,
                    mode=quantile_mode,
                    window=quantile_window,
                    min_periods=quantile_min_periods,
                    train_threshold=params.get("train_short_threshold", None),
                )
                short_raw = (preds_series <= short_thr_s).fillna(False)
                if entry_mode == "cross":
                    prev_s = short_raw.shift(1).fillna(False)
                    short_entries = short_raw & (~prev_s)
                else:
                    short_entries = short_raw
            else:
                short_entries = pd.Series(False, index=index)

            # Exits will be handled by RR exits when enabled; otherwise, no explicit exits by default.
            long_exits = pd.Series(False, index=index)
            short_exits = pd.Series(False, index=index)

            # Optional SR fuse gate (distance to SR)
            long_entries, short_entries, sr_fuse_ok = self._apply_sr_fuse(
                df=df,
                price=price,
                long_entries=long_entries,
                short_entries=short_entries,
                params=params,
            )

            if debug:
                debug_signals = pd.DataFrame(
                    {
                        "price": price,
                        "pred": preds_series,
                        "long_entry": long_entries,
                        "short_entry": short_entries,
                        "sr_fuse_ok": sr_fuse_ok if sr_fuse_ok is not None else True,
                    }
                )
        else:
            # Backward/UX-friendly aliases:
            # Many strategy yamls use `entry_threshold` for direction-fixed binary strategies.
            entry_threshold = params.get("entry_threshold", None)
            entry_quantile = params.get("entry_quantile", None)  # e.g. 0.9 means q90

            long_entry = params.get("long_entry_threshold", 0.6)
            long_exit = params.get("long_exit_threshold", 0.4)
            short_entry = params.get("short_entry_threshold", 0.4)
            short_exit = params.get("short_exit_threshold", 0.6)
            exit_mode = str(params.get("exit_mode", "none")).lower()  # none|threshold
            entry_mode = str(params.get("entry_mode", "level")).lower()  # level|cross

            preds_series = pd.Series(predictions, index=index)

            if use_signal_direction and signal_col in df.columns:
                # SR reversal 等策略：方向由 signal 决定，preds 只控制是否入场
                signal_series = df[signal_col].fillna(0).astype(float)

                base_long_entries = preds_series >= long_entry
                base_short_entries = preds_series <= short_entry

                # 根据策略方向过滤信号
                if strategy_direction == "long_only":
                    long_entries = (signal_series > 0) & base_long_entries
                    short_entries = pd.Series(False, index=index)  # 不做空
                elif strategy_direction == "short_only":
                    long_entries = pd.Series(False, index=index)  # 不做多
                    short_entries = (signal_series < 0) & base_short_entries
                else:  # both
                    long_entries = (signal_series > 0) & base_long_entries
                    short_entries = (signal_series < 0) & base_short_entries

                # 初始的概率退出，后续可能被 RR 覆盖
                long_exits = (
                    (preds_series <= long_exit)
                    if exit_mode == "threshold"
                    else pd.Series(False, index=index)
                )
                short_exits = (
                    (preds_series >= short_exit)
                    if exit_mode == "threshold"
                    else pd.Series(False, index=index)
                )
            else:
                # 仅根据预测得分构造多空信号
                if strategy_direction == "long_only":
                    if entry_quantile is not None:
                        q = float(entry_quantile)
                        q = min(max(q, 0.0), 1.0)
                        thr_s = self._quantile_threshold_series(
                            preds_series,
                            q=q,
                            mode=str(params.get("quantile_mode", "expanding")),
                            window=params.get("quantile_window", None),
                            min_periods=int(params.get("quantile_min_periods", 50)),
                            train_threshold=params.get("train_entry_threshold", None),
                        )
                    else:
                        if entry_threshold is not None:
                            long_entry = float(entry_threshold)
                        thr_s = pd.Series(float(long_entry), index=index)

                    entry_raw = (preds_series >= thr_s).fillna(False)
                    if entry_mode == "cross":
                        prev = entry_raw.shift(1).fillna(False)
                        long_entries = entry_raw & (~prev)
                    else:
                        long_entries = entry_raw

                    long_exits = (
                        (preds_series <= long_exit)
                        if exit_mode == "threshold"
                        else pd.Series(False, index=index)
                    )
                    short_entries = pd.Series(False, index=index)  # 不做空
                    short_exits = pd.Series(False, index=index)
                elif strategy_direction == "short_only":
                    # Direction-fixed short-only: pred is P(success of a SHORT trade), so entry is preds >= threshold.
                    if entry_quantile is not None:
                        q = float(entry_quantile)
                        q = min(max(q, 0.0), 1.0)
                        thr_s = self._quantile_threshold_series(
                            preds_series,
                            q=q,
                            mode=str(params.get("quantile_mode", "expanding")),
                            window=params.get("quantile_window", None),
                            min_periods=int(params.get("quantile_min_periods", 50)),
                            train_threshold=params.get("train_entry_threshold", None),
                        )
                    else:
                        if entry_threshold is not None:
                            short_entry = float(entry_threshold)
                        thr_s = pd.Series(float(short_entry), index=index)

                    long_entries = pd.Series(False, index=index)  # 不做多
                    long_exits = pd.Series(False, index=index)
                    entry_raw = (preds_series >= thr_s).fillna(False)
                    if entry_mode == "cross":
                        prev = entry_raw.shift(1).fillna(False)
                        short_entries = entry_raw & (~prev)
                    else:
                        short_entries = entry_raw
                    short_exits = (
                        (preds_series >= short_exit)
                        if exit_mode == "threshold"
                        else pd.Series(False, index=index)
                    )
                else:  # both
                    long_entries = preds_series >= long_entry
                    long_exits = preds_series <= long_exit
                    short_entries = preds_series <= short_entry
                    short_exits = preds_series >= short_exit

            if debug:
                debug_signals = pd.DataFrame(
                    {
                        "price": price,
                        "pred": preds_series,
                        "long_entry": long_entries,
                        "short_entry": short_entries,
                    }
                )

        # 动态仓位（可选，基于 ATR 控制每笔风险）
        long_size, short_size = self._compute_position_sizes(
            df=df,
            price=price,
            params=params,
            rr_params=params.get("rr", {}),
            init_cash=init_cash,
            index=index,
        )

        # RR 退出逻辑：使用与标签一致的 simulate_rr_exits
        if use_rr_exit:
            # 对于 long_only/short_only 策略，方向是固定的，不需要 use_signal_direction
            # 对于 both 策略，需要 use_signal_direction 来确定方向
            if strategy_direction == "both" and not use_signal_direction:
                raise ValueError(
                    "use_rr_exit=True with strategy_direction='both' requires use_signal_direction=True so direction is defined by signal"
                )

            rr_params = params.get("rr", {})
            rr_max_holding_bars = int(rr_params.get("max_holding_bars", 24))
            rr_stop_loss_r = float(rr_params.get("stop_loss_r", 1.0))
            rr_take_profit_r = float(rr_params.get("take_profit_r", 2.0))
            rr_atr_window = int(rr_params.get("atr_window", 14))
            rr_entry_offset = int(rr_params.get("entry_offset", 1))
            rr_entry_price_col = rr_params.get("entry_price_col", None)
            rr_use_breakeven_stop = bool(rr_params.get("use_breakeven_stop", False))
            rr_use_time_exit = bool(rr_params.get("use_time_exit", True))
            rr_use_trailing_stop = bool(rr_params.get("use_trailing_stop", False))
            rr_trailing_atr_mult = float(rr_params.get("trailing_atr_mult", 1.0))

            # If time-exit is disabled, set a very large horizon but rely on force_close_on_end.
            if not rr_use_time_exit:
                rr_max_holding_bars = max(rr_max_holding_bars, len(df) + 10)

            # 构造仅包含被模型选中的信号方向列：1=多，-1=空
            rr_signal = pd.Series(0.0, index=index)
            rr_signal[long_entries] = 1.0
            rr_signal[short_entries] = -1.0

            df_rr = df.copy()
            df_rr[signal_col] = rr_signal

            long_exits_rr, short_exits_rr = simulate_rr_exits(
                df_rr,
                signal_col=signal_col,
                price_col=price_col,
                atr_col=params.get("atr_col", "atr"),
                atr_window=rr_atr_window,
                max_holding_bars=rr_max_holding_bars,
                stop_loss_r=rr_stop_loss_r,
                take_profit_r=rr_take_profit_r,
                entry_price_col=rr_entry_price_col,
                entry_offset=rr_entry_offset,
                use_breakeven_stop=rr_use_breakeven_stop,
                use_time_exit=rr_use_time_exit,
                use_trailing_stop=rr_use_trailing_stop,
                trailing_atr_mult=rr_trailing_atr_mult,
            )

            long_exits = long_exits_rr.reindex(index).fillna(False)
            short_exits = short_exits_rr.reindex(index).fillna(False)

        # Diagnostics: entry/exit counts (always returned)
        diag = {
            "entries_exits": {
                "long_entries": int(long_entries.sum()),
                "short_entries": int(short_entries.sum()),
                "long_exits": int(long_exits.sum()),
                "short_exits": int(short_exits.sum()),
                "total_entries": int((long_entries | short_entries).sum()),
            }
        }

        # 频率：用于 vectorbt 计算年化指标
        freq = params.get("freq", None)
        if freq is None:
            if isinstance(index, pd.DatetimeIndex):
                inferred_freq = index.inferred_freq
                if inferred_freq:
                    freq = inferred_freq
                elif len(index) > 1:
                    time_diff = index[1] - index[0]
                    sec = time_diff.total_seconds()
                    if sec == 900:
                        freq = "15T"
                    elif sec == 3600:
                        freq = "1H"
                    elif sec == 14400:
                        freq = "4H"
                    elif sec == 86400:
                        freq = "1D"
            if freq is None:
                raise ValueError(
                    "❌ 'freq' must be configured in backtest params for vectorbt metrics."
                )

        try:
            # Some vectorbt versions don't support size_short; use size only
            portfolio = vbt.Portfolio.from_signals(
                price,
                entries=long_entries,
                exits=long_exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=init_cash,
                fees=fee,
                slippage=slippage,
                freq=freq,
                size=long_size,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️  Backtest failed: {exc}")
            return None

        # If there are no trades, return a stable payload (do NOT return None),
        # so downstream (feature-group-search) can still record "0 trades" explicitly.
        if portfolio.wrapper.index.size == 0 or portfolio.trades.count() == 0:
            print(
                "   ⚠️  Backtest produced no trades; metrics like Sharpe/WinRate/Drawdown are N/A."
            )
            return {
                "total_return_pct": 0.0,
                "sharpe": float("nan"),
                "max_drawdown_pct": float("nan"),
                "win_rate": float("nan"),
                "total_trades": 0,
                "diagnostics": diag,
            }

        stats = portfolio.stats()

        result: Dict[str, Any] = {
            "total_return_pct": float(stats.get("Total Return [%]", 0.0)),
            "sharpe": float(stats.get("Sharpe Ratio", 0.0)),
            "max_drawdown_pct": float(stats.get("Max Drawdown [%]", 0.0)),
            "win_rate": float(stats.get("Win Rate [%]", 0.0)),
            "total_trades": int(stats.get("Total Trades", 0)),
        }

        result["diagnostics"] = diag

        if debug:
            debug_payload: Dict[str, Any] = {}
            try:
                trades = portfolio.trades.records_readable
            except Exception:
                trades = None

            if trades is not None and not trades.empty:
                n_trades = int(len(trades))
                n_win = int((trades["PnL"] > 0).sum())
                win_rate_manual = 100.0 * n_win / n_trades if n_trades > 0 else 0.0
                trades_sample = (
                    trades.sort_values("Entry Timestamp")
                    .head(200)
                    .reset_index(drop=True)
                )
                debug_payload["trades"] = trades_sample.to_dict(orient="records")
                debug_payload["trades_meta"] = {
                    "n_trades": n_trades,
                    "n_win": n_win,
                    "win_rate_manual": win_rate_manual,
                }

            if "debug_signals" in locals():
                entry_mask = long_entries | short_entries
                entries_df = (
                    debug_signals[entry_mask]
                    .reset_index()
                    .rename(columns={"index": "timestamp"})
                )

                # Keep both head and tail samples to avoid "stops early" confusion.
                max_rows = int(params.get("debug_signals_max_rows", 200) or 200)
                max_rows = max(1, max_rows)
                mode = str(params.get("debug_signals_sample", "head_tail")).lower()

                if len(entries_df) <= max_rows:
                    debug_payload["signals"] = entries_df.to_dict(orient="records")
                else:
                    if mode in {"tail", "last"}:
                        debug_payload["signals_tail"] = entries_df.tail(
                            max_rows
                        ).to_dict(orient="records")
                    elif mode in {"head", "first"}:
                        debug_payload["signals_head"] = entries_df.head(
                            max_rows
                        ).to_dict(orient="records")
                    else:
                        head_n = max_rows // 2
                        tail_n = max_rows - head_n
                        debug_payload["signals_head"] = entries_df.head(head_n).to_dict(
                            orient="records"
                        )
                        debug_payload["signals_tail"] = entries_df.tail(tail_n).to_dict(
                            orient="records"
                        )

            try:
                returns = portfolio.returns()
                mean_ret = float(returns.mean())
                std_ret = float(returns.std())
                debug_payload["returns_stats"] = {"mean": mean_ret, "std": std_ret}
            except Exception:
                pass

            result["debug"] = debug_payload

        return result

    def _compute_position_sizes(
        self,
        df: pd.DataFrame,
        price: pd.Series,
        params: Dict[str, Any],
        rr_params: Dict[str, Any],
        init_cash: float,
        index: pd.Index,
    ) -> tuple[pd.Series, pd.Series]:
        """
        基于 ATR 的动态仓位（可选）。

        逻辑：
        - 若未配置 position_sizing 或 type=none，则返回全1仓位（等权）。
        - type="atr_risk": 以每笔风险占初始资金的比例控制 size。
          size = (init_cash * risk_pct) / (stop_loss_r * atr * price)
          并可通过 max_size_cap 限制最大 size。
        """
        sizing = params.get("position_sizing", {}) or {}
        sizing_type = sizing.get("type", "none")

        # 默认等权
        default_size = pd.Series(1.0, index=index)
        if sizing_type == "none":
            return default_size, default_size

        if sizing_type == "atr_risk":
            atr_col = sizing.get("atr_col", "atr")
            atr_window = int(sizing.get("atr_window", rr_params.get("atr_window", 14)))
            risk_pct = float(sizing.get("risk_pct", 0.01))  # 1% 风险
            max_size_cap = float(sizing.get("max_size_cap", 10.0))

            # 确保 ATR 存在
            if atr_col not in df.columns:
                high_col = params.get("high_col", "high")
                low_col = params.get("low_col", "low")
                close_col = params.get("price_col", "close")
                if all(col in df.columns for col in (high_col, low_col, close_col)):
                    high = df[high_col]
                    low = df[low_col]
                    close = df[close_col]
                    tr = pd.concat(
                        [
                            high - low,
                            (high - close.shift(1)).abs(),
                            (low - close.shift(1)).abs(),
                        ],
                        axis=1,
                    ).max(axis=1)
                    atr = tr.rolling(window=atr_window, min_periods=1).mean()
                else:
                    atr = pd.Series(1.0, index=index)
            else:
                atr = df[atr_col]

            stop_loss_r = float(rr_params.get("stop_loss_r", 1.0))
            # size 按价格和 ATR 反比，限制上限
            raw_size = (init_cash * risk_pct) / (stop_loss_r * atr * price)
            raw_size = raw_size.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            raw_size = raw_size.clip(upper=max_size_cap)
            return raw_size, raw_size

        # 其他类型未实现，回退等权
        return default_size, default_size
