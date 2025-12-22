from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _is_finite_number(x: object) -> bool:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return False
    return bool(np.isfinite(v))


def _to_float_or_nan(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _to_float_or_none(x: object) -> Optional[float]:
    v = _to_float_or_nan(x)
    return None if np.isnan(v) else float(v)


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize objects for strict JSON:
    - Replace NaN/Infinity/-Infinity with None
    """
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _extract_scalar_metrics(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        if k == "debug":
            continue
        if _is_finite_number(v):
            out[k] = float(v)
    return out


def _all_metric_keys(results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    eval_keys: set[str] = set()
    bt_keys: set[str] = set()
    for item in results:
        base = item.get("base") or {}
        evaluation = base.get("evaluation") or {}
        backtest = base.get("backtest") or {}
        if isinstance(evaluation, dict):
            eval_keys.update([k for k in evaluation.keys()])
        if isinstance(backtest, dict):
            bt_keys.update([k for k in backtest.keys() if k != "debug"])
    return sorted(eval_keys), sorted(bt_keys)


def build_base_summary_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Backward-compatible summary: only the single split ("base") metrics.
    """
    rows: List[Dict[str, Any]] = []
    for item in results:
        base = item.get("base") or {}
        row: Dict[str, Any] = {
            "variant": item.get("variant"),
            "avg_cv_metric": base.get("avg_cv_metric", np.nan),
            "n_train": base.get("n_train", 0),
            "n_test": base.get("n_test", 0),
        }
        evaluation = base.get("evaluation") or {}
        if isinstance(evaluation, dict):
            for key, value in evaluation.items():
                row[f"eval_{key}"] = value
        backtest = base.get("backtest") or {}
        if isinstance(backtest, dict):
            for key, value in backtest.items():
                if key == "debug":
                    continue
                row[f"bt_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_rolling_windows_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Per-window rolling results (one row per variant per rolling window).
    """
    rows: List[Dict[str, Any]] = []
    for item in results:
        variant = item.get("variant")
        rolling = item.get("rolling") or {}
        windows = rolling.get("windows") or []
        if not isinstance(windows, list):
            continue
        for w in windows:
            if not isinstance(w, dict):
                continue
            row: Dict[str, Any] = {
                "variant": variant,
                "window_start": w.get("window_start"),
                "window_end": w.get("window_end"),
                "avg_cv_metric": w.get("avg_cv_metric", np.nan),
                "n_train": w.get("n_train", 0),
                "n_test": w.get("n_test", 0),
            }
            evaluation = w.get("evaluation") or {}
            if isinstance(evaluation, dict):
                for k, v in evaluation.items():
                    row[f"eval_{k}"] = v
            backtest = w.get("backtest") or {}
            if isinstance(backtest, dict):
                for k, v in backtest.items():
                    if k == "debug":
                        continue
                    row[f"bt_{k}"] = v
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normalize timestamps if possible
    for col in ["window_start", "window_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_rolling_monthly_df(rolling_windows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly aggregation of rolling windows, grouped by window_end month.
    This is useful as a "different months" stability view even when windows are bar-based.
    """
    if rolling_windows_df is None or rolling_windows_df.empty:
        return pd.DataFrame()
    df = rolling_windows_df.copy()
    if "window_end" not in df.columns:
        return pd.DataFrame()
    df["month"] = df["window_end"].dt.to_period("M").astype(str)
    metric_cols = [
        c
        for c in df.columns
        if c.startswith("eval_") or c.startswith("bt_") or c == "avg_cv_metric"
    ]
    # Only aggregate numeric-like cols
    agg_map = {c: "mean" for c in metric_cols}
    out = (
        df.groupby(["variant", "month"], as_index=False)
        .agg(agg_map)
        .sort_values(["variant", "month"])
    )
    return out


@dataclass(frozen=True)
class DecisionRubric:
    """
    Simple, opinionated default rubric for ablation decisions.
    Users can override these thresholds later via config if needed.
    """

    # Prefer BT Sharpe if available; else fall back to avg_cv_metric
    min_sharpe_delta: float = 0.10
    min_total_return_delta_pct: float = 2.0
    max_dd_increase_pct: float = 1.0  # allow at most +1pp worse DD for KEEP
    rolling_min_positive_ratio: float = 0.60


def _pick_primary_metrics(
    bt_keys: Iterable[str], eval_keys: Iterable[str]
) -> List[str]:
    bt_keys = list(bt_keys)
    eval_keys = list(eval_keys)
    preferred = []
    for k in ["sharpe", "total_return_pct", "max_drawdown_pct"]:
        if k in bt_keys:
            preferred.append(f"bt_{k}")
    # Add a couple of eval metrics if present (prioritize AUC and Rank IC for model validation)
    for k in ["auc", "rank_ic", "ic", "corr", "correlation", "spearman", "pearson"]:
        if k in eval_keys:
            preferred.append(f"eval_{k}")
    # Always include CV
    preferred.append("avg_cv_metric")
    # De-dup
    seen = set()
    out = []
    for k in preferred:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def build_delta_and_verdict_df(
    base_summary_df: pd.DataFrame,
    rolling_windows_df: pd.DataFrame,
    base_variant: str = "base",
    rubric: DecisionRubric = DecisionRubric(),
) -> pd.DataFrame:
    """
    Build a decision-focused summary:
    - deltas vs base_variant for key metrics
    - rolling stability signals (positive ratio) if available
    - verdict + short rationale
    """
    if base_summary_df is None or base_summary_df.empty:
        return pd.DataFrame()

    df = base_summary_df.copy()
    if "variant" not in df.columns:
        return pd.DataFrame()

    df = df.set_index("variant")
    if base_variant not in df.index:
        # Fall back to first row as baseline
        base_variant = str(df.index[0])

    base_row = df.loc[base_variant]
    metric_cols = [
        c
        for c in df.columns
        if c == "avg_cv_metric" or c.startswith("eval_") or c.startswith("bt_")
    ]

    # Deltas vs base
    delta_cols = {}
    for c in metric_cols:
        delta_cols[f"delta_{c}"] = df[c] - base_row[c]
    delta_df = pd.DataFrame(delta_cols, index=df.index)

    out = df.join(delta_df, how="left")
    out["baseline"] = base_variant

    # Rolling stability: percent windows where delta(primary_metric)>0
    stability: Dict[str, Any] = {}
    if rolling_windows_df is not None and not rolling_windows_df.empty:
        # Determine common window order (use baseline)
        w = rolling_windows_df.copy()
        if "window_end" in w.columns and w["window_end"].notna().any():
            w = w.sort_values(["window_end"])
        primary_key = None
        # Try sharpe first
        for candidate in ["bt_sharpe", "avg_cv_metric"]:
            if candidate in w.columns:
                primary_key = candidate
                break
        if primary_key:
            base_w = w[w["variant"] == base_variant][
                ["window_end", primary_key]
            ].rename(columns={primary_key: "base_val"})
            for variant in out.index:
                vw = w[w["variant"] == variant][["window_end", primary_key]].rename(
                    columns={primary_key: "var_val"}
                )
                merged = pd.merge(base_w, vw, on="window_end", how="inner")
                if merged.empty:
                    stability[variant] = {"rolling_pos_ratio": np.nan, "rolling_n": 0}
                    continue
                delta = merged["var_val"] - merged["base_val"]
                pos_ratio = float(np.mean(delta > 0.0))
                stability[variant] = {
                    "rolling_pos_ratio": pos_ratio,
                    "rolling_n": int(len(delta)),
                }
    if stability:
        s_df = pd.DataFrame.from_dict(stability, orient="index")
        out = out.join(s_df, how="left")
    else:
        out["rolling_pos_ratio"] = np.nan
        out["rolling_n"] = 0

    # Verdicts
    verdicts: List[str] = []
    rationales: List[str] = []

    def _get_val(row: pd.Series, key: str) -> float:
        v = row.get(key, np.nan)
        return _to_float_or_nan(v)

    for variant, row in out.iterrows():
        if variant == base_variant:
            verdicts.append("BASELINE")
            rationales.append("Baseline variant for deltas.")
            continue

        d_sharpe = _get_val(row, "delta_bt_sharpe")
        d_ret = _get_val(row, "delta_bt_total_return_pct")
        d_dd = _get_val(row, "delta_bt_max_drawdown_pct")
        pos_ratio = _get_val(row, "rolling_pos_ratio")

        reasons: List[str] = []
        keep_ok = True

        # Primary improvement
        improved = False
        if not np.isnan(d_sharpe) and d_sharpe >= rubric.min_sharpe_delta:
            improved = True
            reasons.append(f"Sharpe +{d_sharpe:.2f} ≥ {rubric.min_sharpe_delta:.2f}")
        if not improved and (
            not np.isnan(d_ret) and d_ret >= rubric.min_total_return_delta_pct
        ):
            improved = True
            reasons.append(
                f"Return +{d_ret:.2f}pp ≥ {rubric.min_total_return_delta_pct:.2f}pp"
            )

        if not improved:
            keep_ok = False
            reasons.append("No clear primary improvement (Sharpe/Return).")

        # Risk constraint (DD lower is better; delta>0 is worse)
        if not np.isnan(d_dd) and d_dd > rubric.max_dd_increase_pct:
            keep_ok = False
            reasons.append(
                f"DD worse by +{d_dd:.2f}pp > {rubric.max_dd_increase_pct:.2f}pp"
            )

        # Rolling stability (optional)
        if not np.isnan(pos_ratio) and int(row.get("rolling_n", 0)) > 0:
            if pos_ratio < rubric.rolling_min_positive_ratio:
                keep_ok = False
                reasons.append(
                    f"Rolling stability low ({pos_ratio:.0%} < {rubric.rolling_min_positive_ratio:.0%})."
                )
            else:
                reasons.append(
                    f"Rolling stability OK ({pos_ratio:.0%} ≥ {rubric.rolling_min_positive_ratio:.0%})."
                )
        else:
            reasons.append("Rolling stability: N/A (not run).")

        if keep_ok:
            verdict = "KEEP"
        else:
            # If improved but violates constraints => CONSIDER, else DROP
            if improved:
                verdict = "CONSIDER"
            else:
                verdict = "DROP"
        verdicts.append(verdict)
        rationales.append(" ".join(reasons))

    out["verdict"] = verdicts
    out["rationale"] = rationales
    out = out.reset_index().rename(columns={"index": "variant"})
    return out


def _fmt_metric_value(metric: str, v: float) -> str:
    if np.isnan(v):
        return "N/A"
    # backtest common
    if metric.endswith("total_return_pct"):
        return f"{v:.2f}"
    if metric.endswith("max_drawdown_pct"):
        return f"{v:.2f}%"
    if metric.endswith("sharpe"):
        return f"{v:.2f}"
    return f"{v:.4f}"


def _build_html_table(headers: List[str], rows: List[List[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{c}</td>" for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _build_debug_report_html(
    results: List[Dict[str, Any]], symbol: str
) -> Optional[str]:
    debug_sections: List[str] = []
    for item in results:
        variant = item.get("variant")
        base = item.get("base") or {}
        bt_data = base.get("backtest") or {}
        if not isinstance(bt_data, dict):
            continue
        debug_data = bt_data.get("debug")
        if not isinstance(debug_data, dict):
            continue

        summary = debug_data.get("summary", {}) or {}
        trades_meta = debug_data.get("trades_meta", {}) or {}
        returns_stats = debug_data.get("returns_stats", {}) or {}
        diagnostics = base.get("diagnostics") or {}

        section_parts: List[str] = []
        section_parts.append(f'<h2 id="variant-{variant}">Variant: {variant}</h2>')
        section_parts.append("<div class='info-box'><ul>")

        if summary:
            section_parts.append(
                f"<li><strong>Total Return:</strong> {summary.get('total_return_pct', 0.0):.2f}%</li>"
            )
            section_parts.append(
                f"<li><strong>Sharpe:</strong> {summary.get('sharpe', 0.0):.2f}</li>"
            )
            section_parts.append(
                f"<li><strong>Max DD:</strong> {summary.get('max_drawdown_pct', 0.0):.2f}%</li>"
            )
            section_parts.append(
                f"<li><strong>Win Rate:</strong> {summary.get('win_rate_pct', 0.0):.2f}%</li>"
            )
        if trades_meta:
            section_parts.append(
                f"<li><strong>Trades:</strong> {trades_meta.get('n_trades', 0)} "
                f"(wins={trades_meta.get('n_win', 0)}, "
                f"win_rate_manual={trades_meta.get('win_rate_manual', 0.0):.2f}%)</li>"
            )
        if returns_stats:
            section_parts.append(
                f"<li><strong>Returns mean/std:</strong> "
                f"{returns_stats.get('mean', 0.0):.3e} / "
                f"{returns_stats.get('std', 0.0):.3e}</li>"
            )

        section_parts.append("</ul></div>")

        # Helper to render table from list[dict]
        def build_table(records: List[Dict[str, Any]], title: str) -> str:
            if not records:
                return f"<h3>{title}</h3><p>No records.</p>"
            cols = list(records[0].keys())
            header = "".join(f"<th>{c}</th>" for c in cols)
            rows_html = []
            for row in records[:200]:
                cells = "".join(f"<td>{row.get(c, '')}</td>" for c in cols)
                rows_html.append(f"<tr>{cells}</tr>")
            return (
                f"<h3>{title}</h3>"
                "<div class='table-wrapper'><table>"
                f"<thead><tr>{header}</tr></thead>"
                f"<tbody>{''.join(rows_html)}</tbody>"
                "</table></div>"
            )

        # ------------------------------------------------------------------
        # Diagnostics: index mismatch per feature + NaN warnings (Top 20, >=20%)
        # ------------------------------------------------------------------
        try:
            # Index mismatch
            fdbg = (diagnostics.get("feature_debug_stats") or {}).get("train") or {}
            idx_m = (fdbg.get("index_mismatch") or {}) if isinstance(fdbg, dict) else {}
            if idx_m:
                rows = []
                for fname, mm in idx_m.items():
                    if not isinstance(mm, dict):
                        continue
                    rows.append(
                        {
                            "feature": fname,
                            "extra": int(mm.get("extra", 0) or 0),
                            "missing": int(mm.get("missing", 0) or 0),
                        }
                    )
                rows = sorted(
                    rows,
                    key=lambda r: (r["extra"] + r["missing"], r["extra"]),
                    reverse=True,
                )
                section_parts.append("<h3>Feature index mismatches (Top)</h3>")
                section_parts.append(
                    "<p>These features produced outputs whose index did not match the input bars. "
                    "The pipeline now reindexes outputs to the input index, but mismatches can still "
                    "lead to more NaNs and fewer usable rows.</p>"
                )
                section_parts.append(
                    build_table(rows[:20], "Index mismatch stats (train) — Top 20")
                )

            # NaN warnings (research-first): show columns with NaN ratio >= threshold
            nanr = diagnostics.get("nan_report") or {}
            for split in ["train", "test"]:
                s = nanr.get(split) or {}
                top = s.get("top") or []
                thr = s.get("warn_threshold", 0.2)
                n_rows = int(s.get("n_rows", 0) or 0)
                if isinstance(top, list) and top:
                    section_parts.append(
                        f"<h3>NaN warnings (>= {float(thr):.0%}) — {split}</h3>"
                    )
                    section_parts.append(
                        f"<p>Rows in split: <strong>{n_rows}</strong>. "
                        f"Columns below have NaN ratio ≥ {float(thr):.0%}.</p>"
                    )
                    section_parts.append(
                        build_table(top[:20], f"Top 20 NaN columns ({split})")
                    )

            kpol = (
                (nanr.get("key_feature_policy") or {}) if isinstance(nanr, dict) else {}
            )
            if kpol:
                section_parts.append("<h3>Key-feature non-null policy</h3>")
                section_parts.append(
                    "<div class='info-box'><ul>"
                    f"<li><strong>threshold:</strong> {float(kpol.get('threshold', 0.2)):.0%}</li>"
                    f"<li><strong>n_key_features:</strong> {int(kpol.get('n_key_features', 0) or 0)}</li>"
                    f"<li><strong>note:</strong> {str(kpol.get('note','')).strip() or 'N/A'}</li>"
                    "</ul></div>"
                )
        except Exception:
            pass

        signals = debug_data.get("signals") or []
        trades = debug_data.get("trades") or []
        if isinstance(signals, list):
            section_parts.append(build_table(signals, "Entry Signals (sample)"))
        if isinstance(trades, list):
            section_parts.append(build_table(trades, "Trades (sample)"))

        debug_sections.append("".join(section_parts))

    if not debug_sections:
        return None

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Strategy Feature Debug Details: {symbol}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background-color: #f5f5f5; }}
    .container {{ max-width: 95%; width: 100%; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
    h2 {{ color: #555; margin-top: 30px; border-left: 4px solid #2196F3; padding-left: 10px; }}
    .table-wrapper {{ width: 100%; overflow-x: auto; }}
    table {{ width: 100%; min-width: 900px; border-collapse: collapse; margin: 20px 0; font-size: 12px; }}
    th, td {{ white-space: nowrap; padding: 6px 8px; text-align: left; border-bottom: 1px solid #ddd; }}
    th {{ background-color: #2196F3; color: white; font-weight: 600; position: sticky; top: 0; z-index: 2; }}
    .info-box {{ background: #e3f2fd; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Strategy Feature Debug Details</h1>
    <p>This page shows sample signals and trades for each variant where <code>backtest.debug</code> is enabled.</p>
    {''.join(debug_sections)}
  </div>
</body>
</html>
"""


def generate_strategy_feature_compare_html(
    results: List[Dict[str, Any]],
    base_summary_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    rolling_windows_df: pd.DataFrame,
    rolling_monthly_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    test_size: float,
    train_range: Optional[Tuple[str, str]] = None,
    test_range: Optional[Tuple[str, str]] = None,
    test_warmup_bars: Optional[int] = None,
    base_variant: str = "base",
) -> str:
    eval_keys, bt_keys = _all_metric_keys(results)
    variants = [str(x) for x in base_summary_df.get("variant", []).tolist()]

    # Decide which metrics to show prominently
    prominent_metrics = _pick_primary_metrics(bt_keys, eval_keys)
    # Keep only those present in the base summary columns
    prominent_metrics = [m for m in prominent_metrics if m in base_summary_df.columns]
    prominent_canvases: List[Tuple[str, str]] = [
        (m, f"bar_{m}".replace(".", "_").replace(":", "_")) for m in prominent_metrics
    ]

    # Decision summary table (compact)
    dec_headers = ["Variant", "Verdict", "Rolling+", "Key deltas", "Rationale"]
    dec_rows: List[List[str]] = []
    if decision_df is not None and not decision_df.empty:
        for _, r in decision_df.iterrows():
            variant = str(r.get("variant", ""))
            verdict = str(r.get("verdict", ""))
            rp = r.get("rolling_pos_ratio", np.nan)
            rn = int(r.get("rolling_n", 0) or 0)
            rolling_txt = "N/A"
            if _is_finite_number(rp) and rn > 0:
                rolling_txt = f"{float(rp):.0%} (n={rn})"

            deltas = []
            for m in prominent_metrics:
                dcol = f"delta_{m}"
                if dcol in decision_df.columns:
                    dv = _to_float_or_nan(r.get(dcol, np.nan))
                    if not np.isnan(dv):
                        deltas.append(f"{m}:{_fmt_metric_value(dcol, dv)}")
            key_delta_txt = "<br>".join(deltas) if deltas else ""
            dec_rows.append(
                [
                    variant,
                    verdict,
                    rolling_txt,
                    key_delta_txt,
                    str(r.get("rationale", "")),
                ]
            )
    decision_table_html = (
        _build_html_table(dec_headers, dec_rows) if dec_rows else "<p>N/A</p>"
    )

    # Base summary table (full)
    summary_headers = list(base_summary_df.columns)
    summary_rows: List[List[str]] = []
    for _, r in base_summary_df.iterrows():
        row = []
        for c in summary_headers:
            v = r.get(c, "")
            if c == "variant":
                row.append(f"<strong>{v}</strong>")
            elif c.startswith("bt_") or c.startswith("eval_") or c == "avg_cv_metric":
                row.append(_fmt_metric_value(c, _to_float_or_nan(v)))
            else:
                row.append(str(v))
        summary_rows.append(row)
    summary_table_html = _build_html_table(summary_headers, summary_rows)

    # ------------------------------------------------------------------
    # Model Diagnosis (pred distribution & calibration) — per variant
    # ------------------------------------------------------------------
    def _fmt_num(x: Any, decimals: int = 4) -> str:
        try:
            v = float(x)
        except Exception:
            return "N/A"
        if not np.isfinite(v):
            return "N/A"
        return f"{v:.{decimals}f}"

    def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        return cur if cur is not None else default

    def _diagnosis_label(
        pred_mean: Any, pred_std: Any, prior: Any, auc: Any
    ) -> tuple[str, str, str]:
        """
        Return (status, explanation, action).

        High-signal heuristics for binary models:
        - AUC≈0.5 and std tiny => model invalid (no discrimination)
        - AUC good but std tiny => likely calibration/compression (e.g. sigmoid/averaging)
        - Otherwise use std thresholds for discrimination levels.
        """
        try:
            m = float(pred_mean)
            s = float(pred_std)
        except Exception:
            return (
                "UNKNOWN",
                "Missing prediction stats.",
                "Rerun to generate pred_report.",
            )
        if not (np.isfinite(m) and np.isfinite(s)):
            return (
                "UNKNOWN",
                "Non-finite prediction stats.",
                "Check for NaNs/infs in predictions.",
            )

        p = None
        try:
            p = float(prior)
            if not np.isfinite(p):
                p = None
        except Exception:
            p = None

        a = None
        try:
            a = float(auc)
            if not np.isfinite(a):
                a = None
        except Exception:
            a = None

        # AUC-aware: strong statements when std is tiny
        if a is not None and s < 0.02:
            if abs(a - 0.5) <= 0.02:
                return (
                    "INVALID",
                    "AUC≈0.5 and pred.std is tiny (no discrimination; behaves like noise/prior).",
                    "Revisit labels/features/model; avoid deployment.",
                )
            if a >= 0.60:
                return (
                    "LIKELY-CALIBRATION",
                    "AUC is OK but pred.std is tiny (probabilities are compressed).",
                    "Try Platt(sigmoid)/isotonic calibration; tune thresholds on calibrated preds.",
                )

        # Discrimination-only rules
        if s < 0.02 and p is not None and abs(m - p) < 0.02:
            return (
                "PRIOR-LIKE",
                "Predictions are tightly clustered and close to label prior (likely little signal).",
                "Check features/labels/model complexity; improve signal before deployment.",
            )
        if s > 0.20:
            return (
                "GOOD",
                "Predictions have wide spread (good discrimination).",
                "Proceed with calibration + threshold optimization + trade-level validation.",
            )
        if s > 0.10:
            return (
                "MODERATE",
                "Predictions have some spread (moderate discrimination).",
                "Calibrate and tune thresholds; validate robustness before deployment.",
            )
        return (
            "WEAK",
            "Predictions are compressed (weak discrimination).",
            "Consider calibration and/or feature/label improvements; avoid relying on a single threshold.",
        )

    diag_cards: List[str] = []
    for item in results:
        variant = str(item.get("variant", "unknown"))
        base = item.get("base") or {}
        diagnostics = base.get("diagnostics") or {}
        pr = diagnostics.get("pred_report") if isinstance(diagnostics, dict) else None
        fd = (
            diagnostics.get("factor_direction")
            if isinstance(diagnostics, dict)
            else None
        )

        if not isinstance(pr, dict):
            diag_cards.append(
                f"""
<h3>Variant: <code>{variant}</code></h3>
<div class="info-box">
  <p><strong>Model Diagnosis:</strong> N/A (no <code>diagnostics.pred_report</code> found — rerun evaluation to generate it).</p>
</div>
"""
            )
            continue

        pred_stats = pr.get("pred_stats") or {}
        label_stats = pr.get("label_stats") or {}
        q = (pred_stats.get("quantiles") or {}) if isinstance(pred_stats, dict) else {}
        entry_gating = pr.get("entry_gating") or {}
        bins = pr.get("calibration_bins") or []

        pred_mean = _safe_get(pred_stats, "mean")
        pred_std = _safe_get(pred_stats, "std")
        pred_min = _safe_get(pred_stats, "min")
        pred_max = _safe_get(pred_stats, "max")
        prior = _safe_get(label_stats, "pos_rate")
        brier = pr.get("brier", None)

        auc = _safe_get(base, "evaluation", "auc")
        status, expl, action = _diagnosis_label(pred_mean, pred_std, prior, auc)

        # Simple calibration summary (binary): compare first/last bins if present
        calib_txt = "N/A"
        if isinstance(bins, list) and len(bins) >= 2:
            try:
                first = bins[0]
                last = bins[-1]
                calib_txt = (
                    f"Low bin: pred≈{_fmt_num(first.get('pred_mean'))}, win≈{_fmt_num(first.get('label_pos_rate'))} | "
                    f"High bin: pred≈{_fmt_num(last.get('pred_mean'))}, win≈{_fmt_num(last.get('label_pos_rate'))}"
                )
            except Exception:
                calib_txt = "N/A"

        # Quantiles to display (10/25/50/75/90)
        q_display = []
        for k in ["10", "25", "50", "75", "90"]:
            q_display.append(f"q{k}={_fmt_num(q.get(k))}")
        q_txt = ", ".join(q_display)

        eg_txt = "N/A"
        if isinstance(entry_gating, dict) and entry_gating:
            eg_txt = (
                f"thr={_fmt_num(entry_gating.get('entry_threshold'))}, "
                f"mode={str(entry_gating.get('entry_mode','')).strip() or 'N/A'}, "
                f"entry_rate={_fmt_num(entry_gating.get('entry_rate'))}"
            )

        fd_txt = ""
        if isinstance(fd, dict) and bool(fd.get("enabled", False)):
            inv = fd.get("inverted") or []
            n_inv = len(inv) if isinstance(inv, list) else 0
            method = str(fd.get("method", "spearman"))
            thr = fd.get("min_abs_ic", None)
            fd_txt = (
                f"<li><strong>Negative-factor inversion:</strong> enabled (method={method}, min_abs_ic={_fmt_num(thr)}), "
                f"inverted={n_inv}</li>"
            )

        diag_cards.append(
            f"""
<h3>Variant: <code>{variant}</code></h3>
<div class="info-box">
  <ul>
    <li><strong>Conclusion:</strong> {status} — {expl}</li>
    <li><strong>Action:</strong> {action}</li>
    <li><strong>Pred stats:</strong> mean={_fmt_num(pred_mean)}, std={_fmt_num(pred_std)}, min={_fmt_num(pred_min)}, max={_fmt_num(pred_max)}; {q_txt}</li>
    <li><strong>Model validity:</strong> AUC={_fmt_num(auc)} | prior(pos_rate)={_fmt_num(prior)}{f" | <strong>Brier:</strong> {_fmt_num(brier)}" if brier is not None else ""}</li>
    <li><strong>Entry gating (from backtest params):</strong> {eg_txt}</li>
    {fd_txt}
    <li><strong>Calibration (bins):</strong> {calib_txt}</li>
  </ul>
</div>
"""
        )

    model_diagnosis_html = ""
    if diag_cards:
        model_diagnosis_html = f"""
<h2>🧪 Model Diagnosis (pred distribution & calibration)</h2>
<div class="info-box">
  <ul>
    <li><strong>Why this matters:</strong> If <code>pred.std</code> is tiny and <code>pred.mean</code> ≈ label prior, the model behaves like a prior-only baseline (little signal).</li>
    <li><strong>Rule of thumb:</strong> <code>pred.std &lt; 0.02</code> often indicates near-constant predictions; confirm with AUC + calibration bins.</li>
  </ul>
</div>
<div class="table-wrapper">
  <table>
    <thead><tr><th>Condition</th><th>Conclusion</th><th>Action</th></tr></thead>
    <tbody>
      <tr><td><code>AUC ≈ 0.5</code> and <code>pred.std &lt; 0.02</code></td><td>Model invalid (no discrimination)</td><td>Revisit labels/features/model; avoid deployment</td></tr>
      <tr><td><code>AUC &gt; 0.6</code> and <code>pred.std &lt; 0.02</code></td><td>Likely calibration/compression</td><td>Try Platt(sigmoid)/isotonic calibration; tune thresholds</td></tr>
      <tr><td><code>pred.std &lt; 0.02</code> and <code>pred.mean ≈ label_prior</code></td><td>Prior-like baseline</td><td>Check features/labels/model complexity</td></tr>
      <tr><td><code>pred.std &gt; 0.1</code></td><td>Moderate discrimination</td><td>Calibrate + threshold optimize + validate stability</td></tr>
      <tr><td><code>pred.std &gt; 0.2</code></td><td>Good discrimination</td><td>Consider deployment only after robust trade-level validation</td></tr>
    </tbody>
  </table>
</div>
{''.join(diag_cards)}
"""

    # Chart.js scripts: base bars + rolling deltas line chart (if available)
    chart_scripts: List[str] = []

    # Base bar chart for prominent metrics
    for metric, canvas_id in prominent_canvases:
        values = [_to_float_or_none(v) for v in base_summary_df[metric].tolist()]
        chart_scripts.append(
            f"""
const ctx_{canvas_id} = document.getElementById('{canvas_id}');
if (ctx_{canvas_id}) {{
  new Chart(ctx_{canvas_id}, {{
    type: 'bar',
    data: {{
      labels: {json.dumps(variants)},
      datasets: [{{
        label: '{metric}',
        data: {json.dumps(values)},
        backgroundColor: 'rgba(33, 150, 243, 0.6)',
        borderColor: 'rgb(33, 150, 243)',
        borderWidth: 2
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ title: {{ display: true, text: '{metric} (base split)' }} }},
      scales: {{ y: {{ grid: {{ color: 'rgba(0,0,0,0.05)' }} }} }}
    }}
  }});
}}
"""
        )

    rolling_section_html = ""
    if rolling_windows_df is not None and not rolling_windows_df.empty:
        # Choose x-axis labels by baseline window_end
        wdf = rolling_windows_df.copy()
        wdf = wdf.dropna(subset=["window_end"])
        if not wdf.empty:
            base_w = wdf[wdf["variant"] == base_variant].sort_values("window_end")
            x = base_w["window_end"].dt.strftime("%Y-%m-%d").tolist()

            # Rolling delta charts for a few metrics if present
            rolling_metrics = []
            for m in [
                "bt_sharpe",
                "bt_total_return_pct",
                "bt_max_drawdown_pct",
                "avg_cv_metric",
            ]:
                if m in wdf.columns:
                    rolling_metrics.append(m)

            rolling_canvases = []
            for m in rolling_metrics:
                cid = f"roll_delta_{m}".replace(".", "_")
                rolling_canvases.append(
                    f"<h3>Rolling Δ vs {base_variant}: {m}</h3><div class='chart-container'><canvas id='{cid}'></canvas></div>"
                )

                # Build series per variant: (variant_value - base_value) aligned by window_end
                series = {}
                base_map = {
                    str(d): _to_float_or_none(v)
                    for d, v in zip(
                        base_w["window_end"].dt.strftime("%Y-%m-%d").tolist(),
                        base_w[m].tolist(),
                    )
                }
                for vname in sorted(wdf["variant"].unique().tolist()):
                    vw = wdf[wdf["variant"] == vname].sort_values("window_end")
                    vmap = {
                        str(d): _to_float_or_none(v)
                        for d, v in zip(
                            vw["window_end"].dt.strftime("%Y-%m-%d").tolist(),
                            vw[m].tolist(),
                        )
                    }
                    deltas = []
                    for d in x:
                        bv = base_map.get(d, None)
                        vv = vmap.get(d, None)
                        if bv is None or vv is None:
                            deltas.append(None)
                        else:
                            deltas.append(vv - bv)
                    series[vname] = deltas

                datasets_js = []
                palette = [
                    ("rgba(76, 175, 80, 0.25)", "rgb(76, 175, 80)"),
                    ("rgba(255, 152, 0, 0.25)", "rgb(255, 152, 0)"),
                    ("rgba(156, 39, 176, 0.25)", "rgb(156, 39, 176)"),
                    ("rgba(244, 67, 54, 0.25)", "rgb(244, 67, 54)"),
                    ("rgba(33, 150, 243, 0.25)", "rgb(33, 150, 243)"),
                ]
                for i, (vname, data) in enumerate(series.items()):
                    bg, border = palette[i % len(palette)]
                    datasets_js.append(
                        {
                            "label": vname,
                            "data": data,
                            "fill": False,
                            "borderColor": border,
                            "backgroundColor": bg,
                            "borderWidth": 2,
                            "tension": 0.2,
                            "spanGaps": True,
                        }
                    )

                chart_scripts.append(
                    f"""
const ctx_{cid} = document.getElementById('{cid}');
if (ctx_{cid}) {{
  new Chart(ctx_{cid}, {{
    type: 'line',
    data: {{ labels: {json.dumps(x)}, datasets: {json.dumps(datasets_js)} }},
    options: {{
      responsive: true,
      plugins: {{ title: {{ display: true, text: 'Rolling Δ vs {base_variant}: {m}' }} }},
      interaction: {{ mode: 'index', intersect: false }},
      scales: {{ y: {{ grid: {{ color: 'rgba(0,0,0,0.05)' }} }} }}
    }}
  }});
}}
"""
                )

            # Monthly chart (optional)
            monthly_html = ""
            if rolling_monthly_df is not None and not rolling_monthly_df.empty:
                # Render one monthly delta chart for sharpe if present
                if "bt_sharpe" in rolling_monthly_df.columns:
                    months = sorted(rolling_monthly_df["month"].unique().tolist())
                    base_m = rolling_monthly_df[
                        rolling_monthly_df["variant"] == base_variant
                    ]
                    base_map = {
                        m: _to_float_or_none(v)
                        for m, v in zip(base_m["month"], base_m["bt_sharpe"])
                    }
                    datasets_js = []
                    palette = [
                        ("rgba(76, 175, 80, 0.25)", "rgb(76, 175, 80)"),
                        ("rgba(255, 152, 0, 0.25)", "rgb(255, 152, 0)"),
                        ("rgba(156, 39, 176, 0.25)", "rgb(156, 39, 176)"),
                        ("rgba(244, 67, 54, 0.25)", "rgb(244, 67, 54)"),
                        ("rgba(33, 150, 243, 0.25)", "rgb(33, 150, 243)"),
                    ]
                    for i, vname in enumerate(
                        sorted(rolling_monthly_df["variant"].unique().tolist())
                    ):
                        vm = rolling_monthly_df[rolling_monthly_df["variant"] == vname]
                        vmap = {
                            m: _to_float_or_none(v)
                            for m, v in zip(vm["month"], vm["bt_sharpe"])
                        }
                        deltas = []
                        for mo in months:
                            bv = base_map.get(mo, None)
                            vv = vmap.get(mo, None)
                            deltas.append(None if bv is None or vv is None else vv - bv)
                        bg, border = palette[i % len(palette)]
                        datasets_js.append(
                            {
                                "label": vname,
                                "data": deltas,
                                "fill": False,
                                "borderColor": border,
                                "backgroundColor": bg,
                                "borderWidth": 2,
                                "tension": 0.2,
                                "spanGaps": True,
                            }
                        )
                    cid = "month_delta_bt_sharpe"
                    monthly_html = f"<h3>Monthly Δ vs {base_variant}: bt_sharpe</h3><div class='chart-container'><canvas id='{cid}'></canvas></div>"
                    chart_scripts.append(
                        f"""
const ctx_{cid} = document.getElementById('{cid}');
if (ctx_{cid}) {{
  new Chart(ctx_{cid}, {{
    type: 'line',
    data: {{ labels: {json.dumps(months)}, datasets: {json.dumps(datasets_js)} }},
    options: {{
      responsive: true,
      plugins: {{ title: {{ display: true, text: 'Monthly Δ vs {base_variant}: bt_sharpe' }} }},
      interaction: {{ mode: 'index', intersect: false }},
      scales: {{ y: {{ grid: {{ color: 'rgba(0,0,0,0.05)' }} }} }}
    }}
  }});
}}
"""
                    )

            rolling_section_html = f"""
<h2>🧭 Rolling Stability (per-window & per-month)</h2>
<div class="info-box">
  <ul>
    <li><strong>How to read:</strong> Look at <code>Δ vs {base_variant}</code>. If most windows/months are positive and tail is not ugly, the feature change is likely robust.</li>
    <li><strong>Rule of thumb:</strong> Prefer variants with rolling positive ratio ≥ 60% on primary metric, and without big drawdown deterioration.</li>
  </ul>
</div>
{''.join(rolling_canvases)}
{monthly_html}
"""

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Strategy Feature Compare: {symbol}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background-color: #f5f5f5; }}
    .container {{ max-width: 95%; width: 100%; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
    h2 {{ color: #555; margin-top: 30px; border-left: 4px solid #2196F3; padding-left: 10px; }}
    h3 {{ color: #444; margin-top: 20px; }}
    .table-wrapper {{ width: 100%; overflow-x: auto; }}
    table {{ width: 100%; min-width: 900px; border-collapse: collapse; margin: 20px 0; font-size: 13px; }}
    th, td {{ white-space: nowrap; padding: 8px 10px; text-align: left; }}
    th {{ background-color: #2196F3; color: white; padding: 12px; font-weight: 600; position: sticky; top: 0; z-index: 2; }}
    td {{ padding: 10px; border-bottom: 1px solid #ddd; vertical-align: top; }}
    tr:hover {{ background-color: #f5f5f5; }}
    .info-box {{ background: #e3f2fd; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0; border-radius: 4px; }}
    .chart-container {{ margin: 20px 0; padding: 16px; background: #f9f9f9; border-radius: 8px; height: 380px; }}
    code {{ background: #f1f1f1; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🆚 Strategy Feature Comparison Report</h1>
    <div class="info-box">
      <strong>Configuration:</strong><br>
      Symbol: {symbol} | Timeframe: {timeframe} | Test Size: {test_size:.1%}<br>
      Train Range: {train_range[0] if train_range else 'N/A'} → {train_range[1] if train_range else 'N/A'}<br>
      Test Range: {test_range[0] if test_range else 'N/A'} → {test_range[1] if test_range else 'N/A'}
      {f"| Warmup: {int(test_warmup_bars)} bars" if test_warmup_bars is not None else ""}<br>
      Variants: {len(variants)} | Baseline: <code>{base_variant}</code><br>
      Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
      <br>
      Debug Details: <a href="strategy_feature_compare_debug.html" target="_blank">Open detailed signals & trades</a> (only if enabled)
    </div>

    <h2>✅ Decision Summary (quick judgement)</h2>
    <div class="info-box">
      <ul>
        <li><strong>Goal:</strong> pick the smallest/simplest feature set that improves primary performance without worsening tail risk.</li>
        <li><strong>Primary metrics:</strong> Prefer Sharpe/Return. Constraints: drawdown should not deteriorate materially.</li>
        <li><strong>Rolling:</strong> A variant that only wins in a few windows is usually not robust.</li>
      </ul>
    </div>
    <div class="table-wrapper">{decision_table_html}</div>

    <h2>📊 Base Split Summary (metrics)</h2>
    <div class="table-wrapper">{summary_table_html}</div>

    {model_diagnosis_html}

    <h2>📈 Base Split Charts</h2>
    <div class="info-box">
      <strong>Tip:</strong> These are single-split indicators. Always confirm with rolling stability if you care about robustness.
    </div>
    {''.join([f"<h3>{m}</h3><div class='chart-container'><canvas id='{cid}'></canvas></div>" for (m, cid) in prominent_canvases])}

    {rolling_section_html}

    <h2>📝 Output Files</h2>
    <div class="info-box">
      <ul>
        <li><code>strategy_feature_compare_summary.csv</code>: backward-compatible base metrics</li>
        <li><code>strategy_feature_compare_summary_plus.csv</code>: deltas, rolling stability, verdict</li>
        <li><code>strategy_feature_compare_rolling_windows.csv</code>: per-window metrics (if rolling)</li>
        <li><code>strategy_feature_compare_rolling_monthly.csv</code>: month aggregation (if rolling)</li>
        <li><code>strategy_feature_compare_summary.json</code>: full raw results</li>
      </ul>
    </div>
  </div>

  <script>
  {''.join(chart_scripts)}
  </script>
</body>
</html>
"""


def write_strategy_feature_compare_reports(
    comparison_results: List[Dict[str, Any]],
    symbol: str,
    timeframe: str,
    test_size: float,
    start_date: Optional[str],
    end_date: Optional[str],
    train_range: Optional[Tuple[str, str]],
    test_range: Optional[Tuple[str, str]],
    test_warmup_bars: Optional[int],
    output_dir: Path,
    base_variant: str = "base",
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base summary (backward compatible)
    base_summary_df = build_base_summary_df(comparison_results)
    base_summary_df_clean = base_summary_df.replace([np.inf, -np.inf], np.nan)
    summary_csv = output_dir / "strategy_feature_compare_summary.csv"
    base_summary_df_clean.to_csv(summary_csv, index=False)

    # Rolling frames
    rolling_windows_df = build_rolling_windows_df(comparison_results)
    rolling_monthly_df = build_rolling_monthly_df(rolling_windows_df)

    rolling_windows_csv = output_dir / "strategy_feature_compare_rolling_windows.csv"
    rolling_monthly_csv = output_dir / "strategy_feature_compare_rolling_monthly.csv"
    if rolling_windows_df is not None and not rolling_windows_df.empty:
        rolling_windows_df.to_csv(rolling_windows_csv, index=False)
    if rolling_monthly_df is not None and not rolling_monthly_df.empty:
        rolling_monthly_df.to_csv(rolling_monthly_csv, index=False)

    # Decision summary + deltas
    decision_df = build_delta_and_verdict_df(
        base_summary_df_clean, rolling_windows_df, base_variant=base_variant
    )
    summary_plus_csv = output_dir / "strategy_feature_compare_summary_plus.csv"
    if decision_df is not None and not decision_df.empty:
        decision_df.replace([np.inf, -np.inf], np.nan).to_csv(
            summary_plus_csv, index=False
        )

    # Detailed JSON
    detailed_json = output_dir / "strategy_feature_compare_summary.json"
    with open(detailed_json, "w", encoding="utf-8") as fh:
        safe_results = _sanitize_for_json(comparison_results)
        json.dump(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "test_size": test_size,
                "start_date": start_date,
                "end_date": end_date,
                "train_range": list(train_range) if train_range else None,
                "test_range": list(test_range) if test_range else None,
                "test_warmup_bars": test_warmup_bars,
                "results": safe_results,
            },
            fh,
            indent=2,
            default=str,
            allow_nan=False,
        )

    # Debug report (optional)
    debug_html = _build_debug_report_html(comparison_results, symbol=symbol)
    debug_path = output_dir / "strategy_feature_compare_debug.html"
    if debug_html:
        with open(debug_path, "w", encoding="utf-8") as fh:
            fh.write(debug_html)

    # Main HTML report
    html_content = generate_strategy_feature_compare_html(
        results=comparison_results,
        base_summary_df=base_summary_df_clean,
        decision_df=decision_df,
        rolling_windows_df=rolling_windows_df,
        rolling_monthly_df=rolling_monthly_df,
        symbol=symbol,
        timeframe=timeframe,
        test_size=test_size,
        train_range=train_range,
        test_range=test_range,
        test_warmup_bars=test_warmup_bars,
        base_variant=base_variant,
    )
    html_path = output_dir / "strategy_feature_compare_report.html"
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    return {
        "summary_csv": summary_csv,
        "summary_plus_csv": summary_plus_csv,
        "rolling_windows_csv": rolling_windows_csv,
        "rolling_monthly_csv": rolling_monthly_csv,
        "detailed_json": detailed_json,
        "html_report": html_path,
        "debug_html": debug_path,
    }
