"""Offline locked-prefilter threshold suggestion from features_labeled.parquet.

DEPRECATED entry path: prefer ``mlbot research plateau --layer prefilter`` +
``mlbot research calibrate`` (library API retained for ``tune_locked_prefilter_thresholds``).

对齐「meta 阶段常用的踩坑率叙事」：在标注表上对 PASS 掩码最大化
baseline_bad_rate − pass_bad_rate（默认标签列 ``success_no_rr_extreme``）。

说明：
- 该列来自各策略 ``labels_gate``（如 ``labels_rr_extreme.yaml`` 的 ``target_column``），
  与 Prepare 写入 ``features_labeled.parquet`` 的列一致；不是 Gate 模块专有目标。
- 有模型时的 Prefilter 多方法择优（``scoring_method_fallbacks``：KS / mean_effect 等）
  走另一条脚本；本模块是 **无模型 / locked 阈值** 的离线代理，可与 ``label_col``
  对齐或改用你认为更贴近 archetype 的其他二元列（需谨慎可比性）。
"""

from __future__ import annotations

import copy
import operator as op_module
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from scripts.locked_prefilter_utils import (
    apply_locked_thresholds,
    detect_locked_template,
    infer_writeback_bindings_from_prefilter,
)


_OPS_MAP = {
    ">=": op_module.ge,
    ">": op_module.gt,
    "<=": op_module.le,
    "<": op_module.lt,
}


def _pass_single(df: pd.DataFrame, feat: str, op_str: str, val: Any) -> pd.Series:
    op_f = _OPS_MAP.get(str(op_str))
    if op_f is None or feat not in df.columns:
        return pd.Series(False, index=df.index)
    col = df[feat]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return op_f(col.astype(float), float(val))


def prefilter_rules_pass_mask(df: pd.DataFrame, rules: List[Any]) -> pd.Series:
    """与 train_strategy_pipeline archetype prefilter 语义一致：顶层 AND，any_of 为 OR。"""
    mask = pd.Series(True, index=df.index)
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if rule.get("any_of") and isinstance(rule["any_of"], list):
            or_acc = pd.Series(False, index=df.index)
            for sub in rule["any_of"]:
                if not isinstance(sub, dict):
                    continue
                sf = sub.get("feature")
                so = sub.get("operator")
                sv = sub.get("value")
                if sf and so is not None and sv is not None:
                    or_acc |= _pass_single(df, str(sf), str(so), sv)
            mask &= or_acc
            continue
        feat = rule.get("feature")
        o = rule.get("operator")
        v = rule.get("value")
        if feat and o is not None and v is not None:
            mask &= _pass_single(df, str(feat), str(o), v)
    return mask


def invert_value_transform(stored: float, transform: str) -> float:
    """YAML 规则数值反推到 bindings param（与 locked_prefilter_utils._apply_value_transform 对偶）。"""
    t = str(transform or "identity").strip().lower()
    if t in {"identity", "none"}:
        return float(stored)
    if t == "abs":
        return float(abs(stored))
    if t in {"negate", "neg"}:
        return float(-stored)
    if t in {"neg_abs", "negative_abs"}:
        return float(abs(stored))
    raise ValueError(f"unsupported value_transform inverse: {transform}")


def extract_seed_params_from_bindings(
    prefilter_raw: Dict[str, Any], bindings: List[Dict[str, Any]]
) -> Dict[str, float]:
    """从 locked 规则读出阈值；命中规则与 ``_apply_config_bindings`` 一致。"""
    out: Dict[str, float] = {}
    rules = prefilter_raw.get("rules") or []
    for r in rules:
        if not isinstance(r, dict) or not r.get("locked"):
            continue
        targets: List[Tuple[Dict[str, Any], bool]] = [(r, False)]
        subs = r.get("any_of")
        if isinstance(subs, list):
            for sub in subs:
                if isinstance(sub, dict):
                    targets.append((sub, True))
        for node, is_sub in targets:
            feat = node.get("feature")
            op = node.get("operator")
            if feat is None or op is None:
                continue
            sv = node.get("value")
            if sv is None:
                continue
            for b in bindings:
                if not isinstance(b, dict):
                    continue
                p = str(b.get("param", "")).strip()
                if not p or p in out:
                    continue
                bf, bo = b.get("feature"), b.get("operator")
                target = str(b.get("target", "any")).strip().lower()
                if bf and str(bf) != str(feat):
                    continue
                if bo and str(bo) != str(op):
                    continue
                if target == "rule" and is_sub:
                    continue
                if target in {"any_of", "sub"} and not is_sub:
                    continue
                out[p] = invert_value_transform(
                    float(sv), str(b.get("value_transform", "identity"))
                )
                break
    return out


def _suggest_parquet_bindings_coordinate(
    raw: Dict[str, Any],
    df: pd.DataFrame,
    *,
    bindings: List[Dict[str, Any]],
    label_col: str,
    baseline_bad: float,
    min_pass_rate: float,
    max_pass_rate: float,
    scan_points: int,
    n_rounds: int,
    strict_bindings: bool,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """显式 writeback_bindings：不依赖 FER/BPC/ME 内置模板名。"""
    seed = extract_seed_params_from_bindings(raw, bindings)
    req_list = [
        str(b.get("param", "")).strip()
        for b in bindings
        if isinstance(b, dict) and str(b.get("param", "")).strip()
    ]
    req_set = list(dict.fromkeys(req_list))
    missing = [p for p in req_set if p not in seed]
    if missing:
        raise ValueError(
            f"writeback_bindings 种子不完整，未命中 locked 规则: {missing}; "
            f"got keys={sorted(seed.keys())}"
        )

    cur = {p: float(seed[p]) for p in req_set}
    history: List[Dict[str, Any]] = []

    def _binding_for_param(param: str) -> Dict[str, Any]:
        for b in bindings:
            if isinstance(b, dict) and str(b.get("param", "")).strip() == param:
                return b
        raise ValueError(f"no binding for param={param}")

    for _ in range(max(1, n_rounds)):
        for key in req_set:
            b = _binding_for_param(key)
            feat = str(b.get("feature") or "")
            if feat and feat in df.columns:
                cand_vals = _quantile_candidates(df[feat], scan_points)
            else:
                cand_vals = [float(cur[key])]
            if not cand_vals:
                cand_vals = [float(cur[key])]

            xs: List[float] = []
            sc: List[float] = []
            for thr in cand_vals:
                trial = dict(cur)
                trial[key] = float(thr)
                try:
                    tuned = apply_locked_thresholds(
                        copy.deepcopy(raw),
                        params=trial,
                        bindings=bindings,
                        strict_bindings=strict_bindings,
                    )
                    m = prefilter_rules_pass_mask(df, tuned.get("rules") or [])
                    scv = _lift_score(
                        m, df, label_col, baseline_bad, min_pass_rate, max_pass_rate
                    )
                except Exception:
                    scv = -1e9
                xs.append(float(thr))
                sc.append(float(scv))

            if xs and sc:
                pick = _plateau_pick_mid(xs, sc)
                cur[key] = float(pick)
                history.append({"key": key, "pick": pick, "max_lift": max(sc)})

    meta = {
        "template": "bindings",
        "label_col": label_col,
        "baseline_bad_rate": baseline_bad,
        "coord_history": history,
        "n_rows": len(df),
        "binding_params": req_set,
    }
    return cur, meta


def _quantile_candidates(series: pd.Series, n: int) -> List[float]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(s) < 10:
        return sorted(float(s.median()) for _ in range(1)) if len(s) else []
    qs = np.linspace(0.05, 0.95, max(5, min(n, 50)))
    return sorted(float(x) for x in s.quantile(qs).unique())


def _plateau_pick_mid(xs: List[float], scores: List[float]) -> float:
    if not xs:
        raise ValueError("empty plateau scan")
    best = max(scores)
    span = max(scores) - min(scores)
    eps = max(1e-12, float(span) * 0.02) if len(set(scores)) > 1 else 0.0
    good_idx = [i for i, s in enumerate(scores) if s >= best - eps]
    mid_i = good_idx[len(good_idx) // 2]
    return float(xs[mid_i])


def _lift_score(
    mask: pd.Series,
    df: pd.DataFrame,
    label_col: str,
    baseline_bad: float,
    min_pass_rate: float,
    max_pass_rate: float,
) -> float:
    pr = float(mask.mean())
    if pr < min_pass_rate or pr > max_pass_rate:
        return -1e9 + pr
    if mask.sum() < 5:
        return -1e9
    sub = df.loc[mask, label_col]
    bad = float((sub == 0).mean()) if len(sub) else 1.0
    return float(baseline_bad - bad)


def suggest_locked_prefilter_params_parquet(
    *,
    prod_prefilter_path,
    labeled_parquet_path,
    template: str,
    tcfg: Dict[str, Any],
    prefilter_gates: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """返回 apply_locked_thresholds 可用的参数字典与诊断信息（仅 bindings 路径）。"""
    prod_prefilter_path = prod_prefilter_path.resolve()
    labeled_parquet_path = labeled_parquet_path.resolve()
    raw = yaml.safe_load(prod_prefilter_path.read_text(encoding="utf-8")) or {}

    label_col = str(
        tcfg.get("label_col")
        or prefilter_gates.get("label_col")
        or "success_no_rr_extreme"
    )
    scan_points = int(tcfg.get("plateau_scan_points", 25) or 25)
    n_rounds = int(tcfg.get("plateau_coord_rounds", 3) or 3)
    min_pass_rate = float(prefilter_gates.get("min_pass_rate", 0.01) or 0.01)
    max_pass_rate = float(tcfg.get("max_pass_rate", 0.99) or 0.99)

    df = pd.read_parquet(labeled_parquet_path)
    if label_col not in df.columns:
        for alt in ("success_no_rr_extreme", "y", "label"):
            if alt in df.columns:
                label_col = alt
                break
        else:
            raise ValueError(
                f"labeled parquet 缺少标签列 {label_col}；可用列示例: {list(df.columns)[:40]}"
            )

    baseline_bad = float((df[label_col] == 0).mean())

    bindings = [
        b for b in (tcfg.get("writeback_bindings") or []) if isinstance(b, dict)
    ]
    if not bindings:
        bindings = infer_writeback_bindings_from_prefilter(raw)
    if not bindings:
        raise ValueError(
            "locked parquet 调参：请在 locked_threshold_tuning.writeback_bindings 显式列出绑定，"
            "或确保 archetypes/prefilter.yaml 中存在可推导的 locked 阈值规则 "
            "（非 skip_parquet_tune）。"
        )

    strict_bindings = bool(tcfg.get("writeback_strict", True))
    out, meta = _suggest_parquet_bindings_coordinate(
        raw,
        df,
        bindings=bindings,
        label_col=label_col,
        baseline_bad=baseline_bad,
        min_pass_rate=min_pass_rate,
        max_pass_rate=max_pass_rate,
        scan_points=scan_points,
        n_rounds=n_rounds,
        strict_bindings=strict_bindings,
    )
    guess = detect_locked_template(raw)
    cli_tpl = str(template or "").strip().lower()
    meta["archetype_template_guess"] = (
        guess
        if guess and guess != "unknown"
        else (cli_tpl if cli_tpl not in {"", "auto"} else "unknown")
    )
    meta["effective_writeback_bindings"] = bindings
    return out, meta
