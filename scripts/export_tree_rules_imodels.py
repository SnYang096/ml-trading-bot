#!/usr/bin/env python3
"""
Export interpretable rules for tree strategies using `imodels`.

Intent:
- You already have a "best" feature config per strategy (e.g., features_suggested_..._C.yaml).
- This script replays the exact data+feature+label pipeline (same as train_strategy_pipeline),
  then fits a compact rule model and exports pruned, human-readable rules.

Outputs:
- rules.md: ranked rules with coef/support (pruned)
- rules.json: raw rule table (for tooling)

Notes on pruning ("剪掉很长的不靠谱逻辑"):
- limit max rule string length
- limit max number of conjunctions ("and")
- require minimum support
- keep top-K by |coef|
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is importable when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_tools.data_utils import load_raw_data
from src.time_series_model.strategy_config import StrategyConfigLoader

# Reuse the exact same helpers as tree training (keeps labels/filters/feature-cols consistent)
from scripts.train_strategy_pipeline import (
    apply_filters,
    apply_post_label_filters,
    determine_feature_columns,
    import_callable,
    run_feature_pipeline,
)


def _try_import_imodels():
    try:
        from imodels import RuleFitClassifier, RuleFitRegressor  # type: ignore

        return RuleFitClassifier, RuleFitRegressor
    except Exception as e:
        raise RuntimeError(
            "imodels is not installed. Install it via `pip install imodels` "
            "(or `pip install -r requirements.txt`)."
        ) from e


def _infer_task(y: pd.Series) -> str:
    ys = y.dropna()
    if len(ys) == 0:
        return "unknown"
    uniq = sorted(pd.Series(ys).unique().tolist())
    if len(uniq) <= 10 and set(uniq).issubset({0, 1}):
        return "binary"
    if len(uniq) <= 10 and all(isinstance(x, (int, np.integer)) for x in uniq):
        return "multiclass"
    return "regression"


def _split_train_test(
    df: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        raise ValueError("Empty raw df")
    test_size = float(test_size)
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be in (0,1)")
    split_idx = int(len(df) * (1 - test_size))
    split_idx = max(1, min(split_idx, len(df) - 1))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _rule_conditions_count(rule_str: str) -> int:
    # RuleFit typically uses 'and' between conditions
    if not rule_str:
        return 0
    return int(rule_str.count(" and ") + 1)


def _prune_rules(
    df_rules: pd.DataFrame,
    *,
    max_rules: int,
    min_support: float,
    max_conditions: int,
    max_rule_len: int,
) -> pd.DataFrame:
    out = df_rules.copy()
    # Normalize expected cols
    if "rule" not in out.columns:
        # Some versions use 'rule'/'condition' naming
        for cand in ["rule", "rules", "condition", "conditions", "rules_"]:
            if cand in out.columns:
                out["rule"] = out[cand]
                break
    if "coef" not in out.columns:
        for cand in ["coef", "coefficient", "weight", "coef_"]:
            if cand in out.columns:
                out["coef"] = out[cand]
                break

    if "support" not in out.columns:
        for cand in ["support", "coverage", "freq"]:
            if cand in out.columns:
                out["support"] = out[cand]
                break
    if "support" not in out.columns:
        out["support"] = np.nan

    out["rule"] = out["rule"].astype(str)
    out["coef"] = pd.to_numeric(out["coef"], errors="coerce")
    out["support"] = pd.to_numeric(out["support"], errors="coerce")

    out = out.dropna(subset=["coef"])
    out = out[out["rule"].str.len() <= int(max_rule_len)]
    out = out[
        out["rule"].apply(lambda s: _rule_conditions_count(s) <= int(max_conditions))
    ]
    if not math.isnan(float(min_support)):
        out = out[(out["support"].isna()) | (out["support"] >= float(min_support))]

    out["abs_coef"] = out["coef"].abs()
    out = out.sort_values(["abs_coef", "support"], ascending=[False, False])
    out = out.head(int(max_rules)).drop(columns=["abs_coef"])
    return out.reset_index(drop=True)


def _write_rules_md(
    *,
    out_path: Path,
    strategy: str,
    features_yaml: str,
    task: str,
    n_train: int,
    n_test: int,
    rules_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines += [
        f"# Tree rules export (imodels): `{strategy}`",
        "",
        f"- features.yaml: `{features_yaml}`",
        f"- task: `{task}`",
        f"- train_rows: {n_train}, test_rows: {n_test}",
        "",
        "## Top rules (pruned)",
        "",
        "| rank | coef | support | rule |",
        "|---:|---:|---:|---|",
    ]
    for i, row in rules_df.iterrows():
        coef = row.get("coef")
        sup = row.get("support")
        rule = row.get("rule")
        try:
            coef_s = f"{float(coef):.6g}"
        except Exception:
            coef_s = str(coef)
        try:
            sup_s = "" if pd.isna(sup) else f"{float(sup):.4f}"
        except Exception:
            sup_s = str(sup)
        rule_s = str(rule).replace("\n", " ").strip()
        lines.append(f"| {i+1} | {coef_s} | {sup_s} | `{rule_s}` |")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy-config", required=True, help="config/strategies/<strategy> dir"
    )
    ap.add_argument(
        "--features-yaml", required=True, help="Features YAML to use (full or lite)"
    )
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--output-dir", default="results/rules_export")
    ap.add_argument("--max-rules", type=int, default=50)
    ap.add_argument("--min-support", type=float, default=0.01)
    ap.add_argument("--max-conditions", type=int, default=3)
    ap.add_argument("--max-rule-len", type=int, default=160)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    RuleFitClassifier, RuleFitRegressor = _try_import_imodels()

    cfg_dir = Path(args.strategy_config).resolve()
    if not cfg_dir.exists():
        raise FileNotFoundError(cfg_dir)

    feat_yaml_path = Path(args.features_yaml).resolve()
    if not feat_yaml_path.exists():
        raise FileNotFoundError(feat_yaml_path)

    loader = StrategyConfigLoader(cfg_dir)
    strategy_cfg = loader.load()

    # Load raw data (same as training)
    df_raw = load_raw_data(
        data_path="data/parquet_data",
        symbol=str(args.symbol),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        timeframe=str(args.timeframe),
    )
    df_train_raw, df_test_raw = _split_train_test(df_raw, float(args.test_size))

    # Build temp strategy dir so run_feature_pipeline uses the provided features.yaml
    out_dir = Path(args.output_dir).resolve() / f"{cfg_dir.name}__imodels_rules"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = out_dir / "tmp_strategy"
    if tmp_cfg.exists():
        import shutil

        shutil.rmtree(tmp_cfg)
    import shutil

    shutil.copytree(cfg_dir, tmp_cfg)
    (tmp_cfg / "features.yaml").write_text(
        feat_yaml_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    # Reload config with overridden features.yaml
    strategy_cfg = StrategyConfigLoader(tmp_cfg).load()

    # Feature pipeline
    from src.features.loader.strategy_feature_loader import StrategyFeatureLoader

    feature_loader = StrategyFeatureLoader()
    df_train_feat = run_feature_pipeline(
        df_train_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=True,
    )
    df_test_feat = run_feature_pipeline(
        df_test_raw,
        feature_loader=feature_loader,
        pipeline_cfg=strategy_cfg.features,
        fit=False,
    )

    feature_cols = determine_feature_columns(df_train_feat, strategy_cfg.features)

    # Labels + filters (same as training)
    label_func = import_callable(
        strategy_cfg.labels.generator.module, strategy_cfg.labels.generator.function
    )
    _tr = df_train_feat.copy()
    _te = df_test_feat.copy()
    df_train_feat[strategy_cfg.labels.target_column] = label_func(
        _tr, **(strategy_cfg.labels.generator.params or {})
    )
    df_test_feat[strategy_cfg.labels.target_column] = label_func(
        _te, **(strategy_cfg.labels.generator.params or {})
    )
    df_train_f = apply_filters(df_train_feat, strategy_cfg.labels.filters)
    df_test_f = apply_filters(df_test_feat, strategy_cfg.labels.filters)
    df_train_f = apply_post_label_filters(
        df_train_f, strategy_cfg.labels.post_label_filters, feature_cols
    )
    df_test_f = apply_post_label_filters(
        df_test_f, strategy_cfg.labels.post_label_filters, feature_cols
    )

    target_col = strategy_cfg.labels.target_column
    y_train = df_train_f[target_col]
    y_test = df_test_f[target_col]

    # Build X (numeric only; drop non-numeric)
    X_train = df_train_f[feature_cols].select_dtypes(include=[np.number]).copy()
    X_test = df_test_f[feature_cols].select_dtypes(include=[np.number]).copy()
    # Align columns
    common = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common]
    X_test = X_test[common]

    # Drop rows where y is nan
    mtr = y_train.notna()
    mte = y_test.notna()
    X_train = X_train.loc[mtr].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = y_train.loc[mtr]
    X_test = X_test.loc[mte].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = y_test.loc[mte]

    task = _infer_task(y_train)

    # Fit rule models
    models = {}
    if task == "binary":
        m = RuleFitClassifier(random_state=int(args.random_state))
        m.fit(
            X_train.values,
            y_train.astype(int).values,
            feature_names=list(X_train.columns),
        )
        models["binary"] = m
    elif task == "multiclass":
        # one-vs-rest: fit one rule model per class
        classes = sorted(pd.Series(y_train).dropna().astype(int).unique().tolist())
        for c in classes:
            y_bin = (y_train.astype(int) == int(c)).astype(int)
            m = RuleFitClassifier(random_state=int(args.random_state))
            m.fit(X_train.values, y_bin.values, feature_names=list(X_train.columns))
            models[f"class_{c}"] = m
    else:
        m = RuleFitRegressor(random_state=int(args.random_state))
        m.fit(
            X_train.values,
            y_train.astype(float).values,
            feature_names=list(X_train.columns),
        )
        models["regression"] = m

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        # Extract rules
        rules_df = None
        if hasattr(model, "get_rules"):
            rules_df = model.get_rules()
        elif hasattr(model, "rules_"):
            rules_df = getattr(model, "rules_")
        if rules_df is None:
            raise RuntimeError(f"Cannot extract rules from model: {type(model)}")

        if not isinstance(rules_df, pd.DataFrame):
            rules_df = pd.DataFrame(rules_df)

        pruned = _prune_rules(
            rules_df,
            max_rules=int(args.max_rules),
            min_support=float(args.min_support),
            max_conditions=int(args.max_conditions),
            max_rule_len=int(args.max_rule_len),
        )

        # Write artifacts
        md_path = out_dir / f"rules_{name}.md"
        json_path = out_dir / f"rules_{name}.json"
        _write_rules_md(
            out_path=md_path,
            strategy=str(cfg_dir.name),
            features_yaml=str(feat_yaml_path),
            task=task if name == "regression" else name,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            rules_df=pruned,
        )
        json_path.write_text(pruned.to_json(orient="records"), encoding="utf-8")

    print(f"✅ Exported rules to: {out_dir}")


if __name__ == "__main__":
    main()
