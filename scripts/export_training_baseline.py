#!/usr/bin/env python3
"""
导出训练基线 JSON — 汇总 L2-L8 各层 KPI + 特征分布快照

每次训练完成后，从各层产出的 JSON/YAML/Parquet 中提取关键 KPI，
生成标准化 training_baseline.json，供 local_monitor_weekly / monthly 对比。

用法:
    # 从训练产出目录导出 (最常见)
    python scripts/export_training_baseline.py \
        --result-dir results/train_final_xxx/me \
        --strategy me

    # 指定不同的 gate/evidence 目录
    python scripts/export_training_baseline.py \
        --result-dir results/train_final_xxx/me \
        --gate-dir results/train_final_xxx_gate/me \
        --strategy me

    # 传入 backtest 指标 (pipeline 调用时用)
    python scripts/export_training_baseline.py \
        --result-dir results/train_final_xxx/me \
        --strategy me \
        --backtest-json '{"total_trades":500,"sharpe_per_trade":0.35}'
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ====================================================================
# L2: Prefilter KPI — 从 prefilter.yaml last_evaluation 提取
# ====================================================================


def extract_l2_prefilter(
    strategy: str, config_root: str = "config/strategies"
) -> Dict[str, Any]:
    """从 prefilter.yaml 的 last_evaluation 段提取 KPI."""
    path = PROJECT_ROOT / config_root / strategy / "prefilter.yaml"
    if not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ev = data.get("last_evaluation", {})
    if not ev:
        return {}

    # 提取核心 KPI
    positive = ev.get("positive_signals", [])
    top_diff = 0.0
    temporal_cvs: List[float] = []
    for sig in positive if isinstance(positive, list) else []:
        diff = abs(sig.get("bad_rate_diff", 0))
        if diff > top_diff:
            top_diff = diff
        cv = sig.get("temporal_cv")
        if cv is not None:
            temporal_cvs.append(float(cv))

    kpi: Dict[str, Any] = {
        "n_rows": ev.get("n_rows", 0),
        "baseline_bad_rate": ev.get("baseline_bad_rate", 0),
        "top_feature_bad_rate_diff": round(top_diff, 4),
        "n_positive_signals": len(positive) if isinstance(positive, list) else 0,
    }
    if temporal_cvs:
        kpi["mean_temporal_cv"] = round(float(np.mean(temporal_cvs)), 3)
    return kpi


# ====================================================================
# L3: Direction KPI — 从 predictions.parquet 计算
# ====================================================================


def extract_l3_direction(parquet_path: Path) -> Dict[str, Any]:
    """从 predictions.parquet 的 entry_direction 列计算覆盖率和 short 比例."""
    import pandas as pd

    if not parquet_path.exists():
        return {}

    try:
        df = pd.read_parquet(parquet_path, columns=["entry_direction"])
    except Exception:
        # entry_direction 可能不在列中
        try:
            df = pd.read_parquet(parquet_path)
            if "entry_direction" not in df.columns:
                return {}
            df = df[["entry_direction"]]
        except Exception:
            return {}

    n_total = len(df)
    if n_total == 0:
        return {}

    has_direction = (df["entry_direction"] != 0).sum()
    coverage = float(has_direction / n_total)
    short_count = (df["entry_direction"] == -1).sum()
    short_ratio = float(short_count / n_total) if n_total > 0 else 0.0

    return {
        "coverage": round(coverage, 4),
        "short_ratio": round(short_ratio, 4),
        "n_samples": n_total,
    }


# ====================================================================
# L4: Gate KPI — 从 logs_gated.parquet 计算 or gate_optimization.json
# ====================================================================


def extract_l4_gate(
    gate_dir: Path,
    label_col: str = "is_good",
) -> Dict[str, Any]:
    """从 logs_gated.parquet 计算整体 gate KPI."""
    import pandas as pd

    # 尝试从 logs_gated.parquet 计算
    gated_path = gate_dir / "logs_gated.parquet"
    if not gated_path.exists():
        return _extract_l4_from_json(gate_dir)

    try:
        df = pd.read_parquet(gated_path)
    except Exception:
        return _extract_l4_from_json(gate_dir)

    # 自动生成 label
    if label_col not in df.columns:
        for rr_col in ["bpc_impulse_return_atr", "forward_rr", "rr", "return_atr"]:
            if rr_col in df.columns:
                df[label_col] = (df[rr_col] >= -0.8).astype(int)
                break
        else:
            return _extract_l4_from_json(gate_dir)

    n_all = len(df)
    if n_all == 0:
        return {}

    n_good_all = (df[label_col] == 1).sum()
    n_bad_all = n_all - n_good_all
    good_rate_all = n_good_all / n_all

    if "gate_decision" in df.columns:
        allowed = df[df["gate_decision"] == "allow"]
    else:
        # 如果没有 gate_decision 列，尝试用 gate_passed
        if "gate_passed" in df.columns:
            allowed = df[df["gate_passed"] == True]
        else:
            return _extract_l4_from_json(gate_dir)

    n_allowed = len(allowed)
    good_rate_allowed = allowed[label_col].mean() if n_allowed > 0 else 0

    lift = (good_rate_allowed / good_rate_all - 1) if good_rate_all > 0 else 0
    pass_rate = n_allowed / n_all if n_all > 0 else 0

    veto = df[~df.index.isin(allowed.index)]
    bad_rejection = (veto[label_col] == 0).sum() / n_bad_all if n_bad_all > 0 else 0
    good_retention = (
        (allowed[label_col] == 1).sum() / n_good_all if n_good_all > 0 else 0
    )

    return {
        "lift": round(float(lift), 6),
        "pass_rate": round(float(pass_rate), 4),
        "bad_rejection_rate": round(float(bad_rejection), 4),
        "good_retention_rate": round(float(good_retention), 4),
        "n_trades": n_all,
    }


def _extract_l4_from_json(gate_dir: Path) -> Dict[str, Any]:
    """Fallback: 从 gate_optimization.json 提取汇总."""
    path = gate_dir / "gate_optimization.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # gate_optimization.json 是 per-rule 的, 提取汇总
    n_optimized = sum(
        1
        for v in data.values()
        if v.get("status") in ("stable_plateau_found", "no_stable_plateau")
    )
    return {"n_optimized_rules": n_optimized, "source": "gate_optimization.json"}


# ====================================================================
# L5: Evidence KPI — 从 evidence_optimization.json 提取
# ====================================================================


def extract_l5_evidence(evidence_dir: Path) -> Dict[str, Any]:
    """从 evidence_optimization.json 提取 KPI."""
    path = evidence_dir / "evidence_optimization.json"
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    optimized = []
    for feat_id, result in data.items():
        if result.get("status") == "optimized":
            optimized.append(
                {
                    "feature": result.get("feature", feat_id),
                    "bad_suppression": result.get("bad_suppression", 0),
                    "sharpness": result.get("sharpness", 0),
                    "plateau_cv": result.get("plateau_cv", 0),
                }
            )

    if not optimized:
        return {"n_optimized": 0}

    avg_bad_supp = float(np.mean([e["bad_suppression"] for e in optimized]))
    avg_sharpness = float(np.mean([e["sharpness"] for e in optimized]))

    return {
        "n_optimized": len(optimized),
        "avg_bad_suppression": round(avg_bad_supp, 4),
        "avg_sharpness": round(avg_sharpness, 4),
        "features": optimized,
    }


# ====================================================================
# L6: Entry Filter KPI — 从 entry_filters.yaml 提取
# ====================================================================


def extract_l6_entry_filter(
    strategy: str,
    config_root: str = "config/strategies",
) -> Dict[str, Any]:
    """从 archetypes/entry_filters.yaml 的 backtest 段提取 KPI."""
    path = PROJECT_ROOT / config_root / strategy / "archetypes" / "entry_filters.yaml"
    if not path.exists():
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    filters = data.get("filters", data.get("conditions", []))
    if not isinstance(filters, list):
        return {}

    active_filters = []
    for f in filters:
        bt = f.get("backtest", {})
        if bt:
            active_filters.append(
                {
                    "name": f.get("name", f.get("feature", "unknown")),
                    "sharpe_pt": bt.get("sharpe_pt", bt.get("sharpe", 0)),
                    "trades": bt.get("trades", 0),
                    "win_rate": bt.get("win_rate", 0),
                }
            )

    if not active_filters:
        return {}

    return {
        "n_active_filters": len(active_filters),
        "filters_summary": active_filters,
    }


# ====================================================================
# L7: Execution / Backtest KPI
# ====================================================================


def extract_l7_execution(backtest_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """从 backtest 指标中提取执行层 KPI."""
    if not backtest_metrics:
        return {}
    return {
        "total_trades": backtest_metrics.get("total_trades", 0),
        "sharpe_per_trade": backtest_metrics.get("sharpe_per_trade", 0),
        "sharpe_annualized": backtest_metrics.get("sharpe_annualized", 0),
        "sharpe_daily": backtest_metrics.get("sharpe_daily", 0),
        "mean_r": backtest_metrics.get("mean_r", 0),
        "win_rate": backtest_metrics.get("win_rate", 0),
    }


# ====================================================================
# Feature distributions
# ====================================================================


def extract_l4_gate_rule_hit_rates(
    gate_dir: Path,
    strategy: str,
    config_root: str = "config/strategies",
    label_col: str = "is_good",
) -> Dict[str, Any]:
    """Extract per-rule deny_rate (hit_rate) from logs_gated.parquet.

    Used as Alpha Decay baseline: if a rule's hit_rate decays >50%,
    the rule may have lost effectiveness.
    """
    import pandas as pd

    gate_yaml = PROJECT_ROOT / config_root / strategy / "archetypes" / "gate.yaml"
    gated_path = gate_dir / "logs_gated.parquet"
    if not gate_yaml.exists() or not gated_path.exists():
        return {}

    try:
        gate_data = yaml.safe_load(gate_yaml.read_text(encoding="utf-8")) or {}
        df = pd.read_parquet(gated_path)
    except Exception:
        return {}

    hard_gates = gate_data.get("hard_gates", [])
    if not hard_gates:
        return {}

    # Auto-generate label if missing
    if label_col not in df.columns:
        for rr_col in ["bpc_impulse_return_atr", "forward_rr", "rr", "return_atr"]:
            if rr_col in df.columns:
                df[label_col] = (df[rr_col] >= -0.8).astype(int)
                break
        else:
            return {}

    rule_hit_rates: Dict[str, Any] = {}
    for gate in hard_gates:
        gate_id = gate.get("id", "unknown")
        when_clause = gate.get("when", {})
        for feat, condition in when_clause.items():
            if not isinstance(condition, dict) or feat not in df.columns:
                continue
            for op_key, threshold in condition.items():
                col = df[feat]
                if op_key == "value_lt":
                    deny_mask = col < threshold
                elif op_key == "value_gt":
                    deny_mask = col > threshold
                elif op_key == "value_le":
                    deny_mask = col <= threshold
                elif op_key == "value_ge":
                    deny_mask = col >= threshold
                else:
                    continue
                deny_rate = float(deny_mask.mean())
                # Precision of denial: among denied, what % were actually bad?
                denied = df[deny_mask]
                precision = 0.0
                if len(denied) > 0 and label_col in denied.columns:
                    precision = float((denied[label_col] == 0).mean())
                rule_hit_rates[f"{gate_id}__{feat}__{op_key}"] = {
                    "deny_rate": round(deny_rate, 6),
                    "precision": round(precision, 4),
                }

    return rule_hit_rates


def extract_l5_evidence_feature_ic(
    parquet_path: Path,
    target_col: str = "forward_rr",
    max_features: int = 100,
) -> Dict[str, float]:
    """Compute Spearman IC of each numeric feature vs forward_rr.

    Used as Alpha Decay baseline: if rolling IC decays >50%,
    the feature may have lost predictive power.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    if not parquet_path.exists():
        return {}

    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return {}

    if target_col not in df.columns:
        # Fallback names
        for alt in ["rr", "return_atr", "bpc_impulse_return_atr"]:
            if alt in df.columns:
                target_col = alt
                break
        else:
            return {}

    exclude_prefixes = (
        "timestamp",
        "datetime",
        "symbol",
        "split",
        "pred",
        "gate_",
        "is_good",
        "is_bad",
        "entry_direction",
        "Unnamed",
        "forward_",
    )
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not any(c.startswith(p) for p in exclude_prefixes) and c != target_col
    ]
    if len(numeric_cols) > max_features:
        numeric_cols = numeric_cols[:max_features]

    target = df[target_col].values
    valid_mask = ~np.isnan(target)

    ics: Dict[str, float] = {}
    for col in numeric_cols:
        vals = df[col].values.astype(np.float64)
        both_valid = valid_mask & ~np.isnan(vals)
        if both_valid.sum() < 30:
            continue
        try:
            rho, _ = spearmanr(vals[both_valid], target[both_valid])
            if np.isfinite(rho):
                ics[col] = round(float(rho), 6)
        except Exception:
            continue

    return ics


def compute_feature_distributions(
    parquet_path: Path,
    max_features: int = 200,
) -> Dict[str, Dict[str, float]]:
    """从 parquet 计算所有数值特征的分布统计."""
    import pandas as pd

    if not parquet_path.exists():
        return {}

    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return {}

    # 只保留数值列, 排除元数据列
    exclude_prefixes = (
        "timestamp",
        "datetime",
        "symbol",
        "split",
        "pred",
        "gate_",
        "is_good",
        "is_bad",
        "entry_direction",
        "Unnamed",
    )
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not any(c.startswith(p) for p in exclude_prefixes)
    ]

    # 限制特征数量
    if len(numeric_cols) > max_features:
        numeric_cols = numeric_cols[:max_features]

    distributions: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        s = df[col]
        total = len(s)
        nan_count = int(s.isna().sum())
        valid = s.dropna()

        if len(valid) == 0:
            distributions[col] = {
                "mean": 0.0,
                "std": 0.0,
                "p5": 0.0,
                "p95": 0.0,
                "nan_rate": 1.0,
            }
            continue

        distributions[col] = {
            "mean": round(float(valid.mean()), 6),
            "std": round(float(valid.std()), 6),
            "p5": round(float(valid.quantile(0.05)), 6),
            "p95": round(float(valid.quantile(0.95)), 6),
            "nan_rate": round(nan_count / total, 4) if total > 0 else 0.0,
        }

    return distributions


# ====================================================================
# Main export
# ====================================================================


def export_training_baseline(
    strategy: str,
    result_dir: Path,
    *,
    gate_dir: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
    backtest_metrics: Optional[Dict[str, Any]] = None,
    config_root: str = "config/strategies",
    training_period: Optional[Dict[str, str]] = None,
    holdout_period: Optional[Dict[str, str]] = None,
) -> Path:
    """导出 training_baseline.json 到 result_dir."""
    g_dir = gate_dir or result_dir
    e_dir = evidence_dir or result_dir

    # 寻找 predictions.parquet
    pred_path = result_dir / "predictions.parquet"
    if not pred_path.exists():
        pred_path = e_dir / "predictions.parquet"
    if not pred_path.exists():
        pred_path = g_dir / "predictions.parquet"

    # 寻找 features_labeled.parquet (更完整的特征集)
    feat_path = result_dir / "features_labeled.parquet"

    print(f"\n{'='*60}")
    print(f"📊 导出训练基线 — {strategy.upper()}")
    print(f"{'='*60}")

    # ── 逐层提取 ──
    layer_kpis: Dict[str, Any] = {}

    l2 = extract_l2_prefilter(strategy, config_root)
    if l2:
        layer_kpis["L2_prefilter"] = l2
        print(
            f"   L2 Prefilter: ✅ ({l2.get('n_positive_signals', 0)} positive signals)"
        )

    l3 = extract_l3_direction(pred_path)
    if l3:
        layer_kpis["L3_direction"] = l3
        print(
            f"   L3 Direction: ✅ (coverage={l3.get('coverage', 'N/A')}, short={l3.get('short_ratio', 'N/A')})"
        )

    l4 = extract_l4_gate(g_dir)
    if l4:
        layer_kpis["L4_gate"] = l4
        print(
            f"   L4 Gate:      ✅ (lift={l4.get('lift', 'N/A')}, pass_rate={l4.get('pass_rate', 'N/A')})"
        )

    l5 = extract_l5_evidence(e_dir)
    if l5:
        layer_kpis["L5_evidence"] = l5
        print(f"   L5 Evidence:  ✅ ({l5.get('n_optimized', 0)} optimized features)")

    l6 = extract_l6_entry_filter(strategy, config_root)
    if l6:
        layer_kpis["L6_entry_filter"] = l6
        print(f"   L6 EntryFilter: ✅ ({l6.get('n_active_filters', 0)} active filters)")

    # L7 backtest
    if backtest_metrics:
        layer_kpis["L7_execution"] = extract_l7_execution(backtest_metrics)
        print(
            f"   L7 Execution: ✅ (sharpe={backtest_metrics.get('sharpe_per_trade', 'N/A')})"
        )
    else:
        # 尝试从 report.json 加载
        report_path = result_dir / "report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                bm = report.get("backtest_metrics", {})
                if bm:
                    layer_kpis["L7_execution"] = extract_l7_execution(bm)
                    print(f"   L7 Execution: ✅ (from report.json)")
            except Exception:
                pass

    # ── 特征分布 ──
    dist_path = feat_path if feat_path.exists() else pred_path
    feature_distributions = compute_feature_distributions(dist_path)
    n_feat = len(feature_distributions)
    print(f"\n   特征分布: {n_feat} features from {dist_path.name}")

    # ── P5 Alpha Decay 先行指标基线 ──
    gate_rule_hit_rates = extract_l4_gate_rule_hit_rates(g_dir, strategy, config_root)
    if gate_rule_hit_rates:
        print(f"   Gate rule hit_rates: {len(gate_rule_hit_rates)} rules")

    evidence_feature_ics = extract_l5_evidence_feature_ic(dist_path)
    if evidence_feature_ics:
        print(f"   Evidence feature ICs: {len(evidence_feature_ics)} features")

    # ── 组装基线 ──
    baseline: Dict[str, Any] = {
        "version": str(date.today()),
        "strategy": strategy,
        "export_timestamp": str(date.today()),
    }
    if training_period:
        baseline["training_period"] = training_period
    if holdout_period:
        baseline["holdout_period"] = holdout_period

    baseline["layer_kpis"] = layer_kpis
    baseline["feature_distributions"] = feature_distributions
    if gate_rule_hit_rates:
        baseline["gate_rule_hit_rates"] = gate_rule_hit_rates
    if evidence_feature_ics:
        baseline["evidence_feature_ics"] = evidence_feature_ics

    # ── 保存 ──
    output_path = result_dir / "training_baseline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\n   ✅ Saved: {output_path}")
    print(f"   Layers: {len(layer_kpis)}, Features: {n_feat}")

    return output_path


# ====================================================================
# CLI
# ====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="导出训练基线 JSON (L2-L8 KPI + 特征分布)",
    )
    parser.add_argument("--result-dir", required=True, help="训练产出目录")
    parser.add_argument("--strategy", required=True, help="策略名 (bpc/fer/me)")
    parser.add_argument(
        "--gate-dir", default=None, help="Gate 训练产出目录 (默认=result-dir)"
    )
    parser.add_argument(
        "--evidence-dir", default=None, help="Evidence 训练产出目录 (默认=result-dir)"
    )
    parser.add_argument(
        "--backtest-json", default=None, help="Backtest 指标 JSON 字符串"
    )
    parser.add_argument(
        "--config-root", default="config/strategies", help="策略配置根目录"
    )
    parser.add_argument("--training-start", default=None, help="训练开始日期")
    parser.add_argument("--training-end", default=None, help="训练结束日期")
    parser.add_argument("--holdout-start", default=None, help="Holdout 开始日期")
    parser.add_argument("--holdout-end", default=None, help="Holdout 结束日期")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    gate_dir = Path(args.gate_dir) if args.gate_dir else None
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None

    backtest_metrics = None
    if args.backtest_json:
        try:
            backtest_metrics = json.loads(args.backtest_json)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid backtest JSON: {e}")
            return 1

    training_period = None
    if args.training_start and args.training_end:
        training_period = {"start": args.training_start, "end": args.training_end}

    holdout_period = None
    if args.holdout_start and args.holdout_end:
        holdout_period = {"start": args.holdout_start, "end": args.holdout_end}

    export_training_baseline(
        strategy=args.strategy,
        result_dir=result_dir,
        gate_dir=gate_dir,
        evidence_dir=evidence_dir,
        backtest_metrics=backtest_metrics,
        config_root=args.config_root,
        training_period=training_period,
        holdout_period=holdout_period,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
