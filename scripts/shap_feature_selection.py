#!/usr/bin/env python3
"""SHAP Walk-Forward 特征稳定性筛选 — 自动裁剪不稳定特征

设计文档: z实验_005_统一研究/SHAP特征筛选方案.md

核心算法:
  1. 按时间切 N 个 fold
  2. 每个 fold 独立训练 LightGBM + 计算 SHAP values
  3. 聚合: 在 >= stability_threshold 的 fold 中进入 top-K → "稳定特征"
  4. 输出 JSON 报告 + 可选写回 features_gate.yaml / features_evidence.yaml

用法:
    # 独立运行 (不写回配置)
    python scripts/shap_feature_selection.py \\
      --logs results/train_final_.../bpc/features_labeled.parquet \\
      --strategy bpc \\
      --output results/shap_report/

    # 写回配置 (裁剪 features_gate.yaml + features_evidence.yaml)
    python scripts/shap_feature_selection.py \\
      --logs results/train_final_.../bpc/features_labeled.parquet \\
      --strategy bpc \\
      --promote

    # 从 research_pipeline.yaml 读取参数
    python scripts/shap_feature_selection.py \\
      --logs results/train_final_.../bpc/features_labeled.parquet \\
      --strategy bpc \\
      --pipeline-config config/research_pipeline.yaml \\
      --promote
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# ============================================================
# Constants
# ============================================================
META_COLUMNS = {
    "timestamp",
    "datetime",
    "date",
    "symbol",
    "_symbol",
    "forward_rr",
    "success_no_rr_extreme",
    "gate_label",
    "signal",
    "direction",
    "direction_label",
    "atr",  # scale column, excluded from model input
}

# Default protected nodes (never pruned)
DEFAULT_PROTECTED_NODES = ["atr_f"]


# ============================================================
# Feature Dependencies Mapping
# ============================================================


def load_feature_deps(
    deps_path: str = "config/feature_dependencies.yaml",
) -> Dict[str, List[str]]:
    """Load feature_dependencies.yaml → {node_name: [output_columns]}."""
    p = Path(deps_path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    feats = data.get("features", {})
    mapping: Dict[str, List[str]] = {}
    for node, meta in feats.items():
        cols = meta.get("output_columns", []) if isinstance(meta, dict) else []
        mapping[node] = [str(c) for c in cols] if cols else []
    return mapping


def build_column_to_node_map(
    node_to_cols: Dict[str, List[str]],
) -> Dict[str, str]:
    """Reverse mapping: output_column → node_name."""
    col2node: Dict[str, str] = {}
    for node, cols in node_to_cols.items():
        for c in cols:
            col2node[c] = node
    return col2node


# ============================================================
# Data Loading
# ============================================================


def load_data(
    logs_path: str,
    label_col: str = "success_no_rr_extreme",
) -> Tuple[pd.DataFrame, List[str], str]:
    """Load features_labeled.parquet, identify feature columns.

    Returns:
        (df, feature_cols, label_col)
    """
    df = pd.read_parquet(logs_path)
    print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns from {logs_path}")

    # Ensure label exists
    if label_col not in df.columns:
        # Try auto-generating from forward_rr
        if "forward_rr" in df.columns:
            df[label_col] = (df["forward_rr"] >= -0.8).astype(int)
            print(f"   ⚠️  Auto-generated '{label_col}' from forward_rr >= -0.8")
        else:
            raise ValueError(
                f"Label column '{label_col}' not found. "
                f"Available: {sorted(df.columns)[:20]}..."
            )

    # Identify feature columns (exclude meta + label)
    exclude = META_COLUMNS | {label_col}
    # Also exclude any column starting with common meta prefixes
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude
        and not c.startswith("_")
        and not c.startswith("Unnamed")
        and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
    ]

    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   Label: '{label_col}' (positive rate: {df[label_col].mean():.2%})")

    # Ensure timestamp for time-based splitting
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
        print("   ℹ️  Using 'datetime' column as timestamp")
    else:
        raise ValueError(
            "'timestamp' or 'datetime' column required for time-based fold splitting"
        )

    return df, feature_cols, label_col


# ============================================================
# Walk-Forward Fold Splitting
# ============================================================


def split_time_folds(
    df: pd.DataFrame,
    n_folds: int = 4,
) -> List[pd.DataFrame]:
    """Split dataframe into N time-ordered folds of roughly equal size."""
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    fold_size = len(df_sorted) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(df_sorted)
        fold = df_sorted.iloc[start:end]
        folds.append(fold)
        ts_min = fold["timestamp"].min().strftime("%Y-%m-%d")
        ts_max = fold["timestamp"].max().strftime("%Y-%m-%d")
        print(f"   Fold {i+1}: {len(fold):,} rows ({ts_min} → {ts_max})")
    return folds


# ============================================================
# LightGBM Training + SHAP
# ============================================================


def train_and_shap(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    fold_id: int,
    sample_size: int = 2000,
) -> Optional[np.ndarray]:
    """Train LightGBM on fold data, compute SHAP values.

    Returns:
        mean_abs_shap: array of shape (n_features,) or None on failure
    """
    try:
        import lightgbm as lgb
        import shap
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        return None

    warnings.filterwarnings("ignore", category=UserWarning)

    # Remove rows with NaN
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]

    if len(X_clean) < 100:
        print(f"   ⚠️  Fold {fold_id}: too few clean rows ({len(X_clean)}), skipping")
        return None

    # Train LightGBM (same params as gate training)
    dtrain = lgb.Dataset(
        X_clean, label=y_clean, feature_name=feature_names, free_raw_data=False
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "seed": 42 + fold_id,
        "deterministic": True,
        "force_row_wise": True,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(0)],  # suppress output
    )

    # SHAP values
    sample_n = min(sample_size, len(X_clean))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_clean), size=sample_n, replace=False)
    X_sample = X_clean[idx]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"   ⚠️  Fold {fold_id}: SHAP computation failed: {e}")
        return None

    # Handle binary classification output
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_array = shap_values[1]  # positive class
        else:
            shap_array = shap_values[0]
    else:
        shap_array = shap_values

    mean_abs = np.abs(shap_array).mean(axis=0)
    return mean_abs


# ============================================================
# Stability Computation
# ============================================================


def compute_stability(
    shap_per_fold: List[Optional[np.ndarray]],
    feature_names: List[str],
    top_k: int = 20,
    stability_threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    """Compute cross-fold stability for each feature.

    Returns list of feature dicts sorted by mean_rank (best first).
    """
    valid_folds = [(i, s) for i, s in enumerate(shap_per_fold) if s is not None]
    n_valid = len(valid_folds)

    if n_valid == 0:
        print("   ❌ No valid folds, cannot compute stability")
        return []

    # For each fold, compute ranking
    rankings = {}  # feature_name → list of ranks (1-based)
    abs_shaps = {}  # feature_name → list of mean_abs_shap

    for feat_idx, feat_name in enumerate(feature_names):
        rankings[feat_name] = []
        abs_shaps[feat_name] = []

    for fold_idx, shap_vals in valid_folds:
        # Rank features by mean |SHAP| (descending)
        order = np.argsort(-shap_vals)
        for rank, feat_idx in enumerate(order, 1):
            feat_name = feature_names[feat_idx]
            rankings[feat_name].append(rank)
            abs_shaps[feat_name].append(float(shap_vals[feat_idx]))

    # Compute stability metrics
    results = []
    for feat_name in feature_names:
        ranks = rankings[feat_name]
        shaps = abs_shaps[feat_name]

        if not ranks:
            continue

        folds_in_top_k = sum(1 for r in ranks if r <= top_k)
        stability = folds_in_top_k / n_valid
        mean_rank = np.mean(ranks)
        mean_abs_shap = np.mean(shaps)

        results.append(
            {
                "name": feat_name,
                "mean_rank": round(float(mean_rank), 1),
                "folds_in_top_k": folds_in_top_k,
                "n_valid_folds": n_valid,
                "stability": round(float(stability), 3),
                "mean_abs_shap": round(float(mean_abs_shap), 6),
                "per_fold_ranks": ranks,
                "stable": stability >= stability_threshold,
            }
        )

    # Sort by mean_rank
    results.sort(key=lambda x: x["mean_rank"])
    return results


# ============================================================
# Report Generation
# ============================================================


def generate_report(
    stability_results: List[Dict[str, Any]],
    strategy: str,
    n_folds: int,
    top_k: int,
    stability_threshold: float,
    output_dir: Path,
    protected_nodes: List[str],
    col2node: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    """Generate JSON reports and print summary.

    Returns:
        (stable_columns, pruned_columns)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stable = [f for f in stability_results if f["stable"]]
    unstable = [f for f in stability_results if not f["stable"]]

    # Ensure protected nodes' columns are always in stable list
    stable_col_names = {f["name"] for f in stable}
    protected_cols = set()
    for node in protected_nodes:
        # node → output columns
        for f in stability_results:
            node_name = col2node.get(f["name"], "")
            if node_name == node and f["name"] not in stable_col_names:
                f["stable"] = True
                f["protected"] = True
                stable.append(f)
                stable_col_names.add(f["name"])
            if node_name == node:
                protected_cols.add(f["name"])

    stable_cols = [f["name"] for f in stable]
    pruned_cols = [f["name"] for f in unstable if f["name"] not in stable_col_names]

    # ── Print summary ──
    print(f"\n{'='*70}")
    print(f"📊 SHAP Feature Stability Report — {strategy.upper()}")
    print(f"{'='*70}")
    print(
        f"   Folds: {n_folds}  |  Top-K: {top_k}  |  Threshold: {stability_threshold}"
    )
    print(f"   Total features: {len(stability_results)}")
    print(f"   ✅ Stable: {len(stable)}  |  ❌ Pruned: {len(pruned_cols)}")
    print(f"   🔒 Protected: {protected_cols or 'none'}")

    print(f"\n   ── Stable Features (sorted by mean rank) ──")
    for i, f in enumerate(stable[:30], 1):
        node = col2node.get(f["name"], "?")
        prot = " 🔒" if f.get("protected") else ""
        print(
            f"   {i:3d}. {f['name']:<40s}  "
            f"rank={f['mean_rank']:<5.1f}  "
            f"stability={f['stability']:.0%}  "
            f"SHAP={f['mean_abs_shap']:.4f}"
            f"{prot}"
        )

    if unstable:
        print(f"\n   ── Pruned Features (top 10 by mean rank) ──")
        unstable_sorted = sorted(unstable, key=lambda x: x["mean_rank"])
        for f in unstable_sorted[:10]:
            print(
                f"   ✂️  {f['name']:<40s}  "
                f"rank={f['mean_rank']:<5.1f}  "
                f"stability={f['stability']:.0%}  "
                f"SHAP={f['mean_abs_shap']:.4f}"
            )

    # ── Save JSON reports ──
    # 1. Stable features (compact)
    stable_report = {
        "strategy": strategy,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_folds": n_folds,
        "top_k": top_k,
        "stability_threshold": stability_threshold,
        "total_features": len(stability_results),
        "stable_features": len(stable),
        "pruned_features": len(pruned_cols),
        "protected_nodes": protected_nodes,
        "features": [
            {
                "name": f["name"],
                "node": col2node.get(f["name"], "unknown"),
                "mean_rank": f["mean_rank"],
                "folds_in_top_k": f["folds_in_top_k"],
                "stability": f["stability"],
                "mean_abs_shap": f["mean_abs_shap"],
            }
            for f in stable
        ],
    }
    stable_path = output_dir / "shap_stable_features.json"
    with open(stable_path, "w", encoding="utf-8") as fp:
        json.dump(stable_report, fp, indent=2, ensure_ascii=False)
    print(f"\n   💾 Stable features → {stable_path}")

    # 2. Full report (all features)
    full_report = {
        "strategy": strategy,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "n_folds": n_folds,
            "top_k": top_k,
            "stability_threshold": stability_threshold,
        },
        "features": [
            {
                "name": f["name"],
                "node": col2node.get(f["name"], "unknown"),
                "mean_rank": f["mean_rank"],
                "folds_in_top_k": f["folds_in_top_k"],
                "stability": f["stability"],
                "mean_abs_shap": f["mean_abs_shap"],
                "per_fold_ranks": f["per_fold_ranks"],
                "stable": f["stable"],
            }
            for f in stability_results
        ],
    }
    full_path = output_dir / "shap_feature_report.json"
    with open(full_path, "w", encoding="utf-8") as fp:
        json.dump(full_report, fp, indent=2, ensure_ascii=False)
    print(f"   💾 Full report    → {full_path}")

    return stable_cols, pruned_cols


# ============================================================
# SHAP Plots (optional)
# ============================================================


def generate_plots(
    shap_per_fold: List[Optional[np.ndarray]],
    feature_names: List[str],
    stability_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate SHAP visualizations (optional, requires matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   ⚠️  matplotlib not available, skipping plots")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Stability heatmap ──
    valid_folds = [(i, s) for i, s in enumerate(shap_per_fold) if s is not None]
    if not valid_folds:
        return

    # Top 25 features by mean rank
    top_features = stability_results[:25]
    feat_idx_map = {name: idx for idx, name in enumerate(feature_names)}

    n_feats = len(top_features)
    n_folds_valid = len(valid_folds)

    # Build rank matrix
    rank_matrix = np.full((n_feats, n_folds_valid), np.nan)
    for fold_j, (fold_orig_idx, shap_vals) in enumerate(valid_folds):
        order = np.argsort(-shap_vals)
        rank_lookup = {feat_idx: rank + 1 for rank, feat_idx in enumerate(order)}
        for feat_i, feat_info in enumerate(top_features):
            fi = feat_idx_map.get(feat_info["name"])
            if fi is not None and fi in rank_lookup:
                rank_matrix[feat_i, fold_j] = rank_lookup[fi]

    fig, ax = plt.subplots(figsize=(max(6, n_folds_valid * 1.5), max(8, n_feats * 0.4)))
    im = ax.imshow(
        rank_matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(feature_names)
    )

    ax.set_xticks(range(n_folds_valid))
    ax.set_xticklabels([f"Fold {i+1}" for i, _ in valid_folds], fontsize=9)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(
        [f"{f['name']} ({'✅' if f['stable'] else '❌'})" for f in top_features],
        fontsize=8,
    )

    # Annotate cells with rank
    for i in range(n_feats):
        for j in range(n_folds_valid):
            val = rank_matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val > len(feature_names) * 0.6 else "black",
                )

    ax.set_title(
        "Feature Rank Stability Across Folds (lower = more important)", fontsize=11
    )
    plt.colorbar(im, ax=ax, label="Rank")
    plt.tight_layout()
    plt.savefig(plots_dir / "stability_heatmap.png", dpi=150)
    plt.close()
    print(f"   📊 Heatmap → {plots_dir / 'stability_heatmap.png'}")

    # ── Per-fold SHAP bar charts ──
    for fold_j, (fold_orig_idx, shap_vals) in enumerate(valid_folds):
        top_n = min(20, len(feature_names))
        order = np.argsort(-shap_vals)[:top_n]

        fig, ax = plt.subplots(figsize=(8, 6))
        names = [feature_names[i] for i in order]
        vals = [shap_vals[i] for i in order]
        ax.barh(range(top_n), vals[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"Fold {fold_orig_idx + 1}: Top-{top_n} SHAP Importance")
        plt.tight_layout()
        plt.savefig(plots_dir / f"fold_{fold_orig_idx + 1}_shap_bar.png", dpi=150)
        plt.close()

    print(f"   📊 Per-fold plots → {plots_dir}/")


# ============================================================
# Promote: Write Back to features YAML
# ============================================================


def promote_features(
    stable_columns: List[str],
    pruned_columns: List[str],
    strategy: str,
    strategies_root: str,
    col2node: Dict[str, str],
    node2cols: Dict[str, List[str]],
    apply_to: Optional[List[str]] = None,
    protected_nodes: Optional[List[str]] = None,
) -> None:
    """Generate SHAP-pruned feature files (features_gate_shap.yaml / features_evidence_shap.yaml).

    IMPORTANT: Never modifies the original features_gate.yaml / features_evidence.yaml.
    The originals serve as the full candidate pool for future SHAP runs.

    Aggressive pruning: only keep nodes that have at least one SHAP-stable column,
    plus any protected nodes (e.g. atr_f). All other nodes are removed.
    """
    if apply_to is None:
        apply_to = ["features_gate.yaml", "features_evidence.yaml"]
    if protected_nodes is None:
        protected_nodes = []
    protected_set = set(protected_nodes)

    # Collect nodes that have at least one stable column
    stable_nodes = set()
    for col in stable_columns:
        node = col2node.get(col)
        if node:
            stable_nodes.add(node)

    # Keep = stable_nodes + protected_nodes
    keep_nodes = stable_nodes | protected_set

    if not keep_nodes:
        print(f"\n   \u26a0\ufe0f  No stable or protected nodes, skipping pruning")
        return

    config_dir = Path(strategies_root) / strategy
    for yaml_name in apply_to:
        yaml_path = config_dir / yaml_name
        if not yaml_path.exists():
            print(f"   ⚠️  {yaml_path} not found, skipping")
            continue

        text = yaml_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not data or "feature_pipeline" not in data:
            continue

        fp = data["feature_pipeline"]
        requested = fp.get("requested_features", [])
        if not requested:
            continue

        original_count = len(requested)
        new_requested = [n for n in requested if n in keep_nodes]
        removed = [n for n in requested if n not in keep_nodes]

        if not removed:
            print(f"   \u2139\ufe0f  {yaml_name}: no nodes to remove")
            continue

        fp["requested_features"] = new_requested

        # Add SHAP metadata
        data["_shap_pruned"] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": yaml_name,
            "removed_nodes": sorted(removed),
            "kept_nodes": sorted(keep_nodes & set(requested)),
            "stable_nodes": sorted(stable_nodes),
            "protected_nodes": sorted(protected_set & set(requested)),
            "original_count": original_count,
            "new_count": len(new_requested),
        }

        # Write to _shap.yaml (never touch original)
        stem = yaml_name.replace(".yaml", "")
        shap_yaml_name = f"{stem}_shap.yaml"
        shap_yaml_path = config_dir / shap_yaml_name
        shap_yaml_path.write_text(
            yaml.dump(
                data, default_flow_style=False, allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )
        print(
            f"   \u270f\ufe0f  {shap_yaml_name}: {original_count} \u2192 {len(new_requested)} nodes "
            f"(removed {len(removed)}: {', '.join(sorted(removed)[:5])}{'...' if len(removed) > 5 else ''})"
        )


# ============================================================
# Main Pipeline
# ============================================================


def run_shap_selection(
    logs_path: str,
    strategy: str,
    label_col: str = "success_no_rr_extreme",
    n_folds: int = 4,
    top_k: int = 20,
    stability_threshold: float = 0.75,
    min_stable_features: int = 8,
    sample_size: int = 2000,
    protected_nodes: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    strategies_root: str = "config/strategies",
    promote: bool = False,
    apply_to: Optional[List[str]] = None,
    plots: bool = True,
    cutoff_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Run complete SHAP feature selection pipeline.

    Args:
        cutoff_date: Val/Test 切分边界 (ISO-8601, e.g. "2025-10-01").
            非空时仅使用 timestamp < cutoff_date 的样本做 SHAP 训练与 stability
            评估, 避免 Test 段泄漏到特征筛选. 对齐 gate optimize 的 --cutoff-date.
    """

    if protected_nodes is None:
        protected_nodes = list(DEFAULT_PROTECTED_NODES)

    # ── Load data ──
    print(f"\n{'='*70}")
    print(f"🔬 SHAP Feature Selection — {strategy.upper()}")
    print(f"{'='*70}")

    df, feature_cols, label_col = load_data(logs_path, label_col)

    # ── 可选 Val/Test 切分: 防止 Test 段泄漏进 SHAP 稳定性评估 ──
    if cutoff_date:
        _n_before = len(df)
        _cut_ts = pd.to_datetime(cutoff_date, utc=True)
        df = df[df["timestamp"] < _cut_ts].reset_index(drop=True)
        print(
            f"   🛡️  Val-only cutoff {cutoff_date}: {_n_before:,} → {len(df):,} rows "
            f"({len(df)/max(_n_before,1):.1%} retained)"
        )
        if len(df) < 200:
            print(f"   ⚠️  Val-only subset 行数过少 ({len(df)}), SHAP 结果可能不稳定")

    # ── Load feature dependencies ──
    node2cols = load_feature_deps()
    col2node = build_column_to_node_map(node2cols)

    # ── Split into folds ──
    print(f"\n📐 Splitting into {n_folds} time folds...")
    folds = split_time_folds(df, n_folds)

    # ── Train + SHAP per fold ──
    print(f"\n🧠 Training LightGBM + SHAP per fold...")
    shap_per_fold: List[Optional[np.ndarray]] = []

    for i, fold_df in enumerate(folds):
        print(f"\n   ── Fold {i+1}/{n_folds} ({len(fold_df):,} rows) ──")
        X = fold_df[feature_cols].values
        y = fold_df[label_col].values

        mean_abs = train_and_shap(
            X, y, feature_cols, fold_id=i, sample_size=sample_size
        )
        shap_per_fold.append(mean_abs)

        if mean_abs is not None:
            top3_idx = np.argsort(-mean_abs)[:3]
            top3 = [(feature_cols[j], mean_abs[j]) for j in top3_idx]
            print(f"      Top 3: {', '.join(f'{n}={v:.4f}' for n, v in top3)}")

    # ── Compute stability ──
    print(f"\n📊 Computing cross-fold stability...")
    stability_results = compute_stability(
        shap_per_fold, feature_cols, top_k, stability_threshold
    )

    if not stability_results:
        print("   ❌ No stability results, aborting")
        return {"error": "no_stability_results"}

    # ── Check min_stable_features ──
    stable_count = sum(1 for f in stability_results if f["stable"])
    if stable_count < min_stable_features:
        print(
            f"\n   ⚠️  Only {stable_count} stable features (min={min_stable_features}). "
            f"Skipping pruning — 宁可不裁也不裁错。"
        )
        # Still generate report but don't promote
        promote = False

    # ── Generate report ──
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path(logs_path).parent / "shap"

    stable_cols, pruned_cols = generate_report(
        stability_results,
        strategy,
        n_folds,
        top_k,
        stability_threshold,
        out_path,
        protected_nodes,
        col2node,
    )

    # ── Generate plots ──
    if plots:
        print(f"\n📈 Generating plots...")
        generate_plots(shap_per_fold, feature_cols, stability_results, out_path)

    # ── Promote (write back to YAML) ──
    if promote and pruned_cols:
        print(f"\n✏️  Promoting: writing pruned features back to YAML...")
        promote_features(
            stable_cols,
            pruned_cols,
            strategy,
            strategies_root,
            col2node,
            node2cols,
            apply_to=apply_to,
            protected_nodes=protected_nodes,
        )

    # ── Summary ──
    result = {
        "strategy": strategy,
        "total_features": len(feature_cols),
        "stable_features": len(stable_cols),
        "pruned_features": len(pruned_cols),
        "stable_nodes": sorted(set(col2node.get(c, "?") for c in stable_cols)),
        "pruned_nodes": sorted(set(col2node.get(c, "?") for c in pruned_cols)),
        "output_dir": str(out_path),
    }

    print(f"\n{'='*70}")
    print(f"✅ SHAP Feature Selection Complete")
    print(
        f"   Stable: {len(stable_cols)} columns → {len(result['stable_nodes'])} nodes"
    )
    print(
        f"   Pruned: {len(pruned_cols)} columns → {len(result['pruned_nodes'])} nodes"
    )
    print(f"{'='*70}\n")

    return result


# ============================================================
# CLI
# ============================================================


def main():
    p = argparse.ArgumentParser(
        description="SHAP Walk-Forward 特征稳定性筛选",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 独立运行 (不写回)
  python scripts/shap_feature_selection.py \\
    --logs results/train_final_.../bpc/features_labeled.parquet \\
    --strategy bpc --output results/shap_report/

  # 写回配置
  python scripts/shap_feature_selection.py \\
    --logs results/train_final_.../bpc/features_labeled.parquet \\
    --strategy bpc --promote

  # 从 pipeline config 读取参数
  python scripts/shap_feature_selection.py \\
    --logs results/train_final_.../bpc/features_labeled.parquet \\
    --strategy bpc --pipeline-config config/research_pipeline.yaml --promote
        """,
    )

    p.add_argument("--logs", required=True, help="features_labeled.parquet 路径")
    p.add_argument("--strategy", required=True, help="策略名 (bpc/fer/me)")
    p.add_argument(
        "--label-col",
        default="success_no_rr_extreme",
        help="标签列名 (default: success_no_rr_extreme)",
    )
    p.add_argument("--n-folds", type=int, default=4, help="时间窗口数 (default: 4)")
    p.add_argument(
        "--top-k", type=int, default=20, help="每 fold 取 Top-K (default: 20)"
    )
    p.add_argument(
        "--stability-threshold",
        type=float,
        default=0.75,
        help="稳定性阈值 (default: 0.75)",
    )
    p.add_argument(
        "--min-stable",
        type=int,
        default=8,
        help="最少稳定特征数, 不足则跳过裁剪 (default: 8)",
    )
    p.add_argument(
        "--sample-size", type=int, default=2000, help="SHAP 采样数 (default: 2000)"
    )
    p.add_argument("--output", help="输出目录 (默认: logs同级/shap/)")
    p.add_argument(
        "--strategies-root",
        default="config/strategies",
        help="策略配置根目录 (default: config/strategies)",
    )
    p.add_argument(
        "--promote",
        action="store_true",
        help="写回裁剪后的 features_gate.yaml / features_evidence.yaml",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="不生成 SHAP 图表",
    )
    p.add_argument(
        "--pipeline-config",
        help="从 research_pipeline.yaml 读取 SHAP 参数 (覆盖 CLI 参数)",
    )
    p.add_argument(
        "--cutoff-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Val/Test 切分边界. 仅用 timestamp < cutoff-date 的样本做 SHAP 训练 + "
        "stability 评估, 避免 Test 段泄漏 (推荐 pipeline 注入 test_start).",
    )

    args = p.parse_args()

    # 记录 CLI 显式传入的参数 (与 argparse default 不同 → 显式传入)
    _cli_explicit = set()
    _param_defaults = {
        "n_folds": 4,
        "top_k": 20,
        "stability_threshold": 0.75,
        "min_stable": 8,
        "sample_size": 2000,
    }
    for key, default_val in _param_defaults.items():
        if getattr(args, key) != default_val:
            _cli_explicit.add(key)

    # ── Load pipeline config overrides (CLI 显式传入的不被覆盖) ──
    protected_nodes = list(DEFAULT_PROTECTED_NODES)
    apply_to = None

    if args.pipeline_config:
        cfg_path = Path(args.pipeline_config)
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            shap_cfg = cfg.get("shap_feature_selection", {})
            if shap_cfg:
                print(f"   📋 Loading SHAP config from {cfg_path}")
                if "n_folds" in shap_cfg and "n_folds" not in _cli_explicit:
                    args.n_folds = shap_cfg["n_folds"]
                if "top_k" in shap_cfg and "top_k" not in _cli_explicit:
                    args.top_k = shap_cfg["top_k"]
                if (
                    "stability_threshold" in shap_cfg
                    and "stability_threshold" not in _cli_explicit
                ):
                    args.stability_threshold = shap_cfg["stability_threshold"]
                if (
                    "min_stable_features" in shap_cfg
                    and "min_stable" not in _cli_explicit
                ):
                    args.min_stable = shap_cfg["min_stable_features"]
                if "protected_nodes" in shap_cfg:
                    protected_nodes = shap_cfg["protected_nodes"]
                if "apply_to" in shap_cfg:
                    apply_to = shap_cfg["apply_to"]
                if not shap_cfg.get("enabled", True):
                    print("   ⏭️  SHAP feature selection disabled in config")
                    return

    run_shap_selection(
        logs_path=args.logs,
        strategy=args.strategy,
        label_col=args.label_col,
        n_folds=args.n_folds,
        top_k=args.top_k,
        stability_threshold=args.stability_threshold,
        min_stable_features=args.min_stable,
        sample_size=args.sample_size,
        protected_nodes=protected_nodes,
        output_dir=args.output,
        strategies_root=args.strategies_root,
        promote=args.promote,
        apply_to=apply_to,
        plots=not args.no_plots,
        cutoff_date=args.cutoff_date,
    )


if __name__ == "__main__":
    main()
