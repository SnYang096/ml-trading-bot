#!/usr/bin/env python3
"""
Run SHAP analysis for a cross-sectional boosting model.

Example:
    python scripts/cross_sectional/run_shap_analysis.py \
        --model results/cross_sectional/models/cs_boosting.joblib \
        --panel results/feature_exports/cs_panel.parquet \
        --feature-file results/cross_sectional/selected_factors.txt \
        --target future_return_12 \
        --output-dir results/cross_sectional/shap_reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SHAP is required for analysis. Install with `pip install shap`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP summary/dependence plots for cross-sectional models."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model joblib/pkl (expects HistGradientBoostingRegressor).",
    )
    parser.add_argument(
        "--panel",
        required=True,
        help="Path to cross-sectional panel parquet/CSV (MultiIndex with timestamp,symbol).",
    )
    parser.add_argument(
        "--feature-file",
        type=str,
        default=None,
        help="Optional text file listing feature columns (one per line). Overrides auto-detect.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column in panel (default auto-detect first 'future_return_' column).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of top features for dependence plots (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cross_sectional/shap_reports",
        help="Directory to store SHAP plots (summary.png, interaction.png, dependence/).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of samples to use for SHAP computation (default: 2000).",
    )
    parser.add_argument(
        "--interaction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate SHAP interaction values plot for top pair (default: True).",
    )
    return parser.parse_args()


def load_panel(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(file)

    if file.suffix.lower() == ".parquet":
        df = pd.read_parquet(file)
    else:
        df = pd.read_csv(file)

    if not isinstance(df.index, pd.MultiIndex):
        if {"timestamp", "symbol"}.issubset(df.columns):
            df = df.set_index(["timestamp", "symbol"])
        else:
            raise ValueError(
                "Panel must have MultiIndex (timestamp, symbol) or columns 'timestamp','symbol'."
            )

    ts = pd.to_datetime(df.index.get_level_values(0), utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("NaT detected in timestamp index.")
    df.index = pd.MultiIndex.from_arrays(
        [ts, df.index.get_level_values(1)], names=["timestamp", "symbol"]
    )
    return df


def detect_target(panel: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in panel.columns:
            raise ValueError(f"Target column '{explicit}' not found in panel.")
        return explicit
    candidates = [c for c in panel.columns if c.startswith("future_return")]
    if not candidates:
        raise ValueError("Unable to detect target column; specify with --target.")
    return candidates[0]


def load_features(panel: pd.DataFrame, feature_file: Optional[str]) -> List[str]:
    if feature_file:
        path = Path(feature_file)
        if not path.exists():
            raise FileNotFoundError(path)
        features = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        features = [f for f in features if f in panel.columns]
        if not features:
            raise ValueError(
                "Feature file did not contain any columns present in panel."
            )
        return features

    exclude = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp",
        "symbol",
    }
    features = [
        col
        for col in panel.columns
        if col not in exclude and not col.startswith("future_return")
    ]
    if not features:
        raise ValueError("No feature columns detected.")
    return features


def subsample_panel(
    panel: pd.DataFrame,
    features: List[str],
    target_col: str,
    *,
    max_samples: int,
) -> pd.DataFrame:
    df = panel[features + [target_col]].dropna()
    if len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=42)


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    print(f"📦 Loading model from {model_path}")
    model = joblib.load(model_path)

    print("📊 Loading panel data...")
    panel = load_panel(args.panel)
    target_col = detect_target(panel, args.target)

    features = load_features(panel, args.feature_file)
    df = subsample_panel(panel, features, target_col, max_samples=args.max_samples)
    print(f"🧮 SHAP dataset shape: {df.shape}")

    X = df[features].values
    y = df[target_col].values

    print("⚙️  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dependence_dir = output_dir / "dependence"
    dependence_dir.mkdir(parents=True, exist_ok=True)

    features_df = df[features]

    print("🖼  Generating SHAP summary plot...")
    shap.summary_plot(
        shap_values,
        features_df,
        plot_type="bar",
        show=False,
        color=shap.plots.colors.blue_rgb,
    )
    shap_summary_path = output_dir / "summary_bar.png"
    shap.plots._utils.plt.savefig(shap_summary_path, bbox_inches="tight")
    shap.plots._utils.plt.close()

    shap.summary_plot(
        shap_values,
        features_df,
        plot_type="dot",
        show=False,
    )
    shap_dot_path = output_dir / "summary_dot.png"
    shap.plots._utils.plt.savefig(shap_dot_path, bbox_inches="tight")
    shap.plots._utils.plt.close()

    top_indices = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][: args.topk]
    top_features = [features[i] for i in top_indices]
    print(f"📌 Top features for dependence plots: {top_features}")

    for feat in top_features:
        shap.dependence_plot(
            feat,
            shap_values,
            features_df,
            show=False,
            interaction_index=None,
        )
        dependence_path = dependence_dir / f"{feat}_dependence.png"
        shap.plots._utils.plt.savefig(dependence_path, bbox_inches="tight")
        shap.plots._utils.plt.close()

    interaction_path = None
    interaction_pair = None
    if args.interaction and hasattr(explainer, "shap_interaction_values"):
        print("🔗 Computing SHAP interaction values for top pair...")
        interaction_values = explainer.shap_interaction_values(X)
        mean_interactions = np.abs(interaction_values).mean(axis=0)
        np.fill_diagonal(mean_interactions, 0.0)
        top_pair_idx = np.unravel_index(
            np.argmax(mean_interactions), mean_interactions.shape
        )
        interaction_pair = (features[top_pair_idx[0]], features[top_pair_idx[1]])
        shap.dependence_plot(
            interaction_pair[0],
            shap_values,
            features_df,
            interaction_index=interaction_pair[1],
            show=False,
        )
        interaction_path = output_dir / "interaction_dependence.png"
        shap.plots._utils.plt.savefig(interaction_path, bbox_inches="tight")
        shap.plots._utils.plt.close()

    manifest = {
        "model": str(model_path),
        "panel": args.panel,
        "feature_file": args.feature_file,
        "target": target_col,
        "num_samples": int(len(df)),
        "top_features": top_features,
        "interaction_pair": interaction_pair,
        "plots": {
            "summary_bar": str(shap_summary_path),
            "summary_dot": str(shap_dot_path),
            "dependence_dir": str(dependence_dir),
            "interaction": str(interaction_path) if interaction_path else None,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"✅ SHAP analysis complete. Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
