#!/usr/bin/env python3
"""
Generate top_factors.json from grid_search best_combination results.

Usage:
    python scripts/utils/generate_top_factors_from_grid_search.py \
        results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_grid_search_20251114_100417
"""

import json
import sys
from pathlib import Path


def generate_top_factors_from_grid_search(grid_search_dir: str) -> None:
    """Generate top_factors.json from grid_search best_combination results."""
    grid_search_path = Path(grid_search_dir)
    
    if not grid_search_path.exists():
        print(f"❌ Directory not found: {grid_search_dir}")
        return
    
    best_combination_dir = grid_search_path / "best_combination"
    if not best_combination_dir.exists():
        print(f"❌ best_combination directory not found: {best_combination_dir}")
        return
    
    # Try to read from best_combination_summary.json first
    summary_file = best_combination_dir / "best_combination_summary.json"
    selected_features = None
    
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                selected_features = summary.get('selected_features')
                grid_search_params = summary.get('grid_search_params', {})
                robustness_score = summary.get('robustness_score', 0)
                icir = summary.get('icir', 0)
                sharpe_ratio = summary.get('sharpe_ratio', 0)
        except Exception as e:
            print(f"⚠️ Failed to read best_combination_summary.json: {e}")
    
    # Fallback to selected_features.txt
    if not selected_features:
        features_file = best_combination_dir / "selected_features.txt"
        if features_file.exists():
            try:
                with open(features_file, 'r', encoding='utf-8') as f:
                    selected_features = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"⚠️ Failed to read selected_features.txt: {e}")
    
    if not selected_features:
        print("❌ Could not find selected_features in best_combination_summary.json or selected_features.txt")
        return
    
    # Generate top_factors.json
    top_factors_data = {
        "top_factors": [{"name": factor} for factor in selected_features],
        "count": len(selected_features),
        "source": "grid_search",
        "stage": "Stage 3: Representative features (from grid search best combination)",
        "effective": True,
    }
    
    # Add grid_search_params if available
    if 'grid_search_params' in locals():
        top_factors_data["grid_search_params"] = grid_search_params
        top_factors_data["robustness_score"] = robustness_score
        top_factors_data["icir"] = icir
        top_factors_data["sharpe_ratio"] = sharpe_ratio
    
    # Write top_factors.json
    top_factors_file = best_combination_dir / "top_factors.json"
    try:
        with open(top_factors_file, 'w', encoding='utf-8') as f:
            json.dump(top_factors_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Generated top_factors.json with {len(selected_features)} features")
        print(f"   Location: {top_factors_file}")
    except Exception as e:
        print(f"❌ Failed to write top_factors.json: {e}")
        return
    
    # Also create a representative_factors.json for compatibility
    rep_factors_file = best_combination_dir / "representative_factors.json"
    try:
        rep_factors_data = {
            "representative_factors": selected_features,
            "count": len(selected_features),
            "stage": "Stage 3: Correlation-based representative selection (from grid search)",
            "description": "Features selected from grid search best combination",
            "effective": True,
        }
        with open(rep_factors_file, 'w', encoding='utf-8') as f:
            json.dump(rep_factors_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Generated representative_factors.json")
    except Exception as e:
        print(f"⚠️ Failed to write representative_factors.json: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_top_factors_from_grid_search.py <grid_search_dir>")
        print("\nExample:")
        print("  python scripts/utils/generate_top_factors_from_grid_search.py \\")
        print("    results/dim_compare/BTCUSDT-ETHUSDT_comprehensive_grid_search_20251114_100417")
        sys.exit(1)
    
    grid_search_dir = sys.argv[1]
    generate_top_factors_from_grid_search(grid_search_dir)

