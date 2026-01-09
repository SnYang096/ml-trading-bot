"""
Feature selectors for nnmultihead training/eval.

Why:
- FeatureStore layers can contain many columns; for controlled experiments we want
  to train on exactly the columns implied by a requested feature-function set.
- Tree pipeline uses "all non-base columns" by default; this is too broad for nn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


def _load_feature_dependencies(feature_deps_path: str) -> Dict[str, Any]:
    p = Path(feature_deps_path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return obj.get("features", {}) or {}


def select_columns_from_requested_features(
    df: pd.DataFrame,
    all_columns: List[str],
    *,
    requested_features: List[str],
    feature_deps_path: str = "config/feature_dependencies.yaml",
    drop_constant: bool = True,
    exclude_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Return feature columns implied by requested feature functions (xxx_f),
    filtered to those present in df.

    Notes:
    - `requested_features` should be feature compute function names; backward-compatible
      aliases (e.g., volume_profile_vpvr) are allowed if present in feature_dependencies.
    - This returns output columns (not compute function names).
    - By default, drops strictly-constant columns (helps avoid useless inputs like 0-only flags).
    """
    feats = _load_feature_dependencies(feature_deps_path)

    # Reverse map: output_column -> feature node (compute function name, with _f suffix).
    # This lets callers pass either:
    # - feature nodes (recommended): "compression_duration_f" -> include ALL its output columns
    # - output columns (fine-grained): "compression_duration" -> include ONLY that column
    #
    # Why:
    # - Tree-side semantic expansion often operates at output-column granularity.
    # - For nn, the *final model inputs are columns*, while nodes are compute units.
    out_to_node: Dict[str, str] = {}
    for node, info in (feats or {}).items():
        if not isinstance(info, dict):
            continue
        for c in info.get("output_columns") or []:
            c = str(c)
            if c and c not in out_to_node:
                out_to_node[c] = str(node)

    want_cols: List[str] = []
    for name in requested_features or []:
        name = str(name).strip()
        if not name:
            continue

        # Case 1) feature node name (compute function): include all its outputs
        info = feats.get(name, None)
        if isinstance(info, dict):
            out_cols = info.get("output_columns") or []
            for c in out_cols:
                want_cols.append(str(c))
            continue

        # Case 2) output column name: include only that column if known
        if name in out_to_node:
            want_cols.append(name)
            continue

    # De-dup while preserving order
    seen = set()
    want_cols = [c for c in want_cols if not (c in seen or seen.add(c))]

    # Intersect with df columns
    present = [c for c in want_cols if c in set(all_columns)]

    # Optional: explicitly exclude some columns from model input (still computed in df for labels/contracts).
    excl = {str(x) for x in (exclude_columns or []) if str(x).strip()}
    if excl:
        present = [c for c in present if c not in excl]

    if not drop_constant:
        return present

    # Drop strict constants (all non-null values identical)
    kept: List[str] = []
    for c in present:
        s = pd.to_numeric(df[c], errors="coerce")
        v = s.dropna().unique()
        if len(v) <= 1:
            continue
        kept.append(c)
    return kept
