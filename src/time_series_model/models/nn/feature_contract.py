from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass(frozen=True)
class FeatureContract:
    """
    A light-weight, nnmultihead-only feature contract.

    This is intentionally NOT part of StrategyConfigLoader to avoid affecting
    the tree-model strategy pipeline. nnmultihead scripts can optionally load it.
    """

    minimal_required_cols: List[str]
    # Backward compatible:
    # - legacy: List[str] (block names only)
    # - new: Dict[str, List[str]] where values are column patterns (fnmatch) or exact columns
    optional_blocks: Union[List[str], Dict[str, List[str]]]
    missingness_policy: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "minimal_required_cols": list(self.minimal_required_cols),
            "optional_blocks": list(self.optional_blocks),
            "missingness_policy": dict(self.missingness_policy or {}),
        }


def _load_feature_dependencies(
    feature_deps_path: str | Path = "config/feature_dependencies.yaml",
) -> Dict[str, Any]:
    """
    Load feature_dependencies.yaml to get accurate output_columns mapping.

    Returns:
        Dict with 'features' key containing feature definitions.
    """
    feature_deps_path = Path(feature_deps_path)
    if not feature_deps_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        feature_deps_path = project_root / "config" / "feature_dependencies.yaml"
        if not feature_deps_path.exists():
            return {}

    try:
        with open(feature_deps_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _feature_name_to_output_columns(
    feat_name: str, feature_deps: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Convert feature function name to output column names.

    Uses feature_dependencies.yaml to get accurate output_columns mapping.
    Falls back to simple heuristic (remove "_f" suffix) if feature_deps not available.

    Examples:
        - "atr_f" -> ["atr"]
        - "macd_f" -> ["macd", "macd_signal", "macd_histogram"] (from feature_deps)
        - "compression_duration_f" -> ["compression_duration"] (fallback)

    Args:
        feat_name: Feature function name (e.g., "atr_f")
        feature_deps: Optional pre-loaded feature dependencies dict.
                     If None, will try to load from config/feature_dependencies.yaml.

    Returns:
        List of output column names for this feature.
    """
    if not isinstance(feat_name, str):
        return []

    # Try to get accurate mapping from feature_dependencies.yaml
    if feature_deps is None:
        deps = _load_feature_dependencies()
        feature_deps = deps.get("features", {})
    else:
        feature_deps = (
            feature_deps.get("features", {}) if isinstance(feature_deps, dict) else {}
        )

    # Look up feature definition
    if feat_name in feature_deps:
        feat_info = feature_deps[feat_name]
        output_cols = feat_info.get("output_columns", [])
        if output_cols:
            return list(output_cols)

    # Fallback: simple heuristic (remove "_f" suffix)
    # This handles cases where feature_deps is not available or feature not found
    base_name = feat_name.rstrip("_f")
    if base_name:
        return [base_name]
    return []


def _derive_contract_from_requested_features(
    requested_features: Any,
    feature_pipeline: Dict[str, Any],
    feature_deps: Optional[Dict[str, Any]] = None,
) -> Optional[FeatureContract]:
    """
    Derive feature contract from requested_features structure.

    Supports two formats:
    1. New format (structured):
       requested_features:
         required:
           - atr_f
           - trend_r2_20_f
         optional_blocks:
           compression_blocks:
             - compression_duration_f
             - compression_energy_f
           ticks_orderflow_blocks:
             - vpin_f

    2. Legacy format (flat list):
       requested_features:
         - atr_f
         - trend_r2_20_f
       (all are treated as required)
    """
    if not requested_features:
        return None

    # Check if it's the new structured format
    if isinstance(requested_features, dict):
        required_features = requested_features.get("required") or []
        optional_blocks_dict = requested_features.get("optional_blocks") or {}

        if not required_features and not optional_blocks_dict:
            return None

        # Load feature_dependencies if not provided
        if feature_deps is None:
            feature_deps = _load_feature_dependencies()

        # Derive minimal_required_cols from required features
        minimal_required_cols = []
        for feat_name in required_features:
            if isinstance(feat_name, str):
                cols = _feature_name_to_output_columns(feat_name, feature_deps)
                minimal_required_cols.extend(cols)

        # Add basic OHLCV fields (always required)
        basic_cols = ["open", "high", "low", "close", "volume", "atr"]
        for col in basic_cols:
            if col not in minimal_required_cols:
                minimal_required_cols.append(col)

        # Derive optional_blocks from optional_blocks_dict
        # Convert feature names to column patterns
        optional_blocks: Dict[str, List[str]] = {}
        for block_name, feature_list in optional_blocks_dict.items():
            if isinstance(block_name, str) and isinstance(feature_list, list):
                patterns = []
                for feat_name in feature_list:
                    if isinstance(feat_name, str):
                        cols = _feature_name_to_output_columns(feat_name, feature_deps)
                        for col in cols:
                            patterns.append(col)
                            # Also add wildcard pattern for related columns
                            patterns.append(f"*{col}*")
                # De-dup while preserving order
                seen = set()
                patterns = [p for p in patterns if not (p in seen or seen.add(p))]
                if patterns:
                    optional_blocks[block_name] = patterns

        missingness_policy = feature_pipeline.get("missingness_policy") or {
            "optional_blocks_on_missing": "skip",
            "append_block_mask": True,
            "block_dropout_p": 0.05,
        }

        return FeatureContract(
            minimal_required_cols=minimal_required_cols,
            optional_blocks=optional_blocks,
            missingness_policy=missingness_policy,
        )

    # Legacy format: flat list, all required
    return None


def load_feature_contract(config_dir: str | Path) -> Optional[FeatureContract]:
    """
    Load nnmultihead feature contract from <config_dir>/features.yaml.

    The feature_contract is now always embedded in features.yaml (merged format).
    It is materialized by materialize_nnmh_config_from_task_spec() from:
      - feature_plan.yaml (source of truth for feature_contract definition)
      - TaskSpec tiers and optional_blocks_enabled (for minimal_required_cols and optional_blocks)

    Returns None if features.yaml does not contain a feature_contract section.

    Note: Legacy standalone feature_contract.yaml is no longer supported.
    All feature contracts should be defined in feature_plan.yaml and materialized into features.yaml.
    """
    config_dir = Path(config_dir)

    # Load from features.yaml (merged format, materialized from feature_plan.yaml)
    features_path = config_dir / "features.yaml"
    if features_path.exists():
        obj = yaml.safe_load(features_path.read_text(encoding="utf-8")) or {}

        # Prefer explicit feature_contract section if present.
        # Rationale:
        # - Some features have multiple possible output column sets depending on implementation/caching.
        # - For nnmultihead, the author-maintained contract should be the source of truth for
        #   "hard required columns", while derived contracts are useful as a fallback.
        feature_pipeline = obj.get("feature_pipeline") or {}
        fc = obj.get("feature_contract") if isinstance(obj, dict) else None
        if isinstance(fc, dict):
            minimal = fc.get("minimal_required_cols") or []
            optional_blocks = fc.get("optional_blocks") or []
            missingness_policy = fc.get("missingness_policy") or {}
            if not isinstance(minimal, list):
                minimal = []
            if not isinstance(optional_blocks, (list, dict)):
                optional_blocks = []
            if not isinstance(missingness_policy, dict):
                missingness_policy = {}
            return FeatureContract(
                minimal_required_cols=[str(x) for x in minimal if str(x).strip()],
                optional_blocks=optional_blocks,
                missingness_policy=missingness_policy,
            )

        # Fallback: derive from requested_features structure (new format).
        requested_features = feature_pipeline.get("requested_features")
        if requested_features and isinstance(requested_features, dict):
            # Load feature_dependencies for accurate output_columns mapping
            feature_deps = _load_feature_dependencies()
            contract = _derive_contract_from_requested_features(
                requested_features, feature_pipeline, feature_deps
            )
            if contract is not None:
                return contract

    return None


def validate_minimal_required_cols(
    available_columns: List[str], *, contract: FeatureContract
) -> None:
    """
    Fail-fast validation: ensure minimal_required_cols exist.

    Note: we only enforce columns explicitly listed in the contract.
    """
    have = set(str(c) for c in (available_columns or []))
    missing = [c for c in (contract.minimal_required_cols or []) if c not in have]
    if missing:
        raise ValueError(
            "FeatureContract minimal_required_cols missing from feature dataframe: "
            f"{missing}. Available columns include: {sorted(list(have))[:30]} ..."
        )
