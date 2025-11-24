"""Deprecated: legacy multi-asset preprocessing has been removed.

Use the config-driven pipeline (`scripts/train_strategy.py`) with per-strategy
configurations instead of this module.
"""

raise RuntimeError(
    "safe_multi_asset_preprocessing.py is deprecated. "
    "Use scripts/train_strategy.py with config/strategies/*."
)
