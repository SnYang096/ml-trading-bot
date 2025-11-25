"""Deprecated monolithic training entrypoint.

Replaced by scripts/train_strategy_pipeline.py with per-strategy configuration.
"""

raise RuntimeError(
    "time_series_model.pipeline.training.train is deprecated. "
    "Use scripts/train_strategy_pipeline.py with config/strategies/*."
)
