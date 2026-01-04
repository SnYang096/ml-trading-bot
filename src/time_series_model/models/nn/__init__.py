"""
Neural network model family for time-series trading.

This package intentionally focuses on *modeling market path primitives* (dir/mfe/mae/t_to_mfe/...)
and keeps *strategy/policy* in Router/Backtest layers.
"""

from .path_primitives_labels import compute_path_primitives_labels
from .path_primitives_model import MultiHeadPathPrimitivesMLP, PathPrimitivesModelConfig
from .path_primitives_trainer import train_path_primitives_mlp, TrainConfig
