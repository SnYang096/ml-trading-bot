"""
波动率模型配置加载器和特征选择工具
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import pandas as pd
import numpy as np


def load_volatility_model_config(
    config_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    加载波动率模型配置文件

    Args:
        config_path: 配置文件路径，如果为None，使用默认路径

    Returns:
        配置字典
    """
    if config_path is None:
        # 默认路径：项目根目录下的 config/volatility_model.yaml
        project_root = Path(__file__).resolve().parents[4]
        config_path = project_root / "config" / "volatility_model.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Volatility model config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def create_vpin_volatility_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    动态创建VPIN volatility衍生特征（如果不存在）

    根据参考文档，创建以下特征：
    - vpin_vol_ratio: vpin_volatility_10 / vpin_volatility_20
    - vpin_vol_zscore: VPIN volatility的Z-score
    - vpin_spike: VPIN volatility spike flag

    Args:
        df: 包含VPIN特征的DataFrame
        config: 波动率模型配置

    Returns:
        添加了衍生特征的DataFrame（如果原始特征存在）
    """
    if config is None:
        config = load_volatility_model_config()

    feature_engineering = config.get("feature_engineering", {})
    df = df.copy()

    # 检查基础VPIN volatility特征是否存在
    vpin_vol_10_col = "vpin_volatility_10"
    vpin_vol_20_col = "vpin_volatility_20"

    has_vpin_vol_10 = vpin_vol_10_col in df.columns
    has_vpin_vol_20 = vpin_vol_20_col in df.columns

    if not (has_vpin_vol_10 and has_vpin_vol_20):
        # 如果没有基础特征，无法创建衍生特征
        return df

    # 1. 创建 vpin_vol_ratio
    if feature_engineering.get("create_vpin_vol_ratio", True):
        ratio_params = feature_engineering.get("vpin_vol_ratio_params", {})
        epsilon = float(ratio_params.get("epsilon", 1e-8))

        if "vpin_vol_ratio" not in df.columns:
            df["vpin_vol_ratio"] = df[vpin_vol_10_col] / (df[vpin_vol_20_col] + epsilon)

    # 2. 创建 vpin_vol_zscore 和 vpin_spike
    if feature_engineering.get("create_vpin_vol_zscore", True):
        zscore_params = feature_engineering.get("vpin_vol_zscore_params", {})
        window = int(zscore_params.get("window", 50))
        threshold = float(zscore_params.get("threshold", 2.0))
        epsilon = (
            float(ratio_params.get("epsilon", 1e-8))
            if "ratio_params" in locals()
            else 1e-8
        )

        if "vpin_vol_zscore" not in df.columns:
            rolling_mean = (
                df[vpin_vol_10_col].rolling(window=window, min_periods=1).mean()
            )
            rolling_std = (
                df[vpin_vol_10_col].rolling(window=window, min_periods=1).std()
            )
            df["vpin_vol_zscore"] = (df[vpin_vol_10_col] - rolling_mean) / (
                rolling_std + epsilon
            )

        if "vpin_spike" not in df.columns:
            df["vpin_spike"] = (df["vpin_vol_zscore"] > threshold).astype(int)

    return df


def get_volatility_model_params(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    获取波动率模型的训练参数

    Args:
        config: 波动率模型配置

    Returns:
        模型参数字典
    """
    if config is None:
        config = load_volatility_model_config()

    trainer_config = config.get("trainer", {})
    model_params = trainer_config.get("model_params", {})

    return model_params.copy()


def get_categorical_features(
    X: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[List[str]]:
    """
    获取分类特征列表

    Args:
        X: 特征DataFrame
        config: 波动率模型配置

    Returns:
        分类特征列表，如果不存在则返回None
    """
    if config is None:
        config = load_volatility_model_config()

    feature_selection = config.get("feature_selection", {})
    categorical_features = feature_selection.get("categorical_features", [])

    # 检查特征是否存在且有多值
    available_categorical = []
    for cat_col in categorical_features:
        if cat_col in X.columns and X[cat_col].nunique() > 1:
            available_categorical.append(cat_col)

    return available_categorical if available_categorical else None


def prepare_volatility_model_data(
    X: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    feature_loader: Optional[Any] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[List[str]]]:
    """
    准备波动率模型训练数据：确保必需特征存在 + 特征选择 + 特征工程

    Args:
        X: 原始特征DataFrame（可能不包含所有波动率模型需要的特征）
        config: 波动率模型配置
        feature_loader: 特征加载器（可选，用于计算缺失的特征）

    Returns:
        (处理后的特征DataFrame, 选中的特征列表, 分类特征列表)
    """
    if config is None:
        config = load_volatility_model_config()

    X_processed = X.copy()
    feature_config = config.get("volatility_features", {})
    feature_groups = feature_config.get("groups", [])
    selected_columns: List[str] = []

    def _compute_feature(feature_name: str) -> None:
        nonlocal X_processed
        if not feature_loader or not feature_name:
            return
        try:
            X_processed = feature_loader.load_features_from_requested(
                X_processed,
                requested_features=[feature_name],
                fit=True,
            )
            print(f"   ✅ Computed missing feature: {feature_name}")
        except Exception as exc:
            print(f"   ⚠️ Failed to compute feature '{feature_name}': {exc}")

    for group in feature_groups:
        feature_name = group.get("feature_name")
        required = bool(group.get("required", False))
        columns = group.get("columns", [])
        if not columns:
            continue

        missing_cols = [col for col in columns if col not in X_processed.columns]
        if missing_cols and feature_name:
            _compute_feature(feature_name)
            missing_cols = [col for col in columns if col not in X_processed.columns]

        if missing_cols and required:
            print(
                f"   ⚠️ Required feature group '{group.get('name')}' missing columns: {missing_cols}"
            )

        existing_cols = [col for col in columns if col in X_processed.columns]
        if existing_cols:
            selected_columns.extend(existing_cols)

    # 创建VPIN衍生特征
    X_processed = create_vpin_volatility_features(X_processed, config)

    # 去重并保持顺序
    seen = set()
    ordered_columns = []
    for col in selected_columns:
        if col in X_processed.columns and col not in seen:
            ordered_columns.append(col)
            seen.add(col)

    if not ordered_columns:
        ordered_columns = [
            col
            for col in X_processed.columns
            if col not in {"open", "high", "low", "close", "volume", "signal", "label"}
        ]

    categorical_features = get_categorical_features(
        X_processed[ordered_columns], config
    )

    X_vol = X_processed[ordered_columns].copy()

    return X_vol, ordered_columns, categorical_features
