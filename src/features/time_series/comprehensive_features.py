"""
向后兼容包装器：ComprehensiveFeatureEngineer -> ConfigFeatureEngineer

此文件已废弃，所有功能已迁移到基于配置文件的特征加载系统。
保留此文件仅用于向后兼容，新代码应直接使用 ConfigFeatureEngineer。
"""

import warnings
from typing import Optional, Set, Tuple, List
import pandas as pd

from src.features.loader.config_feature_engineer import ConfigFeatureEngineer


class ComprehensiveFeatureEngineer:
    """
    向后兼容包装器：将旧的 ComprehensiveFeatureEngineer 映射到 ConfigFeatureEngineer
    
    注意：此类的所有方法都会发出警告，建议迁移到 ConfigFeatureEngineer。
    """
    
    def __init__(
        self,
        feature_types: str = "comprehensive",
        scaler_type: str = "standard",
        **kwargs
    ):
        """
        初始化（向后兼容）
        
        Args:
            feature_types: 已废弃，将被忽略。请使用 ConfigFeatureEngineer(strategy_name=...)
            scaler_type: 已废弃，配置驱动的特征不使用外部 scaler
            **kwargs: 其他参数（已废弃）
        """
        warnings.warn(
            "ComprehensiveFeatureEngineer is deprecated. "
            "Please use ConfigFeatureEngineer(strategy_name='sr_reversal') or similar. "
            "See config/strategy_features.yaml for available strategies.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # 尝试从 feature_types 推断策略名称（向后兼容）
        # 如果 feature_types 包含策略名称，使用它；否则使用默认策略
        strategy_name = self._infer_strategy_from_feature_types(feature_types)
        
        self._engineer = ConfigFeatureEngineer(strategy_name=strategy_name)
        self.feature_types = feature_types  # 保留用于向后兼容
    
    def _infer_strategy_from_feature_types(self, feature_types: str) -> str:
        """从旧的 feature_types 推断策略名称"""
        # 默认策略映射（向后兼容）
        if "reversal" in feature_types.lower():
            return "sr_reversal"
        elif "breakout" in feature_types.lower() and "compression" not in feature_types.lower():
            return "sr_breakout"
        elif "compression" in feature_types.lower():
            return "compression_breakout"
        elif "trend" in feature_types.lower():
            return "trend_following"
        else:
            # 默认使用 sr_reversal
            return "sr_reversal"
    
    def engineer_all_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        required_features: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """向后兼容的方法"""
        return self._engineer.engineer_all_features(df, fit=fit, required_features=required_features)
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """向后兼容的方法"""
        return self._engineer.engineer_features(df, fit=fit)
    
    def get_feature_columns(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """向后兼容的方法"""
        return self._engineer.get_feature_columns(df)
    
    def save_scalers(self, path: str) -> None:
        """向后兼容的方法（no-op）"""
        self._engineer.save_scalers(path)
    
    def load_scalers(self, path: str) -> None:
        """向后兼容的方法（no-op）"""
        self._engineer.load_scalers(path)


# 向后兼容的函数
def create_comprehensive_feature_engineer(
    feature_types: str = "comprehensive",
    scaler_type: str = "standard",
    **kwargs
) -> ComprehensiveFeatureEngineer:
    """向后兼容的工厂函数"""
    return ComprehensiveFeatureEngineer(
        feature_types=feature_types,
        scaler_type=scaler_type,
        **kwargs
    )


def engineer_features_by_type(
    df: pd.DataFrame,
    feature_types: str = "baseline",
    feature_engineer: Optional[ComprehensiveFeatureEngineer] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, ComprehensiveFeatureEngineer]:
    """向后兼容的函数"""
    if feature_engineer is None:
        feature_engineer = ComprehensiveFeatureEngineer(feature_types=feature_types)
    
    engineered_df = feature_engineer.engineer_all_features(df, fit=fit)
    return engineered_df, feature_engineer


def get_feature_columns_by_type(
    df: pd.DataFrame,
    feature_types: str = "baseline"
) -> List[str]:
    """向后兼容的函数"""
    engineer = ComprehensiveFeatureEngineer(feature_types=feature_types)
    return engineer.get_feature_columns(df)


def engineer_features(
    df: pd.DataFrame,
    feature_engineer: Optional[ComprehensiveFeatureEngineer] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, ComprehensiveFeatureEngineer]:
    """向后兼容的函数"""
    if feature_engineer is None:
        feature_engineer = ComprehensiveFeatureEngineer()
    
    engineered_df = feature_engineer.engineer_features(df, fit=fit)
    return engineered_df, feature_engineer


def add_dl_time_series_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """向后兼容的函数（需要从 dl_sequence_features 导入）"""
    from src.features.time_series.dl_sequence_features import add_dl_sequence_features
    return add_dl_sequence_features(df, **kwargs)
