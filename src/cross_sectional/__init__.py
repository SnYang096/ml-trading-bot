"""
Cross-sectional modelling toolkit.

This package provides utilities to:
    1. Assemble aligned multi-asset panels.
    2. Apply cross-sectional preprocessing (winsorization, z-score, ranking, neutralization).
    3. Train linear (Fama-MacBeth) or boosting-based cross-sectional models.

Typical usage::

    from cross_sectional import (
        FactorPanelBuilder,
        cross_sectional_zscore,
        CrossSectionalBoostingModel,
    )
"""

from .panel import FactorPanelBuilder, PanelConfig
from .processing import (
    cross_sectional_rank,
    cross_sectional_zscore,
    neutralize_against,
    winsorize_by_sigma,
)
from .crypto_factors import CryptoCSFactorConfig, add_crypto_cross_sectional_factors
from .panel_generation import PanelGenerationConfig, generate_cross_sectional_panel
from .model import CrossSectionalRegressor, FactorBacktestResult
from .boosting import CrossSectionalBoostingModel, BoostingEvalResult
from .report import ReportContext, generate_markdown_report, write_report
from .factor_catalog import categorize_columns
from .factor_selection import compute_cross_sectional_ic, apply_factor_selection

__all__ = [
    "FactorPanelBuilder",
    "PanelConfig",
    "cross_sectional_rank",
    "cross_sectional_zscore",
    "winsorize_by_sigma",
    "neutralize_against",
    "CrossSectionalRegressor",
    "FactorBacktestResult",
    "CrossSectionalBoostingModel",
    "BoostingEvalResult",
    "ReportContext",
    "generate_markdown_report",
    "write_report",
    "CryptoCSFactorConfig",
    "add_crypto_cross_sectional_factors",
    "PanelGenerationConfig",
    "generate_cross_sectional_panel",
    "categorize_columns",
    "compute_cross_sectional_ic",
    "apply_factor_selection",
]
