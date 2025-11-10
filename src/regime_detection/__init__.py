"""
Rule-based regime detection module.

This package exposes the primary interface ``RuleBasedRegimeDetector`` along with
the configuration dataclass and enum-style labels used throughout the
trading system.
"""

from .config import RegimeDetectorConfig
from .detector import RegimeLabel, RuleBasedRegimeDetector, RegimeDetectionResult
from .hmm_smoother import RegimeHMMSmoother

__all__ = [
    "RegimeDetectorConfig",
    "RegimeLabel",
    "RegimeDetectionResult",
    "RuleBasedRegimeDetector",
    "RegimeHMMSmoother",
]


