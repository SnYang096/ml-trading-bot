"""
Backward compatibility shim for the former `ml_trading` package.

The codebase was reorganised so the time-series model now lives under
`time_series_model`. Both internal modules and external tooling may still
reference `ml_trading`, so we transparently alias those imports here.
"""

from importlib import import_module
import sys

_module = import_module("time_series_model")

# Mirror module attributes for direct access (e.g., `time_series_model.__version__`)
globals().update(_module.__dict__)

# Ensure subsequent imports resolve to the same module object.
sys.modules[__name__] = _module
"""ML Trading Project - Machine Learning Algorithmic Trading System."""

__version__ = "0.0.2"
__author__ = "Your Name"
