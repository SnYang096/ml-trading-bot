"""Build OrderFlowListener primary + extra ``IncrementalFeatureComputer`` graphs for trend/fat-tail live.

Primary FC always uses ``strategies/bpc/archetypes`` at ``tf_bpc``. Any other **PCM-registered**
strategy whose ``meta.yaml`` timeframe equals ``tf_bpc`` has its feature columns merged into the
primary FC (no separate FC). Remaining registered strategies are grouped by timeframe; each
group gets one FC merging all archetype dirs in that group. FER columns are merged into primary
and into any group that includes the ME disk package.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

from src.config.strategy_layout import resolve_strategy_package_under_root
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.live_feature_plan import extract_features_from_archetypes

logger = logging.getLogger(__name__)


def _merge_archetype_plans(archetypes_dirs: List[str]) -> Tuple[Set[str], List[str]]:
    merged: Set[str] = set()
    nodes: List[str] = []
    for d in archetypes_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            s, n = extract_features_from_archetypes(d)
            merged |= set(s)
            nodes.extend(n)
        except Exception as exc:
            logger.warning("feature extract failed for %s: %s", d, exc)
    return merged, sorted(set(nodes))


def make_primary_feature_computer_factory(
    *,
    strategies_root: str,
    tf_bpc: str,
    bar_minutes_bpc: int,
    bpc_archetypes_dir: str,
    fer_feat: Set[str],
    fer_nodes: List[str],
    same_tf_other_dirs: List[str],
) -> Callable[[str], IncrementalFeatureComputer]:
    """Factory for the primary (BPC clock) feature computer including same-TF merges."""
    same_s, same_n = _merge_archetype_plans(same_tf_other_dirs)

    def _factory(_symbol: str) -> IncrementalFeatureComputer:
        fc = IncrementalFeatureComputer(
            tick_window_minutes=bar_minutes_bpc,
            bar_window_size=bar_minutes_bpc * 2,
            archetypes_dir=bpc_archetypes_dir,
            primary_timeframe=tf_bpc,
        )
        if fer_feat:
            fc.live_feature_set |= fer_feat
            fc.live_feature_nodes = sorted(
                set(fc.live_feature_nodes) | set(fer_nodes)
            )
        if same_s:
            fc.live_feature_set |= same_s
            fc.live_feature_nodes = sorted(
                set(fc.live_feature_nodes) | set(same_n)
            )
        return fc

    return _factory


def build_extra_feature_computers_for_symbol(
    *,
    strategies_root: str,
    registry_tf_map: Dict[str, str],
    tf_bpc: str,
    fer_feat: Set[str],
    fer_nodes: List[str],
) -> Dict[str, IncrementalFeatureComputer]:
    """One FC per distinct non-BPC timeframe, merging all registered strategies on that clock."""
    extras: Dict[str, IncrementalFeatureComputer] = {}
    # tf -> list of (registry_key, archetypes_dir)
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for rk, tf in registry_tf_map.items():
        if rk == "bpc" or tf == tf_bpc:
            continue
        pkg = resolve_strategy_package_under_root(
            Path(strategies_root), rk, allow_bad_candidates=False
        )
        adir = str(pkg / "archetypes")
        if not os.path.isdir(adir):
            logger.warning("extra FC: skip %s (no %s)", rk, adir)
            continue
        groups.setdefault(tf, []).append((rk, adir))

    for tf, members in groups.items():
        dirs = [m[1] for m in members]
        merged_s, merged_n = _merge_archetype_plans(dirs)
        bm = bar_minutes_from_tf(tf)
        base_dir = dirs[0]
        fc = IncrementalFeatureComputer(
            tick_window_minutes=bm,
            bar_window_size=bm * 2,
            archetypes_dir=base_dir,
            primary_timeframe=tf,
        )
        if merged_s:
            fc.live_feature_set |= merged_s
            fc.live_feature_nodes = sorted(
                set(fc.live_feature_nodes) | set(merged_n)
            )
        if fer_feat and any(m[0] == "me" for m in members):
            fc.live_feature_set |= fer_feat
            fc.live_feature_nodes = sorted(
                set(fc.live_feature_nodes) | set(fer_nodes)
            )
        extras[tf] = fc
    return extras


def bar_minutes_from_tf(tf: str) -> int:
    """Parse ``240T`` / ``15min`` / ``2h``-style timeframes to integer minutes."""
    t = str(tf).strip().lower()
    if t.endswith("t"):
        try:
            return int(t[:-1])
        except ValueError:
            return 60
    if t.endswith("min"):
        try:
            return int(t[:-3])
        except ValueError:
            return 60
    if t.endswith("h"):
        try:
            return int(float(t[:-1]) * 60)
        except ValueError:
            return 60
    return 60
