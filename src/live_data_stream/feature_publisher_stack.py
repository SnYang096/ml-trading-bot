"""Build MultiSymbolManager for quant-feature-bus (ticks → features → disk).

Uses the same ``resource_allocation.enabled_archetypes`` as trend/fat-tail live, plus
``MLBOT_PUBLISHER_PRIMARY_ARCHETYPE`` (default ``bpc``) to pick the primary
``IncrementalFeatureComputer`` timeframe. Extra timeframes are merged per
``meta.yaml`` timeframe key (multiple strategies on the same bar clock share one FC).
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config.strategy_layout import resolve_strategy_package_under_root
from src.live_data_stream import MultiSymbolManager, StorageManager
from src.live_data_stream.gap_filler import GapFiller
from src.live_data_stream.constitution_config import (
    enabled_archetypes_from_constitution,
    load_constitution_dict,
    resolve_constitution_yaml,
)
from src.live_data_stream.feature_bus import FeatureBusWriter
from src.live_data_stream.strategy_runtime_config import (
    load_strategy_timeframe,
    me_enabled_in_allowlist,
)
from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.live.live_feature_plan import (
    extract_features_from_archetypes,
)

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

logger = logging.getLogger(__name__)


class _DataOnlyOrderManager:
    """Sentinel so the publisher never initializes consumer-side trading."""


class FeatureBusDecisionSink:
    """DecisionHandler-compatible sink that publishes every computed timeframe."""

    def __init__(
        self, writer: FeatureBusWriter, *, default_timeframe_key: str = "primary"
    ) -> None:
        self.writer = writer
        self.default_timeframe_key = (
            str(default_timeframe_key or "primary").strip() or "primary"
        )

    def decide(
        self,
        *,
        features: Dict[str, Any],
        symbol: str,
        bars: Any = None,
        features_by_timeframe: Dict[str, Dict[str, Any]] | None = None,
        decision_time: Any = None,
    ) -> list:
        ts = decision_time or features.get("timestamp") or pd.Timestamp.now(tz="UTC")
        by_tf = features_by_timeframe or {}
        if not by_tf:
            by_tf = {self.default_timeframe_key: dict(features)}
        for tf, feat in by_tf.items():
            feat = dict(feat)
            feat["_feature_timeframe"] = str(tf)
            feat["_feature_bus_published_at"] = pd.Timestamp.now(tz="UTC").isoformat()
            self.writer.append_features(
                symbol=symbol,
                timeframe=tf,
                features=feat,
                timestamp=ts,
            )
            if str(tf).upper() == "120T":
                self.writer.append_features(
                    symbol=symbol,
                    timeframe="2h",
                    features=feat,
                    timestamp=ts,
                )
        return []


def make_bar_write_callback(writer: FeatureBusWriter, symbol: str):
    def _callback(bar: Dict[str, Any]) -> None:
        try:
            writer.append_bar_1m(symbol, bar)
        except Exception:
            logger.exception("feature bus bar write failed: %s", symbol)

    return _callback


def timeframe_to_bar_minutes(tf: str) -> int:
    tf_norm = str(tf).strip().lower()
    if tf_norm.endswith("min"):
        return int(tf_norm[:-3])
    if tf_norm.endswith("t"):
        return int(tf_norm[:-1])
    if tf_norm.endswith("h"):
        return int(float(tf_norm[:-1]) * 60)
    return int(pd.Timedelta(tf_norm).total_seconds() // 60)


def _extract_plan(archetypes_dir: str) -> Tuple[set, List[str]]:
    if not os.path.isdir(archetypes_dir):
        return set(), []
    try:
        return extract_features_from_archetypes(archetypes_dir)
    except Exception as exc:
        logger.warning("feature extraction failed for %s: %s", archetypes_dir, exc)
        return set(), []


def _disk_package(archetype: str, strategies_root: str) -> Optional[str]:
    a = archetype.lower().strip()
    if a == "fer":
        return None
    slug = "me" if me_enabled_in_allowlist([a]) else a
    pkg = resolve_strategy_package_under_root(
        Path(strategies_root), slug, allow_bad_candidates=False
    )
    if (pkg / "archetypes").is_dir() or (pkg / "meta.yaml").is_file():
        return slug
    return None


@dataclass
class _ArcSpec:
    archetype: str
    disk: str
    timeframe: str
    archetypes_dir: str


def _collect_secondary_specs(
    *,
    enabled: List[str],
    primary_arch: str,
    strategies_root: str,
) -> List[_ArcSpec]:
    out: List[_ArcSpec] = []
    seen: set[str] = set()
    for raw in enabled:
        a = str(raw).lower().strip()
        if not a or a == "fer" or a == primary_arch:
            continue
        disk = _disk_package(a, strategies_root)
        if not disk or disk in seen:
            continue
        adir_pkg = resolve_strategy_package_under_root(
            Path(strategies_root), disk, allow_bad_candidates=False
        )
        adir = str(adir_pkg / "archetypes")
        if not os.path.isdir(adir):
            logger.warning(
                "publisher: skip archetype %s (no archetypes under %s)", a, adir
            )
            continue
        tf = load_strategy_timeframe(strategies_root, disk)
        out.append(_ArcSpec(archetype=a, disk=disk, timeframe=tf, archetypes_dir=adir))
        seen.add(disk)
    return out


def _pick_primary_archetype(
    enabled: List[str], strategies_root: str
) -> str:
    pref = os.getenv("MLBOT_PUBLISHER_PRIMARY_ARCHETYPE", "bpc").lower().strip()
    if pref in enabled:
        d = _disk_package(pref, strategies_root)
        if d and (
            resolve_strategy_package_under_root(
                Path(strategies_root), d, allow_bad_candidates=False
            )
            / "archetypes"
        ).is_dir():
            return pref
    for a in enabled:
        if a == "fer":
            continue
        d = _disk_package(a, strategies_root)
        if d and (
            resolve_strategy_package_under_root(
                Path(strategies_root), d, allow_bad_candidates=False
            )
            / "archetypes"
        ).is_dir():
            logger.info(
                "publisher: primary archetype fallback %s (wanted %s)", a, pref or "bpc"
            )
            return a
    raise FileNotFoundError(
        "No enabled archetype with a valid strategies/*/archetypes directory under "
        f"{strategies_root!r}; check resource_allocation.enabled_archetypes."
    )


def build_feature_bus_manager(
    args: Namespace, writer: FeatureBusWriter
) -> MultiSymbolManager:
    """Constitution-driven feature stack for ``run_market_feature_publisher``."""
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    strategies_root = str(args.strategies_root)
    co_override = getattr(args, "constitution_yaml", None)
    if isinstance(co_override, str) and not co_override.strip():
        co_override = None
    constitution_path = resolve_constitution_yaml(strategies_root, override=co_override)
    cfg = load_constitution_dict(constitution_path)
    enabled = enabled_archetypes_from_constitution(cfg)
    logger.info(
        "📋 quant-feature-bus enabled_archetypes=%s (constitution=%s)",
        enabled,
        constitution_path,
    )

    primary_arch = _pick_primary_archetype(enabled, strategies_root)
    primary_disk = _disk_package(primary_arch, strategies_root)
    assert primary_disk
    primary_pkg = resolve_strategy_package_under_root(
        Path(strategies_root), primary_disk, allow_bad_candidates=False
    )
    primary_dir = str(primary_pkg / "archetypes")
    tf_primary = load_strategy_timeframe(strategies_root, primary_disk)
    bm_primary = timeframe_to_bar_minutes(tf_primary)

    fer_dir = os.path.join(strategies_root, "fer", "archetypes")
    fer_set, fer_nodes = _extract_plan(fer_dir)

    primary_feat_extra: set = set()
    primary_nodes_extra: List[str] = []
    if os.path.isdir(fer_dir):
        primary_feat_extra |= fer_set
        primary_nodes_extra.extend(fer_nodes)

    def _primary_factory(_symbol: str) -> IncrementalFeatureComputer:
        fc = IncrementalFeatureComputer(
            tick_window_minutes=bm_primary,
            bar_window_size=bm_primary * 2,
            archetypes_dir=primary_dir,
            primary_timeframe=tf_primary,
        )
        if primary_feat_extra:
            fc.live_feature_set |= primary_feat_extra
            fc.live_feature_nodes = sorted(
                set(fc.live_feature_nodes) | set(primary_nodes_extra)
            )
        return fc

    secondaries = _collect_secondary_specs(
        enabled=enabled,
        primary_arch=primary_arch,
        strategies_root=strategies_root,
    )
    same_tf_as_primary: List[_ArcSpec] = []
    rest_secondaries: List[_ArcSpec] = []
    for spec in secondaries:
        if spec.timeframe == tf_primary:
            same_tf_as_primary.append(spec)
        else:
            rest_secondaries.append(spec)
    for spec in same_tf_as_primary:
        s_m, n_m = _extract_plan(spec.archetypes_dir)
        primary_feat_extra |= s_m
        primary_nodes_extra.extend(n_m)
    if primary_feat_extra:
        primary_nodes_extra = sorted(set(primary_nodes_extra))

    tf_groups: Dict[str, List[_ArcSpec]] = {}
    for spec in rest_secondaries:
        tf_groups.setdefault(spec.timeframe, []).append(spec)

    storage = StorageManager(args.live_storage_base)
    gap_filler: Optional[GapFiller] = None
    if CCXT_AVAILABLE:
        try:
            exchange = ccxt.binanceusdm({"enableRateLimit": True})
            gap_filler = GapFiller(storage_manager=storage, exchange=exchange)
            logger.info(
                "quant-feature-bus: GapFiller on (public USD-M ccxt; ticks/bars gap fill)"
            )
        except Exception as exc:
            logger.warning(
                "quant-feature-bus: GapFiller disabled (%s); warmup uses disk only",
                exc,
            )

    manager = MultiSymbolManager(
        symbols=symbols,
        storage_manager=storage,
        feature_computer_factory=_primary_factory,
        gap_filler=gap_filler,
        memory_window_hours=args.memory_window_hours,
        feature_compute_interval_minutes=args.feature_compute_interval_minutes,
        orderflow_window_minutes=args.orderflow_window_minutes,
        feature_4h_interval_hours=args.feature_4h_interval_hours,
        order_manager=_DataOnlyOrderManager(),
    )
    sink = FeatureBusDecisionSink(writer, default_timeframe_key=tf_primary)

    for symbol, listener in manager.listeners.items():
        listener.on_bar_callback = make_bar_write_callback(writer, symbol)
        listener.decision_handler = sink
        listener.order_manager = None
        extras: Dict[str, IncrementalFeatureComputer] = {}
        for tf_key, group in tf_groups.items():
            bm = timeframe_to_bar_minutes(tf_key)
            base = group[0].archetypes_dir
            fc = IncrementalFeatureComputer(
                tick_window_minutes=bm,
                bar_window_size=bm * 2,
                archetypes_dir=base,
                primary_timeframe=tf_key,
            )
            merged_s: set = set()
            merged_nodes: List[str] = []
            for g in group:
                s2, n2 = _extract_plan(g.archetypes_dir)
                merged_s |= s2
                merged_nodes.extend(n2)
            if merged_s:
                fc.live_feature_set |= merged_s
                fc.live_feature_nodes = sorted(
                    set(fc.live_feature_nodes) | set(merged_nodes)
                )
            if fer_set and any(g.disk == "me" for g in group):
                fc.live_feature_set |= fer_set
                fc.live_feature_nodes = sorted(
                    set(fc.live_feature_nodes) | set(fer_nodes)
                )
            extras[tf_key] = fc
        listener.extra_feature_computers = extras

    logger.info(
        "✅ quant-feature-bus stack: primary=%s/%s tf=%s extras=%s",
        primary_arch,
        primary_disk,
        tf_primary,
        list(tf_groups.keys()),
    )
    return manager
