"""
ME Live Strategy — 纯逻辑决策引擎

ME = MomentumExpansion (动量扩张)

决策管线:
  1. ME expansion detection → 方向 (+1/-1/0)
  2. Gate (hard deny + soft weight) → 结构性否决
  3. Entry Filter → 入场时机
  4. Evidence Score → Tier 选择
  5. Tier → 执行参数 (SL/TP/size/timeout)

参考 BPCLiveStrategy 实现。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.archetype.loader import (
    StrategyArchetype,
    load_strategy_archetype,
)
from src.time_series_model.execution.entry_filter import (
    DerivedEntryFeatureState,
    check_entry_filters_or_single,
    load_entry_filters_config,
)
from src.time_series_model.live.execution_profile_apply import pick_atr

logger = logging.getLogger(__name__)


# ================================================================
# Tier selection helper
# ================================================================


def select_tier(
    evidence_score: float,
    tiers_cfg: Dict[str, Any],
    exec_config: Dict[str, Any],
) -> Dict[str, Any]:
    """根据 evidence_score 选择 tier，返回执行参数。"""
    levels = tiers_cfg.get("levels", [])
    levels = sorted(levels, key=lambda x: x.get("evidence_min", 0), reverse=True)

    for level in levels:
        if evidence_score >= level.get("evidence_min", 0):
            sl_cfg = level.get("stop_loss", {})
            trail_cfg = sl_cfg.get("trailing", {})
            return {
                "tier_name": level.get("name", "unknown"),
                "initial_r": sl_cfg.get("initial_r", 2.0),
                "activation_r": trail_cfg.get("activation_r", 1.0),
                "trail_r": trail_cfg.get("trail_r", 1.5),
                "time_stop_bars": level.get("time_stop_bars", 50),
                "size_multiplier": level.get("size_multiplier", 1.0),
            }

    # 未匹配任何 tier → 使用全局默认参数
    global_sl = exec_config.get("stop_loss", {})
    global_trail = global_sl.get("trailing", {})
    return {
        "tier_name": "default",
        "initial_r": global_sl.get("initial_r", 2.0),
        "activation_r": global_trail.get("activation_r", 1.0),
        "trail_r": global_trail.get("trail_r", 1.5),
        "time_stop_bars": exec_config.get("holding", {}).get("time_stop_bars", 50),
        "size_multiplier": 1.0,
    }


# ================================================================
# MELiveStrategy — pure-logic decision engine
# ================================================================


class MELiveStrategy:
    """
    ME 实盘决策引擎 — 纯逻辑，不依赖 Nautilus

    接口与 OrderFlowListener 对齐:
        decide(features, symbol) → List[TradeIntent]

    可直接赋给 OrderFlowListener.decision_handler 使用。
    """

    def __init__(
        self,
        strategies_root: str = "config/strategies",
        holding_yaml_path: Optional[str] = None,
        trade_size: float = 1.0,
        primary_timeframe: str = "240T",
        bar_minutes: int = 240,
    ):
        self.strategies_root = strategies_root
        self.trade_size = trade_size
        self._primary_timeframe = primary_timeframe
        self._bar_minutes = bar_minutes
        self._holding_yaml_path = holding_yaml_path

        # 配置 — 在 load_configs() 中初始化
        self._archetype: Optional[StrategyArchetype] = None
        self._entry_cfg: Dict[str, Any] = {}
        self._exec_config: Dict[str, Any] = {}
        self._tiers_cfg: Dict[str, Any] = {}
        self._holding_cfg: Dict[str, Any] = {}
        self._ef_state: Optional[DerivedEntryFeatureState] = None

        # Evidence 分位数查找表（从历史数据计算）
        self._quantiles: Dict[str, Dict[str, float]] = {}

        # 当前决策缓存
        self._last_tier_params: Optional[Dict[str, Any]] = None

    # ────────────────────────────────────────────────────
    # 配置加载
    # ────────────────────────────────────────────────────

    def load_configs(self) -> None:
        """加载所有 ME 配置（archetype / entry_filter / execution / holding）"""
        try:
            # 1. StrategyArchetype（Gate + Evidence + Execution）
            self._archetype = load_strategy_archetype(
                "me", strategies_root=self.strategies_root
            )
            logger.info(
                "✅ ME Archetype loaded: "
                f"{len(self._archetype.gate.all_rules)} gate rules, "
                f"{len(self._archetype.evidence.features)} evidence features"
            )

            # 2. Entry Filters
            self._entry_cfg = load_entry_filters_config(
                "me", strategies_root=self.strategies_root
            )
            enabled = [
                f["id"]
                for f in self._entry_cfg.get("filters", [])
                if f.get("enabled", False)
            ]
            logger.info(
                f"✅ Entry Filters loaded: {len(enabled)} enabled ({', '.join(enabled)})"
            )

            # 3. Execution 配置（tiers + 全局参数）
            exec_path = (
                Path(self.strategies_root) / "me" / "archetypes" / "execution.yaml"
            )
            if exec_path.exists():
                with open(exec_path, "r", encoding="utf-8") as f:
                    self._exec_config = yaml.safe_load(f) or {}
            self._tiers_cfg = self._exec_config.get("tiers", {})
            tiers_enabled = self._tiers_cfg.get("enabled", False)
            n_tiers = len(self._tiers_cfg.get("levels", []))
            logger.info(
                f"✅ Execution config: tiers={'ON' if tiers_enabled else 'OFF'} ({n_tiers} levels)"
            )

            # 4. Holding 配置
            holding_path = self._holding_yaml_path
            if holding_path is None:
                candidates = [
                    "config/strategies/me/archetypes/holding.yaml",
                ]
                for c in candidates:
                    if Path(c).exists():
                        holding_path = c
                        break
            if holding_path and Path(holding_path).exists():
                with open(holding_path, "r", encoding="utf-8") as f:
                    self._holding_cfg = yaml.safe_load(f) or {}
                logger.info("✅ Holding config loaded")

            # 5. Entry Filter 派生特征状态
            self._ef_state = DerivedEntryFeatureState()
            logger.info("✅ Entry filter state initialized")

        except Exception as e:
            logger.error(f"❌ Failed to load ME configs: {e}")
            raise

    def set_quantiles(self, quantiles: Dict[str, Dict[str, float]]) -> None:
        """设置 Evidence 特征的分位数查找表（从外部传入）"""
        self._quantiles = quantiles
        logger.info(f"✅ Quantiles set: {len(quantiles)} features")

    def set_quantiles_from_df(self, df: pd.DataFrame) -> None:
        """从 DataFrame 计算并设置 quantiles"""
        if self._archetype is None:
            raise RuntimeError(
                "Must call load_configs() before set_quantiles_from_df()"
            )

        quantiles = {}
        for feat_cfg in self._archetype.evidence.features:
            feat_name = feat_cfg["name"]
            if feat_name in df.columns:
                bins = feat_cfg.get("bins", [0.25, 0.45, 0.65, 0.85])
                quantiles[feat_name] = {
                    f"q{int(b*100)}": df[feat_name].quantile(b) for b in bins
                }
        self._quantiles = quantiles
        logger.info(f"✅ Quantiles computed from df: {len(quantiles)} features")

    # ────────────────────────────────────────────────────
    # 决策主流程
    # ────────────────────────────────────────────────────

    def decide(
        self,
        features: Dict[str, Any],
        symbol: str,
        bars: Optional[pd.DataFrame] = None,
    ) -> List[TradeIntent]:
        """
        ME 决策主流程

        Args:
            features: 特征字典（来自 IncrementalFeatureComputer）
            symbol: 交易对
            bars: 历史 bars (可选，用于派生特征)

        Returns:
            List[TradeIntent]: 0 或 1 个交易意图
        """
        if self._archetype is None:
            logger.warning("⚠️ ME archetype not loaded, call load_configs() first")
            return []

        # 1. 检测扩张信号 + 方向
        expansion_signal, direction = self._detect_me_expansion(features)
        if not expansion_signal:
            return []

        # 2. Gate 检查
        gate_passed, gate_reasons, gate_weight = self._archetype.apply_gate(features)
        if not gate_passed:
            logger.debug(f"ME Gate deny: {symbol} | {', '.join(gate_reasons)}")
            return []

        # 3. Entry Filter 检查
        ef_passed = self._check_entry_filter(features, bars)
        if not ef_passed:
            logger.debug(f"ME Entry filter reject: {symbol}")
            return []

        # 4. Evidence Score
        evidence_score = self._compute_evidence_score(features)

        # 5. Tier 选择
        tier_params = select_tier(evidence_score, self._tiers_cfg, self._exec_config)
        self._last_tier_params = tier_params

        # 6. ATR
        atr = pick_atr(features, self._bar_minutes)
        if atr is None or atr <= 0:
            logger.warning(f"⚠️ Invalid ATR for {symbol}: {atr}")
            return []

        # 7. 构造 TradeIntent
        intent = TradeIntent(
            symbol=symbol,
            side="BUY" if direction > 0 else "SELL",
            quantity=self.trade_size * tier_params["size_multiplier"],
            archetype="ME",
            confidence=evidence_score,
            initial_stop_loss_r=tier_params["initial_r"],
            trailing_activation_r=tier_params["activation_r"],
            trailing_stop_r=tier_params["trail_r"],
            time_stop_bars=tier_params["time_stop_bars"],
            atr=atr,
            reason=f"ME expansion | tier={tier_params['tier_name']} | evidence={evidence_score:.2f}",
            metadata={
                "gate_weight": gate_weight,
                "gate_reasons": gate_reasons,
                "direction": direction,
            },
        )

        logger.info(
            f"✅ ME signal: {symbol} {intent.side} | "
            f"tier={tier_params['tier_name']} | evidence={evidence_score:.2f}"
        )

        return [intent]

    # ────────────────────────────────────────────────────
    # 辅助方法
    # ────────────────────────────────────────────────────

    def _detect_me_expansion(self, features: Dict[str, Any]) -> tuple[bool, int]:
        """
        检测 ME 扩张信号 + 方向

        Returns:
            (expansion_signal, direction): (True/False, +1/-1/0)
        """
        # ME 核心逻辑：ATR 扩张 + 放量
        atr_pct = features.get("atr_percentile", 0)
        volume_ratio = features.get("volume_ratio_pct", 0)

        # 扩张条件：ATR 百分位 > 60 且成交量放大
        expansion = atr_pct > 0.6 and volume_ratio > 0.6

        if not expansion:
            return False, 0

        # 方向判断：使用 breakout_sign
        # 简化实现：使用价格方向一致性
        dir_consistency = features.get("price_dir_consistency_pct", 0.5)
        if dir_consistency > 0.6:
            direction = 1  # 上涨突破
        elif dir_consistency < 0.4:
            direction = -1  # 下跌突破
        else:
            direction = 0  # 无明确方向

        return True, direction

    def _check_entry_filter(
        self, features: Dict[str, Any], bars: Optional[pd.DataFrame]
    ) -> bool:
        """检查 Entry Filter（OR 逻辑）"""
        if not self._entry_cfg.get("filters"):
            return True  # 无 filter 配置，默认通过

        # 更新派生特征状态
        if bars is not None and len(bars) > 0:
            self._ef_state.update(bars.iloc[-1].to_dict())

        # OR 逻辑：任一 filter 通过即可
        passed = check_entry_filters_or_single(
            features, self._entry_cfg, self._ef_state
        )
        return passed

    def _compute_evidence_score(self, features: Dict[str, Any]) -> float:
        """计算 Evidence Score"""
        if self._archetype is None or not self._archetype.evidence.features:
            return 0.5  # 默认中性

        scores = []
        for feat_cfg in self._archetype.evidence.features:
            feat_name = feat_cfg["name"]
            feat_val = features.get(feat_name)
            if feat_val is None:
                continue

            # 分位数映射
            bins = feat_cfg.get("bins", [0.25, 0.45, 0.65, 0.85])
            quantiles = self._quantiles.get(feat_name, {})
            if not quantiles:
                continue

            # 映射到 [0, 1]
            if feat_val < quantiles.get("q25", float("-inf")):
                score = 0.0
            elif feat_val < quantiles.get("q45", float("-inf")):
                score = 0.25
            elif feat_val < quantiles.get("q65", float("-inf")):
                score = 0.5
            elif feat_val < quantiles.get("q85", float("-inf")):
                score = 0.75
            else:
                score = 1.0

            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.5
