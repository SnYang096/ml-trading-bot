"""
事件驱动策略（实盘）

基于 Nautilus Trader 的事件驱动架构，支持：
1. Tick 级特征计算（VPIN、订单流等）
2. Bar 级特征计算（技术指标、时间框架特征等）
3. 定时器触发信号融合和交易决策
4. 与训练流程统一的特征计算逻辑
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from collections import deque
import pandas as pd
import numpy as np

try:
    from nautilus_trader.model import Bar, QuoteTick, TradeTick
    from nautilus_trader.model import InstrumentId, BarType
    from nautilus_trader.model import OrderSide, MarketOrder
    from nautilus_trader.trading import Strategy
    from nautilus_trader.model.enums import AggressorSide

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    Strategy = object
    Bar = None
    TradeTick = None
    QuoteTick = None
    InstrumentId = None
    BarType = None
    OrderSide = None
    MarketOrder = None
    AggressorSide = None

from src.time_series_model.live.incremental_feature_computer import (
    IncrementalFeatureComputer,
)
from src.time_series_model.strategies.models import ModelArtifact
from src.time_series_model.core.constitution.constitution_executor import (
    ConstitutionExecutor,
)
from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
    load_evidence_quantiles,
)
from src.time_series_model.live.enforcement import enforce_before_order
from src.time_series_model.live.execution_manager import (
    ExecutionManager,
    GuardedOrderContext,
)
from src.time_series_model.nnmultihead.strategy_profile import (
    resolve_execution_profile,
    resolve_execution_profile_paths,
)
from src.time_series_model.live.live_runtime_paths import resolve_live_runtime_paths
from src.time_series_model.diagnostics.execution_log import (
    build_decision_id,
    build_stage_record,
    ExecutionStageLogWriter,
)


class EventDrivenStrategy(Strategy):
    """
    事件驱动策略

    特点：
    - 在 on_tick 中计算 tick 相关特征
    - 在 on_bar 中计算时间框架特征
    - 在定时器中融合信号并执行交易
    - 支持多时间框架特征融合
    """

    def __init__(
        self,
        strategy_name: str,
        instrument_id: InstrumentId,
        bar_types: Dict[str, BarType],  # {timeframe: BarType}
        trade_size: float,
        config_base_path: str = "config/strategies",
        model_path: Optional[str] = None,
        check_interval_minutes: int = 15,  # 信号检查间隔（分钟）
        min_order_interval_minutes: int = 15,  # 最小开仓间隔（分钟）
        vpin_bucket_volume_usd: Optional[float] = None,  # VPIN bucket volume (USD)
        constitution_yaml: Optional[str] = None,
    ):
        """
        Args:
            strategy_name: 策略名称
            instrument_id: 交易标的
            bar_types: 时间框架字典 {timeframe: BarType}
            trade_size: 交易规模
            config_base_path: 配置基础路径
            model_path: 模型路径
            check_interval_minutes: 信号检查间隔（分钟）
            min_order_interval_minutes: 最小开仓间隔（分钟）
            vpin_bucket_volume_usd: VPIN bucket volume (USD)
        """
        super().__init__()
        self.strategy_name = strategy_name
        self.instrument_id = instrument_id
        self.bar_types = bar_types
        self.trade_size = trade_size
        self.config_base_path = config_base_path
        self.model_path = model_path
        self.check_interval_minutes = check_interval_minutes
        self.min_order_interval_ns = min_order_interval_minutes * 60 * 1_000_000_000
        live_paths = resolve_live_runtime_paths()
        self.constitution_yaml = constitution_yaml or live_paths["constitution_yaml"]

        # 特征计算器
        self.feature_computer = IncrementalFeatureComputer(
            tick_window_minutes=30,
            bar_window_size=1000,
            vpin_bucket_volume_usd=vpin_bucket_volume_usd,
            vpin_n_buckets=50,
        )

        # 状态管理
        self.strategy_config = None
        self.model = None
        self.model_artifact: Optional[ModelArtifact] = (
            None  # 使用 ModelArtifact 统一管理
        )
        self.last_order_time_ns: Optional[int] = None
        self._constitution_executor: Optional[ConstitutionExecutor] = None
        self._constitution_runtime_state = None
        self._xm: Optional[ExecutionManager] = None
        self._exec_stage_writers: dict[str, ExecutionStageLogWriter] = {}

        # 时间框架特征缓存
        self.timeframe_features: Dict[str, Dict[str, float]] = {}

    def on_start(self) -> None:
        """策略启动"""
        self.log.info(f"🚀 Starting {self.strategy_name} strategy (event-driven)")

        try:
            # 1. Tree strategy configs are NOT supported in live (research-only).
            self.log.info(
                "ℹ️ Live does not load config/strategies/* (tree configs are research-only)."
            )
            log_dir = Path(os.getenv("MLBOT_EXECUTION_LOG_DIR", "results/live_logs"))
            for stage in [
                "features",
                "preds",
                "router",
                "gate",
                "evidence",
                "execution",
                "returns",
                "observability",
            ]:
                self._exec_stage_writers[stage] = ExecutionStageLogWriter(
                    base_dir=log_dir, stage=stage
                )

            # 2. 加载模型（优先使用 ModelArtifact）
            if self.model_path:
                model_dir = Path(self.model_path)
                # 检查是否是 ModelArtifact 目录（包含 model_artifact_metadata.json）
                if (model_dir / "model_artifact_metadata.json").exists():
                    self.model_artifact = ModelArtifact.load(model_dir)
                    self.model = self.model_artifact.model
                    self.log.info(f"✅ Loaded ModelArtifact from {self.model_path}")
                    self.log.info(
                        f"   Features: {len(self.model_artifact.used_features)}"
                    )
                else:
                    # 兼容旧格式：只加载 model.pkl
                    self.model = self._load_model(self.model_path)
                    self.log.info(
                        f"✅ Loaded model (legacy format) from {self.model_path}"
                    )
            else:
                # 尝试默认路径
                default_model_dir = Path("results") / self.strategy_name
                if (default_model_dir / "model_artifact_metadata.json").exists():
                    self.model_artifact = ModelArtifact.load(default_model_dir)
                    self.model = self.model_artifact.model
                    self.log.info(f"✅ Loaded ModelArtifact from {default_model_dir}")
                else:
                    default_model_path = (
                        Path("models") / self.strategy_name / "model.pkl"
                    )
                    if default_model_path.exists():
                        self.model = self._load_model(str(default_model_path))
                        self.log.info(
                            f"✅ Loaded model (legacy format) from {default_model_path}"
                        )
                    else:
                        self.log.warning("⚠️ No model found. Using rule-based signals.")

            # 2.5 Load constitution executor (for live safety / whitelist)
            try:
                if self.constitution_yaml and Path(self.constitution_yaml).exists():
                    self._constitution_executor = ConstitutionExecutor(
                        constitution_yaml=str(self.constitution_yaml)
                    )
                    self._constitution_runtime_state = (
                        self._constitution_executor.load_runtime_state()
                    )
                    self._xm = ExecutionManager(
                        strategy=self,
                        executor=self._constitution_executor,
                        runtime_state=self._constitution_runtime_state,
                    )
                    self.log.info(f"✅ Constitution loaded: {self.constitution_yaml}")
            except Exception as e:
                self.log.error(f"⚠️ Constitution init failed: {e}")

            # 3. 订阅市场数据
            for timeframe, bar_type in self.bar_types.items():
                self.subscribe_bars(bar_type)
                self.log.info(f"✅ Subscribed to {timeframe}: {bar_type}")

            # 4. 订阅 trade ticks（用于订单流特征）
            self.subscribe_trade_ticks(self.instrument_id)
            self.log.info(f"✅ Subscribed to trade ticks: {self.instrument_id}")

            # 5. 启动定时器
            self._schedule_next_check()

            self.log.info("✅ Strategy initialization complete")

        except Exception as e:
            self.log.error(f"❌ Error during strategy initialization: {e}")
            import traceback

            self.log.error(traceback.format_exc())
            raise

    def on_tick(self, tick: QuoteTick) -> None:
        """处理 quote tick（可选）"""
        # Quote tick 通常用于价格更新，不用于订单流特征
        pass

    def on_trade_tick(self, tick: TradeTick) -> None:
        """处理 trade tick（用于订单流特征）"""
        try:
            # 更新特征计算器
            self.feature_computer.on_tick(tick)
        except Exception as e:
            self.log.error(f"❌ Error processing trade tick: {e}")

    def on_bar(self, bar: Bar) -> None:
        """处理 bar 数据（更新时间框架特征）"""
        try:
            # 确定时间框架
            timeframe = self._get_timeframe_from_bar(bar)
            if timeframe is None:
                return

            # 更新特征计算器
            self.feature_computer.on_bar(bar, timeframe=timeframe)

            # 缓存时间框架特征
            if timeframe not in self.timeframe_features:
                self.timeframe_features[timeframe] = {}

            tf_features = self.feature_computer.timeframe_features.get(timeframe, {})
            self.timeframe_features[timeframe].update(tf_features)

        except Exception as e:
            self.log.error(f"❌ Error processing bar: {e}")
            import traceback

            self.log.error(traceback.format_exc())

    def _get_timeframe_from_bar(self, bar: Bar) -> Optional[str]:
        """从 bar 获取时间框架字符串"""
        # 从 bar_type 推断时间框架
        for timeframe, bar_type in self.bar_types.items():
            if bar.bar_type == bar_type:
                return timeframe
        return None

    def _schedule_next_check(self) -> None:
        """安排下一次信号检查（对齐到整点）"""
        now_ns = self.clock.timestamp_ns()
        now_sec = now_ns // 1_000_000_000
        current_min = (now_sec // 60) % 60

        # 计算下一个检查点（对齐到 check_interval_minutes 的倍数）
        next_check_min = (
            (current_min // self.check_interval_minutes) + 1
        ) * self.check_interval_minutes

        if next_check_min >= 60:
            next_check_min = 0
            delay_sec = (60 - current_min) * 60 - (now_sec % 60)
        else:
            delay_sec = (next_check_min - current_min) * 60 - (now_sec % 60)

        if delay_sec <= 0:
            delay_sec += self.check_interval_minutes * 60

        self.clock.set_timer(
            name="signal_check",
            interval=delay_sec,
            callback=self._on_signal_check,
        )

    def _on_signal_check(self, event=None) -> None:
        """定时器回调：执行信号融合和交易决策"""
        try:
            current_time_ns = self.clock.timestamp_ns()

            # 冷却期检查
            if (
                self.last_order_time_ns is not None
                and current_time_ns - self.last_order_time_ns
                < self.min_order_interval_ns
            ):
                self.log.debug("Signal check skipped: within cooldown period")
                self._schedule_next_check()
                return

            # 获取所有特征
            all_features = self.feature_computer.get_features()

            # 获取订单流特征（最近 15 分钟）
            orderflow_features = self.feature_computer.get_orderflow_features(
                window_minutes=15
            )

            # 融合信号
            should_enter, signal_reason = self._evaluate_entry_signal(
                all_features,
                orderflow_features,
            )

            decision_id = build_decision_id(
                strategy_name=str(self.strategy_name),
                symbol=str(self.instrument_id),
                decision_ts_ns=current_time_ns,
            )
            if self._exec_stage_writers:
                feats = {**(all_features or {}), **(orderflow_features or {})}
                record = build_stage_record(
                    stage="features",
                    decision_id=decision_id,
                    decision_ts_ns=current_time_ns,
                    source="live",
                    run_id=(
                        str(os.getenv("MLBOT_RUN_ID"))
                        if os.getenv("MLBOT_RUN_ID")
                        else None
                    ),
                    symbol=str(self.instrument_id),
                    timeframe="event",
                    strategy_name=str(self.strategy_name),
                    instrument_id=str(self.instrument_id),
                    data=feats,
                )
                self._exec_stage_writers["features"].write(
                    record, decision_ts_ns=current_time_ns
                )

            if should_enter:
                quote = self.cache.quote_tick(self.instrument_id)
                if quote:
                    self._execute_entry(
                        side=signal_reason.get("side", OrderSide.BUY),
                        price=float(quote.mid),
                        reason=signal_reason.get("reason", "Unknown"),
                    )
                    self.last_order_time_ns = current_time_ns
                    self.log.info(f"📊 Entry executed: {signal_reason.get('reason')}")
                    if self._exec_stage_writers:
                        record = build_stage_record(
                            stage="execution",
                            decision_id=decision_id,
                            decision_ts_ns=current_time_ns,
                            source="live",
                            run_id=(
                                str(os.getenv("MLBOT_RUN_ID"))
                                if os.getenv("MLBOT_RUN_ID")
                                else None
                            ),
                            symbol=str(self.instrument_id),
                            timeframe="event",
                            strategy_name=str(self.strategy_name),
                            instrument_id=str(self.instrument_id),
                            data={
                                "intent": True,
                                "submit_order": True,
                                "side": str(signal_reason.get("side")),
                                "qty": float(self.trade_size),
                                "price": float(quote.mid),
                                "reason": str(signal_reason.get("reason")),
                            },
                        )
                        self._exec_stage_writers["execution"].write(
                            record, decision_ts_ns=current_time_ns
                        )
                else:
                    self.log.warning("No quote available for entry execution")
            else:
                if self._exec_stage_writers:
                    record = build_stage_record(
                        stage="execution",
                        decision_id=decision_id,
                        decision_ts_ns=current_time_ns,
                        source="live",
                        run_id=(
                            str(os.getenv("MLBOT_RUN_ID"))
                            if os.getenv("MLBOT_RUN_ID")
                            else None
                        ),
                        symbol=str(self.instrument_id),
                        timeframe="event",
                        strategy_name=str(self.strategy_name),
                        instrument_id=str(self.instrument_id),
                        data={"intent": False, "submit_order": False},
                    )
                    self._exec_stage_writers["execution"].write(
                        record, decision_ts_ns=current_time_ns
                    )

            # 安排下一次检查
            self._schedule_next_check()

        except Exception as e:
            self.log.error(f"❌ Error in signal check: {e}")
            import traceback

            self.log.error(traceback.format_exc())
            self._schedule_next_check()

    def _evaluate_entry_signal(
        self,
        all_features: Dict[str, float],
        orderflow_features: Dict[str, float],
    ) -> tuple[bool, Dict[str, Any]]:
        """
        评估入场信号

        Returns:
            (should_enter, signal_info)
        """
        # 1. 基础特征检查
        if not all_features:
            return False, {}

        # 2. 如果有模型，使用模型预测
        if self.model is not None:
            try:
                # 准备特征向量
                feature_vector = self._prepare_feature_vector(all_features)
                if feature_vector is None:
                    return False, {}

                # 模型预测（使用 ModelArtifact 或直接使用 model）
                if self.model_artifact is not None:
                    # 使用 ModelArtifact 进行预测（自动使用 preprocessor）
                    # 合并所有特征
                    all_feature_dict = {**all_features, **orderflow_features}
                    feature_df = pd.DataFrame([all_feature_dict])
                    predictions = self.model_artifact.predict(feature_df)
                    prediction = predictions[0] if len(predictions) > 0 else 0
                    # 尝试获取概率（如果模型支持）
                    if hasattr(self.model_artifact.model, "predict_proba"):
                        X = self.model_artifact.preprocessor.transform(feature_df)
                        proba = self.model_artifact.model.predict_proba(X)
                        probability = proba[0] if len(proba) > 0 else None
                    else:
                        probability = None
                else:
                    # 兼容旧格式：直接使用 model
                    if hasattr(self.model, "predict_proba"):
                        prediction = self.model.predict([feature_vector])[0]
                        probability = self.model.predict_proba([feature_vector])[0]
                    else:
                        prediction = self.model.predict([feature_vector])[0]
                        probability = None

                # 根据预测生成信号
                if prediction == 1:
                    return True, {
                        "side": OrderSide.BUY,
                        "reason": f"Model_Buy (prob: {probability[1] if probability is not None else 'N/A'})",
                    }
                elif prediction == -1:
                    return True, {
                        "side": OrderSide.SELL,
                        "reason": f"Model_Sell (prob: {probability[-1] if probability is not None else 'N/A'})",
                    }
            except Exception as e:
                self.log.warning(
                    f"Model prediction failed: {e}, falling back to rule-based"
                )

        # 3. 规则-based 信号（示例）
        vpin = orderflow_features.get("vpin", 0.0)
        imbalance = orderflow_features.get("imbalance", 0.0)
        total_vol = orderflow_features.get("total_vol", 0.0)

        # 高 VPIN + 卖方主导 → 做空
        if vpin > 0.6 and imbalance < -0.2 and total_vol > 1.0:
            return True, {
                "side": OrderSide.SELL,
                "reason": "High VPIN + Sell Imbalance",
            }

        # 低 VPIN + 买方突增 → 做多
        if vpin < 0.2 and imbalance > 0.3 and total_vol > 2.0:
            return True, {
                "side": OrderSide.BUY,
                "reason": "Low VPIN + Buy Surge",
            }

        return False, {}

    def _prepare_feature_vector(
        self, features: Dict[str, float]
    ) -> Optional[np.ndarray]:
        """准备特征向量（需要与训练时一致）"""
        # TODO: 从策略配置中获取特征列表
        # 这里简化处理，实际应该从 config 中读取
        if not features:
            return None

        # 转换为数组（按字母顺序排序以保持一致性）
        feature_names = sorted(features.keys())
        feature_values = [features[name] for name in feature_names]

        return np.array(feature_values)

    def _execute_entry(self, side: OrderSide, price: float, reason: str) -> None:
        """执行入场"""
        try:
            quantity = self.instrument.make_qty(self.trade_size)

            # Constitution enforcement (whitelist + slots) BEFORE submit_order
            if (
                self._constitution_executor is not None
                and self._constitution_runtime_state is not None
            ):
                root, reg = resolve_execution_profile_paths()
                ex = resolve_execution_profile(
                    strategy_id=str(self.strategy_name),
                    profile_root=root,
                    archetype_registry_path=reg,
                )
                if ex is None:
                    mode, exec_id, rules = "TREND", "MomentumExpansion", []
                else:
                    mode, exec_id, rules = (
                        ex.router_mode,
                        ex.execution_strategy_id,
                        ex.evidence_rules,
                    )
                # Build evidence from merged feature set keys (all_features+orderflow_features are used upstream)
                merged_feats = {}
                try:
                    merged_feats.update(self.feature_computer.get_features() or {})
                    merged_feats.update(
                        self.feature_computer.get_orderflow_features(window_minutes=15)
                        or {}
                    )
                except Exception:
                    merged_feats = {}
                try:
                    quantiles = load_evidence_quantiles(
                        os.getenv("MLBOT_EVIDENCE_QUANTILES_JSON")
                    )
                    evidence = compute_execution_evidence(
                        features=merged_feats,
                        rules=rules,
                        quantiles=quantiles,
                    )
                except Exception as e:
                    self.log.error(f"❌ evidence_dsl_error -> NO_TRADE: {e}")
                    return
                enforce_before_order(
                    executor=self._constitution_executor,
                    runtime_state=self._constitution_runtime_state,
                    position_id=f"{self.strategy_name}:{int(self.clock.timestamp_ns())}",
                    symbol=str(self.instrument_id),
                    mode=mode,
                    execution_strategy=exec_id,
                    execution_tags=[str(reason)],
                    execution_evidence=evidence,
                )

            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=quantity,
            )
            if (
                self._xm is not None
                and self._constitution_executor is not None
                and self._constitution_runtime_state is not None
            ):
                self._xm.submit_order_guarded(
                    order=order,
                    ctx=GuardedOrderContext(
                        position_id=f"{self.strategy_name}:{int(self.clock.timestamp_ns())}",
                        symbol=str(self.instrument_id),
                        mode=mode,
                        execution_strategy=exec_id,
                        execution_tags=[str(reason)],
                        execution_evidence=evidence,
                    ),
                )
            else:
                self.log.error(
                    "❌ No ExecutionManager/ConstitutionExecutor; refusing to submit order"
                )
                return

            self.log.info(f"📊 Entry: {side} {quantity} @ {price} ({reason})")

        except Exception as e:
            self.log.error(f"❌ Error executing entry: {e}")
            import traceback

            self.log.error(traceback.format_exc())

    def _load_model(self, model_path: str) -> Any:
        """加载模型"""
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def on_stop(self) -> None:
        """策略停止"""
        self.log.info(f"🛑 Stopping {self.strategy_name} strategy")
        self.feature_computer.reset()
