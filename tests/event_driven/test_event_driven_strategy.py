"""
EventDrivenStrategy 单元测试
"""

import pytest
from unittest.mock import MagicMock, Mock, patch
from pathlib import Path

try:
    from nautilus_trader.model import InstrumentId, BarType, BarSpecification
    from nautilus_trader.model import BarAggregation, PriceType, AggregationSource
    from nautilus_trader.model import TradeTick
    from nautilus_trader.model.enums import AggressorSide
    from nautilus_trader.model.identifiers import TradeId

    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    pytest.skip("Nautilus Trader not available", allow_module_level=True)

from src.time_series_model.live.event_driven_strategy import EventDrivenStrategy


class TestEventDrivenStrategy:
    """EventDrivenStrategy 单元测试"""

    @pytest.fixture
    def instrument_id(self):
        """创建测试用的 instrument ID"""
        return InstrumentId.from_str("BTCUSDT-PERP.BINANCE")

    @pytest.fixture
    def bar_types(self, instrument_id):
        """创建测试用的 bar types"""
        bar_spec_15m = BarSpecification(15, BarAggregation.MINUTE, PriceType.LAST)
        bar_spec_1h = BarSpecification(1, BarAggregation.HOUR, PriceType.LAST)

        return {
            "15T": BarType(
                instrument_id=instrument_id,
                bar_spec=bar_spec_15m,
                aggregation_source=AggregationSource.EXTERNAL,
            ),
            "1H": BarType(
                instrument_id=instrument_id,
                bar_spec=bar_spec_1h,
                aggregation_source=AggregationSource.EXTERNAL,
            ),
        }

    @pytest.fixture
    def strategy_config(self, instrument_id, bar_types):
        """创建测试用的策略配置"""
        return {
            "strategy_name": "test_strategy",
            "instrument_id": instrument_id,
            "bar_types": bar_types,
            "trade_size": 0.001,
            "config_base_path": "config/strategies",
            "check_interval_minutes": 15,
            "min_order_interval_minutes": 15,
        }

    def test_init(self, strategy_config):
        """测试初始化"""
        with patch(
            "src.time_series_model.live.event_driven_strategy.Strategy.__init__"
        ):
            strategy = EventDrivenStrategy(**strategy_config)

            assert strategy.strategy_name == "test_strategy"
            assert strategy.trade_size == 0.001
            assert strategy.check_interval_minutes == 15
            assert strategy.min_order_interval_ns == 15 * 60 * 1_000_000_000
            assert strategy.feature_computer is not None

    def test_get_timeframe_from_bar(self, strategy_config, bar_types):
        """测试从 bar 获取时间框架"""
        with patch(
            "src.time_series_model.live.event_driven_strategy.Strategy.__init__"
        ):
            strategy = EventDrivenStrategy(**strategy_config)

            # 创建模拟 bar
            bar = MagicMock()
            bar.bar_type = bar_types["15T"]

            timeframe = strategy._get_timeframe_from_bar(bar)

            assert timeframe == "15T"

    def test_prepare_feature_vector(self, strategy_config):
        """测试准备特征向量"""
        with patch(
            "src.time_series_model.live.event_driven_strategy.Strategy.__init__"
        ):
            strategy = EventDrivenStrategy(**strategy_config)

            features = {
                "vpin": 0.5,
                "imbalance": 0.3,
                "rsi": 60.0,
                "atr": 100.0,
            }

            feature_vector = strategy._prepare_feature_vector(features)

            assert feature_vector is not None
            assert len(feature_vector) == 4
            assert isinstance(
                feature_vector, type(features["vpin"]).__module__ == "numpy" or list
            )

    def test_prepare_feature_vector_empty(self, strategy_config):
        """测试空特征向量"""
        with patch(
            "src.time_series_model.live.event_driven_strategy.Strategy.__init__"
        ):
            strategy = EventDrivenStrategy(**strategy_config)

            feature_vector = strategy._prepare_feature_vector({})

            assert feature_vector is None

    def test_evaluate_entry_signal_no_features(self, strategy_config):
        """测试无特征时的信号评估"""
        with patch(
            "src.time_series_model.live.event_driven_strategy.Strategy.__init__"
        ):
            strategy = EventDrivenStrategy(**strategy_config)

            should_enter, signal_info = strategy._evaluate_entry_signal({}, {})

            assert should_enter is False
            assert signal_info == {}

    def test_evaluate_entry_signal_rule_based(self, strategy_config):
        """测试规则-based 信号评估"""
        with patch(
            "src.time_series_model.live.event_driven_strategy.Strategy.__init__"
        ):
            strategy = EventDrivenStrategy(**strategy_config)
            strategy.model = None  # 无模型，使用规则-based

            all_features = {
                "vpin": 0.7,  # 高 VPIN
                "imbalance": -0.3,  # 卖方主导
            }

            orderflow_features = {
                "vpin": 0.7,
                "imbalance": -0.3,
                "total_vol": 2.0,  # 足够大的成交量
            }

            should_enter, signal_info = strategy._evaluate_entry_signal(
                all_features,
                orderflow_features,
            )

            # 高 VPIN + 卖方主导 → 应该做空
            assert should_enter is True
            assert "side" in signal_info
            assert "reason" in signal_info

    def test_evaluate_entry_signal_with_model(self, strategy_config):
        """测试使用模型的信号评估"""
        with patch(
            "src.time_series_model.live.event_driven_strategy.Strategy.__init__"
        ):
            strategy = EventDrivenStrategy(**strategy_config)

            # 创建模拟模型
            mock_model = MagicMock()
            mock_model.predict.return_value = [1]  # 买入信号
            mock_model.predict_proba.return_value = [[0.3, 0.7]]  # 70% 买入概率
            strategy.model = mock_model

            all_features = {
                "vpin": 0.5,
                "imbalance": 0.2,
                "rsi": 50.0,
            }

            orderflow_features = {
                "vpin": 0.5,
                "imbalance": 0.2,
                "total_vol": 1.0,
            }

            should_enter, signal_info = strategy._evaluate_entry_signal(
                all_features,
                orderflow_features,
            )

            # 模型预测买入，应该返回 True
            assert should_enter is True
            assert "side" in signal_info
            assert "reason" in signal_info

    def test_load_model(self, strategy_config, tmp_path):
        """测试加载模型"""
        import pickle

        with patch(
            "src.time_series_model.live.event_driven_strategy.Strategy.__init__"
        ):
            strategy = EventDrivenStrategy(**strategy_config)

            # 创建测试模型文件
            test_model = {"type": "test_model", "params": {"test": "value"}}
            model_path = tmp_path / "test_model.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(test_model, f)

            loaded_model = strategy._load_model(str(model_path))

            assert loaded_model == test_model
