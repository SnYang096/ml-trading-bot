"""
冒烟测试: LivePCM 接入 run_live.py 链路

验证:
1. _setup_bpc 返回 LivePCM（而非裸策略实例）
2. LivePCM 注册了 bpc 策略
3. listener.decision_handler 指向 LivePCM
4. quantiles 可以透传
"""

import pytest
from unittest.mock import MagicMock, patch


class TestLivePCMSmoke:
    """冒烟测试：验证 run_live._setup_bpc 的 PCM 接入"""

    @patch("scripts.run_live.init_order_manager_from_env")
    @patch("scripts.run_live.MultiSymbolManager")
    @patch(
        "src.time_series_model.live.generic_live_strategy.GenericLiveStrategy.load_configs"
    )
    def test_setup_bpc_returns_live_pcm(
        self, mock_load_configs, mock_multi_mgr_cls, mock_init_om
    ):
        """_setup_bpc 返回 LivePCM 而非裸策略实例"""
        from time_series_model.portfolio.live_pcm import LivePCM

        # Mock MultiSymbolManager
        mock_manager = MagicMock()
        mock_listener = MagicMock()
        mock_manager.get_listener.return_value = mock_listener
        mock_multi_mgr_cls.return_value = mock_manager

        mock_init_om.return_value = MagicMock()

        # 设置环境变量
        env = {
            "MLBOT_STRATEGIES_ROOT": "config/strategies",
            "MLBOT_BPC_BAR_MINUTES": "240",
            "MLBOT_BPC_WINDOW_MINUTES": "15",
            "MLBOT_MAX_SLOTS": "2",
        }

        with patch.dict(os.environ, env, clear=False):
            from scripts.run_live import _setup_bpc

            manager, pcm = _setup_bpc(
                symbols=["BTCUSDT"],
                storage=MagicMock(),
                gap_filler=None,
                trade_size=0.0,
            )

        # 1. 返回 LivePCM
        assert type(pcm).__name__ == "LivePCM", f"Expected LivePCM, got {type(pcm)}"

        # 2. 注册了 bpc 策略
        assert "bpc" in pcm.registered_archetypes

        # 3. listener 的 decision_handler 指向 pcm
        assert mock_listener.decision_handler is pcm

    def test_live_pcm_quantiles_passthrough(self):
        """quantiles 可以透传到内部策略"""
        from time_series_model.portfolio.live_pcm import LivePCM

        # Mock BPC strategy
        mock_bpc = MagicMock()
        mock_bpc.decide.return_value = []

        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", mock_bpc)

        # set_quantiles_from_df 应透传
        pcm.set_quantiles_from_df("fake_df")
        mock_bpc.set_quantiles_from_df.assert_called_once_with("fake_df")

    def test_live_pcm_decide_delegates_to_bpc(self):
        """PCM.decide() 正确委托给注册的 BPC 策略"""
        from time_series_model.portfolio.live_pcm import LivePCM
        from time_series_model.core.trade_intent import TradeIntent

        intent = TradeIntent(
            action="LONG",
            symbol="BTCUSDT",
            archetype="bpc",
            confidence=0.8,
        )

        mock_bpc = MagicMock()
        mock_bpc.decide.return_value = [intent]

        pcm = LivePCM(max_slots=2)
        pcm.register("bpc", mock_bpc)

        result = pcm.decide(
            features={"close": 50000.0},
            symbol="BTCUSDT",
        )

        assert len(result) == 1
        assert result[0] is intent
        mock_bpc.decide.assert_called_once()
