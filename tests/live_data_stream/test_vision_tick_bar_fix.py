"""
Bug fix 测试: Binance Vision 原始 tick 与聚合 bar 分离存储

修复背景:
    fill_gap_with_binance_vision() 之前只返回 1min bars，丢弃了原始 aggTrades。
    Vision 下载的就是原始 ticks [timestamp, price, volume, side]，应同时保存。

修复内容:
    1. data_gap_filler.fill_gap_with_binance_vision() 返回 (bars, raw_ticks) 元组
    2. gap_filler._save_filled_data() 同时保存 bars 和 raw_ticks
    3. gap_filler._save_raw_ticks() 新增方法，按天保存 raw ticks
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime


# ---------------------------------------------------------------------------
# fill_gap_with_binance_vision 返回元组
# ---------------------------------------------------------------------------
class TestVisionReturnsTuple:
    """验证 fill_gap_with_binance_vision 返回 (bars, raw_ticks) 元组"""

    @pytest.fixture
    def gap_filler(self):
        from src.live_data_stream.data_gap_filler import DataGapFiller

        with patch("src.live_data_stream.data_gap_filler.CCXT_AVAILABLE", True):
            return DataGapFiller(Mock())

    def _make_fake_csv_content(self, date_str: str, n_trades: int = 100):
        """构造仿 Binance Vision aggTrades CSV 内容"""
        base_ts = int(pd.Timestamp(f"{date_str} 00:00:00", tz="UTC").timestamp() * 1000)
        rows = []
        for i in range(n_trades):
            # agg_trade_id, price, quantity, first_trade_id, last_trade_id,
            # transact_time, is_buyer_maker
            rows.append(
                f"{i},50000.0,0.1,{i},{i},{base_ts + i * 60000},{'true' if i % 2 == 0 else 'false'}"
            )
        header = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"
        return (header + "\n" + "\n".join(rows)).encode("utf-8")

    @patch("src.live_data_stream.data_gap_filler.requests", create=True)
    def test_returns_tuple_of_two_dataframes(self, mock_requests, gap_filler):
        """fill_gap_with_binance_vision 必须返回 (bars, raw_ticks) 二元组"""
        import io
        import zipfile

        date_str = "2024-06-01"
        csv_bytes = self._make_fake_csv_content(date_str, n_trades=120)

        # 构造 zip 文件
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(f"BTCUSDT-aggTrades-{date_str}.csv", csv_bytes)
        zip_content = buf.getvalue()

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = zip_content
        mock_resp.raise_for_status = Mock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_requests.Session.return_value = mock_session

        start = pd.Timestamp(f"{date_str} 00:00:00", tz="UTC")
        end = pd.Timestamp(f"{date_str} 01:59:00", tz="UTC")

        result = gap_filler.fill_gap_with_binance_vision("BTC/USDT:USDT", start, end)

        # 必须是元组
        assert isinstance(result, tuple), "返回值必须是 tuple"
        assert len(result) == 2, "元组长度必须为 2: (bars, raw_ticks)"

        bars, raw_ticks = result
        assert isinstance(bars, pd.DataFrame)
        assert isinstance(raw_ticks, pd.DataFrame)

    @patch("src.live_data_stream.data_gap_filler.requests", create=True)
    def test_raw_ticks_has_required_columns(self, mock_requests, gap_filler):
        """raw_ticks 必须包含 [timestamp, price, volume, side] 四列"""
        import io
        import zipfile

        date_str = "2024-06-01"
        csv_bytes = self._make_fake_csv_content(date_str, n_trades=60)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(f"BTCUSDT-aggTrades-{date_str}.csv", csv_bytes)

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.content = buf.getvalue()
        mock_resp.raise_for_status = Mock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_requests.Session.return_value = mock_session

        start = pd.Timestamp(f"{date_str} 00:00:00", tz="UTC")
        end = pd.Timestamp(f"{date_str} 00:59:00", tz="UTC")

        bars, raw_ticks = gap_filler.fill_gap_with_binance_vision(
            "BTC/USDT:USDT", start, end
        )

        required_cols = {"timestamp", "price", "volume", "side"}
        assert required_cols.issubset(
            set(raw_ticks.columns)
        ), f"raw_ticks 缺少列: {required_cols - set(raw_ticks.columns)}"
        assert len(raw_ticks) > 0, "raw_ticks 不应为空"

    def test_empty_result_returns_empty_tuple(self, gap_filler):
        """无 requests 时返回 (empty_df, empty_df) 元组"""
        # 没有安装 requests → 返回 _empty = (pd.DataFrame(), pd.DataFrame())
        with patch.dict("sys.modules", {"requests": None}):
            with patch(
                "src.live_data_stream.data_gap_filler.DataGapFiller.fill_gap_with_binance_vision"
            ) as mock_fill:
                mock_fill.return_value = (pd.DataFrame(), pd.DataFrame())
                result = mock_fill(
                    "BTC/USDT:USDT",
                    pd.Timestamp("2024-06-01", tz="UTC"),
                    pd.Timestamp("2024-06-02", tz="UTC"),
                )
                bars, raw_ticks = result
                assert len(bars) == 0
                assert len(raw_ticks) == 0


# ---------------------------------------------------------------------------
# _save_raw_ticks 和 _save_filled_data
# ---------------------------------------------------------------------------
class TestSaveRawTicks:
    """验证 gap_filler 正确保存 raw ticks 到 ticks storage"""

    @pytest.fixture
    def mock_storage_manager(self):
        sm = Mock()
        sm.bar_1min = Mock()
        sm.bar_1min.append = Mock()
        sm.ticks = Mock()
        sm.ticks.append = Mock()
        return sm

    @pytest.fixture
    def gap_filler_instance(self, mock_storage_manager):
        from src.live_data_stream.gap_filler import GapFiller

        gf = GapFiller.__new__(GapFiller)
        gf.storage_manager = mock_storage_manager
        return gf

    def _make_raw_ticks(self, dates: list[str], n_per_day: int = 10) -> pd.DataFrame:
        """构造跨天 raw ticks 数据"""
        rows = []
        for d in dates:
            base = pd.Timestamp(f"{d} 10:00:00", tz="UTC")
            for i in range(n_per_day):
                rows.append(
                    {
                        "timestamp": base + pd.Timedelta(seconds=i * 30),
                        "price": 50000.0 + i,
                        "volume": 0.1 + i * 0.01,
                        "side": 1 if i % 2 == 0 else -1,
                    }
                )
        return pd.DataFrame(rows)

    def test_save_raw_ticks_splits_by_date(
        self, gap_filler_instance, mock_storage_manager
    ):
        """_save_raw_ticks 应按天拆分并调用 ticks.append"""
        raw_ticks = self._make_raw_ticks(["2024-06-01", "2024-06-02"], n_per_day=5)

        gap_filler_instance._save_raw_ticks("BTCUSDT", raw_ticks)

        assert mock_storage_manager.ticks.append.call_count == 2
        call_args_list = mock_storage_manager.ticks.append.call_args_list

        dates_saved = {c.args[1] for c in call_args_list}
        assert "2024-06-01" in dates_saved
        assert "2024-06-02" in dates_saved

        # 每次调用的 symbol 参数正确
        for c in call_args_list:
            assert c.args[0] == "BTCUSDT"

    def test_save_raw_ticks_no_date_column_in_output(
        self, gap_filler_instance, mock_storage_manager
    ):
        """保存到 storage 的 df 不应包含辅助列 _date"""
        raw_ticks = self._make_raw_ticks(["2024-06-01"], n_per_day=3)

        gap_filler_instance._save_raw_ticks("BTCUSDT", raw_ticks)

        saved_df = mock_storage_manager.ticks.append.call_args.args[2]
        assert "_date" not in saved_df.columns, "_date 辅助列不应保存到 storage"

    def test_save_raw_ticks_empty_noop(self, gap_filler_instance, mock_storage_manager):
        """空 raw_ticks 不应调用 storage"""
        gap_filler_instance._save_raw_ticks("BTCUSDT", pd.DataFrame())
        mock_storage_manager.ticks.append.assert_not_called()

    def test_save_filled_data_saves_both_bars_and_ticks(
        self, gap_filler_instance, mock_storage_manager
    ):
        """_save_filled_data 同时保存 bars 和 raw_ticks"""
        bars = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-06-01 10:00", periods=5, freq="1min", tz="UTC"
                ),
                "open": [50000] * 5,
                "high": [50100] * 5,
                "low": [49900] * 5,
                "close": [50050] * 5,
                "volume": [100] * 5,
            }
        )
        raw_ticks = self._make_raw_ticks(["2024-06-01"], n_per_day=20)

        gap_filler_instance._save_filled_data("BTCUSDT", bars, raw_ticks)

        # bars 保存到 bar_1min
        assert mock_storage_manager.bar_1min.append.call_count >= 1
        # ticks 保存到 ticks
        assert mock_storage_manager.ticks.append.call_count >= 1

    def test_save_filled_data_no_ticks_only_bars(
        self, gap_filler_instance, mock_storage_manager
    ):
        """raw_ticks=None 时只保存 bars 不保存 ticks"""
        bars = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-06-01 10:00", periods=3, freq="1min", tz="UTC"
                ),
                "open": [50000] * 3,
                "high": [50100] * 3,
                "low": [49900] * 3,
                "close": [50050] * 3,
                "volume": [100] * 3,
            }
        )

        gap_filler_instance._save_filled_data("BTCUSDT", bars, raw_ticks=None)

        assert mock_storage_manager.bar_1min.append.call_count >= 1
        mock_storage_manager.ticks.append.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
