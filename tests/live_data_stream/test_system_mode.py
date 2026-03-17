"""测试系统模式管理器

测试场景：
1. 数据完整性检查
2. 模式决策逻辑
3. 模式切换历史
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.live_data_stream.system_mode import (
    SystemMode,
    SystemModeManager,
    ModeDecision,
)


class TestSystemModeManager:
    """测试SystemModeManager"""

    def test_decide_mode_offline_no_data(self):
        """测试：无数据 → OFFLINE"""
        manager = SystemModeManager()
        warmup_data = {"ticks_1min": pd.DataFrame()}

        decision = manager.decide_mode(warmup_data)

        assert decision.mode == SystemMode.OFFLINE
        assert decision.bar_count == 0
        assert "No ticks_1min data" in decision.reason

    def test_decide_mode_offline_insufficient_data(self):
        """测试：数据 < 2小时 → OFFLINE"""
        manager = SystemModeManager()

        # 创建100条bar（不足120）
        now = datetime.utcnow()
        timestamps = [now - timedelta(minutes=i) for i in range(100, 0, -1)]
        warmup_data = {
            "ticks_1min": pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": [100] * 100,
                    "close": [100] * 100,
                }
            )
        }

        decision = manager.decide_mode(warmup_data)

        assert decision.mode == SystemMode.OFFLINE
        assert decision.bar_count == 100
        assert "100 bars < 120" in decision.reason

    def test_decide_mode_degraded_partial_data(self):
        """测试：数据 2-4小时 → DEGRADED"""
        manager = SystemModeManager()

        # 创建180条bar（2-4小时之间）
        now = datetime.utcnow()
        timestamps = [now - timedelta(minutes=i) for i in range(180, 0, -1)]
        warmup_data = {
            "ticks_1min": pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": [100] * 180,
                    "close": [100] * 180,
                }
            )
        }

        decision = manager.decide_mode(warmup_data)

        assert decision.mode == SystemMode.DEGRADED
        assert decision.bar_count == 180
        assert "180 bars < 240" in decision.reason

    def test_decide_mode_degraded_large_gap(self):
        """测试：有大缺口（>5min） → DEGRADED"""
        manager = SystemModeManager()

        # 创建数据：先100条，然后10分钟缺口，再140条（总计240条）
        now = datetime.utcnow()
        timestamps = []

        # 第一段：100条bar
        base_time = now - timedelta(minutes=250)  # 从250分钟前开始
        for i in range(100):
            timestamps.append(base_time + timedelta(minutes=i))

        # 缺口：跳过10分钟
        gap_end = base_time + timedelta(minutes=110)  # 100 + 10

        # 第二段：140条bar
        for i in range(140):
            timestamps.append(gap_end + timedelta(minutes=i))

        warmup_data = {
            "ticks_1min": pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": [100] * 240,
                    "close": [100] * 240,
                }
            )
        }

        decision = manager.decide_mode(warmup_data)

        assert decision.mode == SystemMode.DEGRADED
        assert "Large gap detected" in decision.reason
        assert len(decision.missing_periods) > 0
        assert decision.missing_periods[0]["minutes"] > 5  # 缺口大于5分钟

    def test_decide_mode_normal_complete_data(self):
        """测试：完整4小时数据 → NORMAL"""
        manager = SystemModeManager()

        # 创建240条连续bar
        now = datetime.utcnow()
        timestamps = [now - timedelta(minutes=i) for i in range(240, 0, -1)]
        warmup_data = {
            "ticks_1min": pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": [100] * 240,
                    "close": [100] * 240,
                }
            )
        }

        decision = manager.decide_mode(warmup_data)

        assert decision.mode == SystemMode.NORMAL
        assert decision.bar_count == 240
        assert "Data complete" in decision.reason
        assert decision.data_coverage_hours >= 3.9  # 约4小时

    def test_detect_gaps(self):
        """测试：缺口检测"""
        manager = SystemModeManager()

        # 创建有缺口的数据：0-9分钟（10条），跳过到15分钟（15-24分钟，10条）
        # 相邻两条的间隔：9min到15min = 6分钟
        now = datetime.utcnow()
        timestamps = []

        # 第一段：0-9分钟
        for i in range(10):
            timestamps.append(now + timedelta(minutes=i))

        # 第二段：15-24分钟（跳过5分钟）
        for i in range(15, 25):
            timestamps.append(now + timedelta(minutes=i))

        ticks_1min = pd.DataFrame({"timestamp": timestamps})
        gaps = manager._detect_gaps(ticks_1min)

        # 9到15的间隔是6分钟，大于1.5分钟阈值
        assert len(gaps) == 1
        assert gaps[0]["minutes"] == pytest.approx(6.0, abs=0.1)

    def test_mode_history(self):
        """测试：模式切换历史记录"""
        manager = SystemModeManager()

        # 第一次：OFFLINE
        decision1 = ModeDecision(
            mode=SystemMode.OFFLINE,
            reason="Test offline",
            bar_count=0,
            data_coverage_hours=0.0,
        )
        manager.set_mode(decision1)

        # 第二次：DEGRADED
        decision2 = ModeDecision(
            mode=SystemMode.DEGRADED,
            reason="Test degraded",
            bar_count=180,
            data_coverage_hours=3.0,
        )
        manager.set_mode(decision2)

        # 第三次：NORMAL
        decision3 = ModeDecision(
            mode=SystemMode.NORMAL,
            reason="Test normal",
            bar_count=240,
            data_coverage_hours=4.0,
        )
        manager.set_mode(decision3)

        # 检查历史
        history = manager.get_mode_history()
        assert len(history) == 3
        assert history[0]["old_mode"] == "NORMAL"
        assert history[0]["new_mode"] == "OFFLINE"
        assert history[1]["old_mode"] == "OFFLINE"
        assert history[1]["new_mode"] == "DEGRADED"
        assert history[2]["old_mode"] == "DEGRADED"
        assert history[2]["new_mode"] == "NORMAL"

    def test_is_trading_allowed(self):
        """测试：是否允许交易"""
        manager = SystemModeManager()

        # OFFLINE → 禁止交易
        decision1 = ModeDecision(
            mode=SystemMode.OFFLINE,
            reason="Test",
            bar_count=0,
            data_coverage_hours=0.0,
        )
        manager.set_mode(decision1)
        assert not manager.is_trading_allowed()

        # DEGRADED → 禁止交易
        decision2 = ModeDecision(
            mode=SystemMode.DEGRADED,
            reason="Test",
            bar_count=180,
            data_coverage_hours=3.0,
        )
        manager.set_mode(decision2)
        assert not manager.is_trading_allowed()

        # NORMAL → 允许交易
        decision3 = ModeDecision(
            mode=SystemMode.NORMAL,
            reason="Test",
            bar_count=240,
            data_coverage_hours=4.0,
        )
        manager.set_mode(decision3)
        assert manager.is_trading_allowed()

    def test_abnormal_requires_manual_reset(self):
        """ABNORMAL 不自动恢复，必须手动 reset_to_normal"""
        manager = SystemModeManager()
        manager.set_mode(
            ModeDecision(
                mode=SystemMode.NORMAL,
                reason="ready",
                bar_count=300,
                data_coverage_hours=5.0,
            )
        )
        manager.trigger_abnormal("quick stop")
        assert manager.get_current_mode() == SystemMode.ABNORMAL
        assert not manager.is_trading_allowed()

        for _ in range(10):
            assert manager.on_realtime_bar() is False
            assert manager.get_current_mode() == SystemMode.ABNORMAL

        manager.reset_to_normal("manual fixed by cicd")
        assert manager.get_current_mode() == SystemMode.NORMAL
        assert manager.is_trading_allowed()

    def test_default_mode_on_boot_is_normal(self):
        manager = SystemModeManager()
        assert manager.get_current_mode() == SystemMode.NORMAL

    def test_mode_on_boot_can_be_overridden(self, monkeypatch):
        monkeypatch.setenv("MLBOT_MODE_ON_BOOT", "OFFLINE")
        manager = SystemModeManager()
        assert manager.get_current_mode() == SystemMode.OFFLINE


class TestModeDecision:
    """测试ModeDecision"""

    def test_to_dict(self):
        """测试：转换为字典"""
        decision = ModeDecision(
            mode=SystemMode.NORMAL,
            reason="Test reason",
            bar_count=240,
            data_coverage_hours=4.0,
            missing_periods=[
                {"start": "2024-01-01", "end": "2024-01-02", "minutes": 5.0}
            ],
        )

        data = decision.to_dict()

        assert data["mode"] == "NORMAL"
        assert data["reason"] == "Test reason"
        assert data["bar_count"] == 240
        assert data["data_coverage_hours"] == 4.0
        assert len(data["missing_periods"]) == 1
        assert "timestamp" in data

    def test_repr(self):
        """测试：字符串表示"""
        decision = ModeDecision(
            mode=SystemMode.DEGRADED,
            reason="Incomplete data",
            bar_count=180,
            data_coverage_hours=3.0,
        )

        repr_str = repr(decision)

        assert "DEGRADED" in repr_str
        assert "180" in repr_str
        assert "3.00h" in repr_str
        assert "Incomplete data" in repr_str
