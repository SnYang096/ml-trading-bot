#!/usr/bin/env python3
"""测试 system_mode.py 的 features_4h 推算逻辑

验证：当 ticks_1min 为空但 features_4h 有数据时，系统能够正确判断为 NORMAL 模式
"""
import pandas as pd

from live_data_stream.system_mode import SystemModeManager, SystemMode


def test_features_4h_fallback():
    """测试 features_4h 作为数据覆盖判断的备用方案"""

    manager = SystemModeManager()

    # 场景 1: 无任何数据 → 应判定为 OFFLINE
    print("\n" + "=" * 60)
    print("测试场景 1: 无任何数据")
    print("=" * 60)
    warmup_data = {
        "ticks_1min": pd.DataFrame(),
        "features_4h": pd.DataFrame(),
    }
    decision = manager.decide_mode(warmup_data)
    print(f"✓ 判定结果: {decision.mode.value}")
    print(f"  原因: {decision.reason}")
    assert decision.mode == SystemMode.OFFLINE, "无数据应判定为 OFFLINE"

    # 场景 2: 只有 features_4h (109条) → 应判定为 NORMAL
    print("\n" + "=" * 60)
    print("测试场景 2: 只有 features_4h (109条 = 436小时)")
    print("=" * 60)

    # 模拟 109 条 4h 特征数据
    timestamps = pd.date_range(start="2025-12-01 00:00:00", periods=109, freq="4h")
    features_4h = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [50000] * 109,
            "volume": [1000] * 109,
        }
    )

    warmup_data = {
        "ticks_1min": pd.DataFrame(),
        "features_4h": features_4h,
    }

    decision = manager.decide_mode(warmup_data)
    print(f"✓ 判定结果: {decision.mode.value}")
    print(f"  原因: {decision.reason}")
    print(f"  等效 bars: {decision.bar_count} (109 × 240 = 26,160)")
    print(f"  覆盖时长: {decision.data_coverage_hours:.2f} 小时")

    assert (
        decision.mode == SystemMode.NORMAL
    ), f"109条4h特征应判定为NORMAL，实际为{decision.mode.value}"
    assert (
        decision.bar_count == 109 * 240
    ), f"应等效26,160个1min bars，实际为{decision.bar_count}"

    # 场景 3: features_4h 不足（只有 20条 = 80小时） → 应判定为 DEGRADED 或 OFFLINE
    print("\n" + "=" * 60)
    print("测试场景 3: features_4h 不足 (20条 = 80小时)")
    print("=" * 60)

    timestamps_short = pd.date_range(start="2025-12-01 00:00:00", periods=20, freq="4h")
    features_4h_short = pd.DataFrame(
        {
            "timestamp": timestamps_short,
            "close": [50000] * 20,
        }
    )

    warmup_data = {
        "ticks_1min": pd.DataFrame(),
        "features_4h": features_4h_short,
    }

    decision = manager.decide_mode(warmup_data)
    print(f"✓ 判定结果: {decision.mode.value}")
    print(f"  原因: {decision.reason}")
    print(f"  等效 bars: {decision.bar_count} (20 × 240 = 4,800)")
    print(f"  覆盖时长: {decision.data_coverage_hours:.2f} 小时")

    assert decision.mode in [
        SystemMode.NORMAL,
        SystemMode.DEGRADED,
    ], f"20条4h特征（4800个1min bars）应判定为NORMAL或DEGRADED，实际为{decision.mode.value}"

    # 场景 4: 同时有 ticks_1min 和 features_4h → 应优先使用 ticks_1min
    print("\n" + "=" * 60)
    print("测试场景 4: 同时有 ticks_1min 和 features_4h")
    print("=" * 60)

    ticks_1min = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2025-12-01", periods=300, freq="1min"),
            "close": [50000] * 300,
        }
    )

    warmup_data = {
        "ticks_1min": ticks_1min,
        "features_4h": features_4h,
    }

    decision = manager.decide_mode(warmup_data)
    print(f"✓ 判定结果: {decision.mode.value}")
    print(f"  原因: {decision.reason}")
    print(f"  实际 bars: {decision.bar_count} (使用ticks_1min)")
    print(f"  覆盖时长: {decision.data_coverage_hours:.2f} 小时")

    assert (
        decision.bar_count == 300
    ), f"应使用ticks_1min的300个bars，实际为{decision.bar_count}"

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_features_4h_fallback()
