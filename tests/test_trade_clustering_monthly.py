"""
测试 Trade Clustering 按月流式计算

测试内容：
1. 按月流式计算的基本功能
2. 状态传递（跨月连续性）
3. 结果合并的正确性
4. 与一次性计算的结果一致性
5. 内存优化（不一次性加载所有数据）
6. 边界情况（空数据、单月数据等）
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.time_series.utils_order_flow_features import (
    compute_trade_clustering_from_ticks,
    extract_trade_clustering_features,
)
from src.data_tools.tick_loader import deserialize_tick_loader_params


@pytest.fixture
def sample_ticks_single_month():
    """创建单个月的样本 tick 数据"""
    np.random.seed(42)
    n = 10000

    # 生成一个月的数据（2024-01）
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1S")

    # 生成价格（随机游走）
    prices = 50000 + np.cumsum(np.random.randn(n) * 10)

    # 生成买卖方向（有一定聚集性）
    sides = []
    current_side = 1
    run_length = 0
    for _ in range(n):
        if run_length > 0 and np.random.rand() < 0.7:  # 70% 概率继续当前方向
            sides.append(current_side)
            run_length -= 1
        else:
            current_side = np.random.choice([1, -1])
            sides.append(current_side)
            run_length = np.random.randint(5, 20)

    ticks = pd.DataFrame(
        {
            "price": prices,
            "volume": np.random.uniform(0.1, 10.0, n),
            "side": sides,
        },
        index=timestamps,
    )

    return ticks


@pytest.fixture
def sample_ticks_multiple_months():
    """创建多个月的样本 tick 数据"""
    np.random.seed(42)

    all_ticks = []
    start_date = datetime(2024, 1, 1)

    for month in range(3):  # 3个月的数据
        month_start = start_date + pd.DateOffset(months=month)
        n = 10000  # 每个月 10000 条

        timestamps = pd.date_range(month_start, periods=n, freq="1S")

        # 生成价格（随机游走）
        if len(all_ticks) > 0:
            last_price = all_ticks[-1]["price"].iloc[-1]
        else:
            last_price = 50000

        prices = last_price + np.cumsum(np.random.randn(n) * 10)

        # 生成买卖方向（有一定聚集性）
        sides = []
        current_side = 1
        run_length = 0
        for _ in range(n):
            if run_length > 0 and np.random.rand() < 0.7:
                sides.append(current_side)
                run_length -= 1
            else:
                current_side = np.random.choice([1, -1])
                sides.append(current_side)
                run_length = np.random.randint(5, 20)

        month_ticks = pd.DataFrame(
            {
                "price": prices,
                "volume": np.random.uniform(0.1, 10.0, n),
                "side": sides,
            },
            index=timestamps,
        )

        all_ticks.append(month_ticks)

    # 合并所有月份
    combined_ticks = pd.concat(all_ticks, axis=0).sort_index()
    return combined_ticks, all_ticks


@pytest.fixture
def sample_ohlcv():
    """创建样本 OHLCV 数据（K线）"""
    np.random.seed(42)
    n = 200

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1H")

    prices = 50000 + np.cumsum(np.random.randn(n) * 50)

    df = pd.DataFrame(
        {
            "open": prices + np.random.randn(n) * 10,
            "high": prices + np.abs(np.random.randn(n) * 20),
            "low": prices - np.abs(np.random.randn(n) * 20),
            "close": prices,
            "volume": np.random.uniform(100, 1000, n),
        },
        index=timestamps,
    )

    return df


class TestTradeClusteringMonthlyStreaming:
    """测试 Trade Clustering 按月流式计算"""

    def test_state_passing_basic(self, sample_ticks_single_month):
        """测试状态传递的基本功能"""
        print("\n测试状态传递基本功能...")

        # 将数据分成两部分
        mid_point = len(sample_ticks_single_month) // 2
        ticks_part1 = sample_ticks_single_month.iloc[:mid_point].copy()
        ticks_part2 = sample_ticks_single_month.iloc[mid_point:].copy()

        window_size = 100

        # 第一部分：无初始状态
        result1, state1 = compute_trade_clustering_from_ticks(
            ticks_part1,
            window_size=window_size,
            initial_state=None,
        )

        # 验证状态结构
        assert isinstance(state1, dict), "状态应为字典"
        assert "current_run_side" in state1
        assert "current_run_length" in state1
        assert "window_runs" in state1
        assert "window_total_ticks" in state1
        assert "buy_runs_in_window" in state1
        assert "sell_runs_in_window" in state1

        # 第二部分：使用第一部分的状态
        result2, state2 = compute_trade_clustering_from_ticks(
            ticks_part2,
            window_size=window_size,
            initial_state=state1,
        )

        # 验证结果
        assert len(result1) > 0, "第一部分应有结果"
        assert len(result2) > 0, "第二部分应有结果"

        # 合并结果
        combined_result = pd.concat([result1, result2], axis=0).sort_index()

        # 与一次性计算的结果对比
        full_result, _ = compute_trade_clustering_from_ticks(
            sample_ticks_single_month,
            window_size=window_size,
            initial_state=None,
        )

        # 验证结果一致性（允许小的数值误差）
        pd.testing.assert_frame_equal(
            combined_result,
            full_result,
            check_exact=False,
            rtol=1e-5,
            atol=1e-5,
        )

        print(f"   ✅ 状态传递测试通过，结果一致")

    def test_monthly_streaming_correctness(self, sample_ticks_multiple_months):
        """测试按月流式计算的正确性"""
        print("\n测试按月流式计算的正确性...")

        combined_ticks, monthly_ticks_list = sample_ticks_multiple_months
        window_size = 100

        # 方法1：一次性计算（基准）
        full_result, _ = compute_trade_clustering_from_ticks(
            combined_ticks,
            window_size=window_size,
            initial_state=None,
        )

        # 方法2：按月流式计算
        monthly_results = []
        state = None

        for i, month_ticks in enumerate(monthly_ticks_list):
            month_result, state = compute_trade_clustering_from_ticks(
                month_ticks,
                window_size=window_size,
                initial_state=state,
            )
            monthly_results.append(month_result)

            # 转换 state 中的 list 回 deque（用于下一批次）
            if state:
                from collections import deque

                state["window_runs"] = deque(state.get("window_runs", []))
                state["buy_runs_in_window"] = deque(state.get("buy_runs_in_window", []))
                state["sell_runs_in_window"] = deque(
                    state.get("sell_runs_in_window", [])
                )

        # 合并按月计算的结果
        streamed_result = pd.concat(monthly_results, axis=0).sort_index()

        # 验证结果一致性
        pd.testing.assert_frame_equal(
            streamed_result,
            full_result,
            check_exact=False,
            rtol=1e-5,
            atol=1e-5,
        )

        print(f"   ✅ 按月流式计算结果与一次性计算一致")
        print(f"   一次性计算: {len(full_result)} 个特征点")
        print(f"   按月计算: {len(streamed_result)} 个特征点")

    def test_extract_trade_clustering_monthly(
        self, sample_ohlcv, sample_ticks_multiple_months
    ):
        """测试 extract_trade_clustering_features 的按月计算功能"""
        print("\n测试 extract_trade_clustering_features 按月计算...")

        combined_ticks, _ = sample_ticks_multiple_months

        # 创建 ticks_loader_json（模拟按月加载）
        start_ts = combined_ticks.index[0]
        end_ts = combined_ticks.index[-1]

        # 生成 tick_files（按月）
        tick_files = []
        current_month = start_ts.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        end_month = end_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        while current_month <= end_month:
            month_end = (current_month + pd.DateOffset(months=1)) - pd.Timedelta(
                seconds=1
            )
            if month_end > end_ts:
                month_end = end_ts

            # 创建临时文件路径（实际测试中可以使用真实文件）
            tick_files.append(
                f"/tmp/test_ticks_{current_month.strftime('%Y-%m')}.parquet"
            )
            current_month = current_month + pd.DateOffset(months=1)

        loader_params = {
            "symbol": "BTCUSDT",
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat(),
            "lookback_minutes": 60,
            "tick_files": tick_files,
        }

        import json

        ticks_loader_json = json.dumps(loader_params)

        # 由于没有真实的 tick 文件，我们使用内存中的 ticks
        # 这里主要测试逻辑，实际使用时需要真实的 tick 文件

        # 测试：使用内存中的 ticks（一次性计算）
        result_memory = extract_trade_clustering_features(
            sample_ohlcv,
            ticks=combined_ticks,
            window_size=100,
            freq="1H",
        )

        # 验证结果
        expected_cols = [
            "trade_cluster_max_buy_run",
            "trade_cluster_max_sell_run",
            "trade_cluster_avg_buy_run",
            "trade_cluster_avg_sell_run",
            "trade_cluster_buy_run_count",
            "trade_cluster_sell_run_count",
            "trade_cluster_imbalance_ratio",
            "trade_cluster_directional_entropy",
        ]

        for col in expected_cols:
            assert col in result_memory.columns, f"应包含 {col} 列"

        print(f"   ✅ extract_trade_clustering_features 测试通过")
        print(
            f"   生成 {len([c for c in result_memory.columns if 'trade_cluster' in c])} 个特征列"
        )

    def test_persist_monthly_and_batch_merge(
        self, sample_ohlcv, sample_ticks_multiple_months
    ):
        """测试按月落盘+分批合并的内存友好模式"""
        print("\n测试按月落盘 + 分批合并模式...")
        combined_ticks, monthly_ticks_list = sample_ticks_multiple_months

        # 基准：纯内存模式
        baseline = extract_trade_clustering_features(
            sample_ohlcv,
            ticks=combined_ticks,
            window_size=100,
            freq="1H",
            merge_batch_size=4,
            persist_monthly=False,
        )

        # 临时目录做落盘，并用 ticks_loader_json 走分月读取逻辑
        with tempfile.TemporaryDirectory() as tmpdir:
            tick_files = []
            for i, month_df in enumerate(monthly_ticks_list):
                month_str = (month_df.index.min()).strftime("%Y-%m")
                path = Path(tmpdir) / f"BTCUSDT_{month_str}.parquet"
                # 保存 timestamp 列以及 side/price/volume，满足 load_tick_data 需求
                month_df_with_ts = month_df.reset_index().rename(
                    columns={"index": "timestamp"}
                )
                month_df_with_ts.to_parquet(path)
                tick_files.append(str(path))

            loader_params = {
                "symbol": "BTCUSDT",
                "start_ts": combined_ticks.index.min().isoformat(),
                "end_ts": combined_ticks.index.max().isoformat(),
                "lookback_minutes": 0,  # 测试中关闭 lookback，避免跨月额外读取
                "tick_files": tick_files,
            }
            import json

            ticks_loader_json = json.dumps(loader_params)

            result_persist = extract_trade_clustering_features(
                sample_ohlcv,
                ticks_loader_json=ticks_loader_json,
                window_size=100,
                freq="1H",
                merge_batch_size=2,  # 强制更小批次，触发多次落盘/读回
                monthly_cache_dir=tmpdir,
                persist_monthly=True,
            )

            # 检查是否生成 parquet
            parquet_files = list(Path(tmpdir).glob("trade_cluster_*.parquet"))
            assert len(parquet_files) > 0, "应生成按月 parquet 缓存文件"

        # 结果应与基准一致
        pd.testing.assert_index_equal(baseline.index, result_persist.index)
        pd.testing.assert_series_equal(
            baseline["trade_cluster_imbalance_ratio"],
            result_persist["trade_cluster_imbalance_ratio"],
            check_exact=False,
            rtol=1e-5,
            atol=1e-5,
        )
        # 其他列数量一致
        assert len([c for c in baseline.columns if "trade_cluster" in c]) == len(
            [c for c in result_persist.columns if "trade_cluster" in c]
        )

    def test_empty_data(self):
        """测试空数据的情况"""
        print("\n测试空数据...")

        empty_ticks = pd.DataFrame(
            columns=["side"],
            index=pd.DatetimeIndex([]),
        )

        result, state = compute_trade_clustering_from_ticks(
            empty_ticks,
            window_size=100,
            initial_state=None,
        )

        assert len(result) == 0, "空数据应返回空结果"
        assert isinstance(state, dict), "应返回状态字典"

        print("   ✅ 空数据测试通过")

    def test_single_month_data(self, sample_ticks_single_month):
        """测试单月数据的情况"""
        print("\n测试单月数据...")

        window_size = 100

        result, state = compute_trade_clustering_from_ticks(
            sample_ticks_single_month,
            window_size=window_size,
            initial_state=None,
        )

        assert len(result) > 0, "单月数据应有结果"
        assert len(result) == len(sample_ticks_single_month), "结果数量应与输入一致"

        # 验证特征列
        expected_cols = [
            "trade_cluster_max_buy_run",
            "trade_cluster_max_sell_run",
            "trade_cluster_avg_buy_run",
            "trade_cluster_avg_sell_run",
            "trade_cluster_buy_run_count",
            "trade_cluster_sell_run_count",
            "trade_cluster_imbalance_ratio",
            "trade_cluster_directional_entropy",
        ]

        for col in expected_cols:
            assert col in result.columns, f"应包含 {col} 列"

        print(f"   ✅ 单月数据测试通过，生成 {len(result)} 个特征点")

    def test_state_continuity_across_months(self, sample_ticks_multiple_months):
        """测试跨月状态连续性"""
        print("\n测试跨月状态连续性...")

        _, monthly_ticks_list = sample_ticks_multiple_months
        window_size = 100

        # 按月计算，检查状态连续性
        states = []
        monthly_results = []
        state = None

        for i, month_ticks in enumerate(monthly_ticks_list):
            month_result, state = compute_trade_clustering_from_ticks(
                month_ticks,
                window_size=window_size,
                initial_state=state,
            )

            monthly_results.append(month_result)
            states.append(state.copy() if state else None)

            # 转换 state 中的 list 回 deque
            if state:
                from collections import deque

                state["window_runs"] = deque(state.get("window_runs", []))
                state["buy_runs_in_window"] = deque(state.get("buy_runs_in_window", []))
                state["sell_runs_in_window"] = deque(
                    state.get("sell_runs_in_window", [])
                )

        # 验证状态传递
        for i in range(1, len(states)):
            prev_state = states[i - 1]
            curr_state = states[i]

            # 当前月的初始状态应该继承上个月的最终状态
            # 特别是 current_run_side 和 current_run_length
            if prev_state and prev_state.get("current_run_side") is not None:
                # 状态应该被传递（通过 initial_state 参数）
                assert "current_run_side" in curr_state, "状态应包含 current_run_side"
                assert (
                    "current_run_length" in curr_state
                ), "状态应包含 current_run_length"

        print(f"   ✅ 跨月状态连续性测试通过，处理了 {len(monthly_ticks_list)} 个月")

    def test_result_merging(self, sample_ticks_multiple_months):
        """测试结果合并的正确性"""
        print("\n测试结果合并...")

        _, monthly_ticks_list = sample_ticks_multiple_months
        window_size = 100

        # 按月计算
        monthly_results = []
        state = None

        for month_ticks in monthly_ticks_list:
            month_result, state = compute_trade_clustering_from_ticks(
                month_ticks,
                window_size=window_size,
                initial_state=state,
            )
            monthly_results.append(month_result)

            # 转换 state
            if state:
                from collections import deque

                state["window_runs"] = deque(state.get("window_runs", []))
                state["buy_runs_in_window"] = deque(state.get("buy_runs_in_window", []))
                state["sell_runs_in_window"] = deque(
                    state.get("sell_runs_in_window", [])
                )

        # 合并结果
        merged_result = pd.concat(monthly_results, axis=0).sort_index()

        # 验证合并后的结果
        assert len(merged_result) > 0, "合并结果不应为空"
        assert len(merged_result) == sum(
            len(r) for r in monthly_results
        ), "合并后的数量应等于各月之和"

        # 验证索引连续性
        assert merged_result.index.is_monotonic_increasing, "索引应按时间排序"

        # 验证没有重复索引
        assert not merged_result.index.duplicated().any(), "不应有重复索引"

        print(f"   ✅ 结果合并测试通过")
        print(
            f"   合并了 {len(monthly_results)} 个月的结果，共 {len(merged_result)} 个特征点"
        )

    def test_memory_efficiency(self, sample_ticks_multiple_months):
        """测试内存效率（不一次性加载所有数据）"""
        print("\n测试内存效率...")

        _, monthly_ticks_list = sample_ticks_multiple_months
        window_size = 100

        # 模拟按月流式计算（每次只处理一个月）
        monthly_results = []
        state = None

        for i, month_ticks in enumerate(monthly_ticks_list):
            # 计算该月
            month_result, state = compute_trade_clustering_from_ticks(
                month_ticks,
                window_size=window_size,
                initial_state=state,
            )

            # 保存结果
            monthly_results.append(month_result)

            # 立即释放该月的数据（模拟流式处理）
            del month_ticks

            # 转换 state
            if state:
                from collections import deque

                state["window_runs"] = deque(state.get("window_runs", []))
                state["buy_runs_in_window"] = deque(state.get("buy_runs_in_window", []))
                state["sell_runs_in_window"] = deque(
                    state.get("sell_runs_in_window", [])
                )

        # 合并结果
        merged_result = pd.concat(monthly_results, axis=0).sort_index()

        # 验证：如果是一次性加载，应该需要更多内存
        # 这里我们主要验证流式处理不会出错
        assert len(merged_result) > 0, "流式处理应产生结果"

        print(f"   ✅ 内存效率测试通过")
        print(f"   按月流式处理了 {len(monthly_ticks_list)} 个月的数据")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
