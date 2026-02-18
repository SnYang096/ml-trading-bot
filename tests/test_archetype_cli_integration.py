#!/usr/bin/env python3
"""
CLI Integration Test for Archetype Module

测试 mlbot gate apply-archetype 命令
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_test_logs():
    """创建测试用的 logs parquet 文件"""
    np.random.seed(42)
    n_rows = 100

    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n_rows,
            "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1min"),
            "price": np.random.uniform(40000, 45000, n_rows),
            "volume": np.random.uniform(1, 100, n_rows),
            "signal": np.random.choice(["buy", "sell", "hold"], n_rows),
            "confidence": np.random.uniform(0.5, 1.0, n_rows),
            # Gate 相关特征
            "vpin_percentile": np.random.uniform(0, 1, n_rows),
            "direction_commitment_pct": np.random.uniform(0, 1, n_rows),
            "volume_expansion_pct": np.random.uniform(0, 2, n_rows),
            "price_zone_score": np.random.uniform(0, 1, n_rows),
            "pullback_depth_pct": np.random.uniform(0, 1, n_rows),
            # Evidence 相关特征
            "sr_strength_max": np.random.uniform(0, 1, n_rows),
            "vpin_ma20": np.random.uniform(0, 1, n_rows),
            "vol_slope_20": np.random.uniform(-0.5, 0.5, n_rows),
        }
    )

    return df


def test_cli_gate_apply_archetype():
    """测试 CLI gate apply-archetype 命令参数解析"""
    from click.testing import CliRunner
    from src.cli.main import cli

    runner = CliRunner()

    # 测试 help 命令 - 验证命令参数结构
    result = runner.invoke(cli, ["gate", "apply-archetype", "--help"])

    print(f"\n=== CLI Help Output ===")
    print(result.output)

    # 验证关键参数存在
    assert "--strategy" in result.output, "Missing --strategy option"
    assert "--strategies-root" in result.output, "Missing --strategies-root option"
    assert "--logs" in result.output, "Missing --logs option"
    assert "--out" in result.output, "Missing --out option"
    assert (
        "--features-store-layer" in result.output
    ), "Missing --features-store-layer option"

    print("\n✓ All required CLI options are present")

    # 测试缺少必要参数时的错误处理
    result_missing = runner.invoke(cli, ["gate", "apply-archetype"])
    assert result_missing.exit_code != 0, "Should fail without required options"
    print("✓ Properly rejects missing required options")

    return True


def format_when_condition(when: dict) -> str:
    """格式化 when 条件"""
    if "any_of" in when:
        return f"any_of({len(when['any_of'])} conditions)"
    elif "all_of" in when:
        return f"all_of({len(when['all_of'])} conditions)"
    else:
        # 直接的特征条件
        feature_names = list(when.keys())
        if feature_names:
            return f"{feature_names[0]}"
        return "unknown"


def test_archetype_loading_from_cli():
    """测试从 CLI 上下文加载 archetype"""
    from src.time_series_model.archetype import load_strategy_archetype

    # 测试加载 BPC 策略
    arch = load_strategy_archetype("bpc")

    print("\n=== BPC Archetype Configuration ===")
    print(f"Strategy Name: {arch.name}")
    print(f"\nGate Configuration:")
    print(f"  - Hard Gates: {len(arch.gate.hard_gates)}")
    for g in arch.gate.hard_gates:
        when_str = format_when_condition(g.when)
        print(f"    - {g.id}: {g.tag} ({when_str})")

    print(f"\nEvidence Configuration:")
    print(f"  - Features: {len(arch.evidence.features)}")
    for ef in arch.evidence.features[:3]:  # 只显示前3个
        print(f"    - {ef.id}: {ef.feature}")
    if len(arch.evidence.features) > 3:
        print(f"    ... and {len(arch.evidence.features) - 3} more")

    print(f"\nExecution Configuration:")
    print(f"  - Stop Loss R: {arch.execution.stop_loss_r}")
    print(f"  - Take Profit R: {arch.execution.take_profit_r}")
    print(f"  - Direction Source: {arch.execution.direction_source}")

    print(f"\nBackward Compatibility Check:")
    print(f"  - gate_rules: {type(arch.gate_rules).__name__}")
    print(f"  - when_then_rules count: {len(arch.when_then_rules)}")
    print(f"  - default_action: {arch.default_action}")
    print(f"  - direction_policy: {type(arch.direction_policy).__name__}")

    return True


def test_gate_apply_with_mock_data():
    """测试 Gate 应用流程"""
    from src.time_series_model.archetype import load_strategy_archetype

    arch = load_strategy_archetype("bpc")

    # 模拟特征数据
    test_cases = [
        {
            "name": "Normal Trade",
            "features": {
                "vpin_percentile": 0.5,
                "direction_commitment_pct": 0.8,
                "volume_expansion_pct": 0.7,
                "price_zone_score": 0.5,
                "pullback_depth_pct": 0.4,
            },
        },
        {
            "name": "High VPIN (should be filtered)",
            "features": {
                "vpin_percentile": 0.95,
                "direction_commitment_pct": 0.8,
                "volume_expansion_pct": 0.7,
                "price_zone_score": 0.5,
                "pullback_depth_pct": 0.4,
            },
        },
        {
            "name": "Low Commitment (should be filtered)",
            "features": {
                "vpin_percentile": 0.5,
                "direction_commitment_pct": 0.2,
                "volume_expansion_pct": 0.7,
                "price_zone_score": 0.5,
                "pullback_depth_pct": 0.4,
            },
        },
    ]

    print("\n=== Gate Application Test Cases ===")
    for tc in test_cases:
        passed, reasons, weight = arch.apply_gate(tc["features"])
        print(f"\n{tc['name']}:")
        print(f"  - Passed: {passed}")
        print(f"  - Weight: {weight:.2f}")
        if reasons:
            print(f"  - Reasons: {reasons}")

    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Running Archetype CLI Integration Tests")
    print("=" * 60)

    tests = [
        ("Archetype Loading", test_archetype_loading_from_cli),
        ("Gate Application", test_gate_apply_with_mock_data),
        ("CLI Command", test_cli_gate_apply_archetype),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print("=" * 60)
        try:
            success = test_fn()
            results.append((name, success))
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"\nResult: {status}")
        except Exception as e:
            results.append((name, False))
            print(f"\nResult: ✗ ERROR - {e}")
            import traceback

            traceback.print_exc()

    # 汇总
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, s in results if s)
    total = len(results)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    print(f"\nTotal: {passed}/{total} passed")

    return all(s for _, s in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
