"""
检查特征质量：未来数据泄露、多资产归一化、特征有效性

检查内容：
1. 未来数据泄露检查（look-ahead bias）
2. 多资产归一化支持
3. 特征有效性验证
4. 模拟数据测试
"""

import sys
from pathlib import Path
import yaml
import re
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_feature_dependencies() -> dict:
    """加载特征依赖配置"""
    config_path = PROJECT_ROOT / "config" / "feature_dependencies.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_test_files() -> List[Path]:
    """获取所有测试文件"""
    tests_dir = PROJECT_ROOT / "tests"
    return list(tests_dir.glob("test_*.py"))


def check_future_data_leakage() -> Dict[str, List[str]]:
    """
    检查测试中是否有未来数据泄露验证

    Returns:
        dict: {
            "has_tests": [特征名列表],
            "missing_tests": [特征名列表]
        }
    """
    test_files = get_test_files()

    # 查找包含未来数据泄露测试的文件
    future_leak_keywords = [
        "future.*leak",
        "look.*ahead",
        "lookahead",
        "causality.*no.*future",
        "test.*causality",
        "no.*future.*information",
        "shift.*1",
        "rolling.*window",
        "历史数据",
        "无未来信息",
        "仅依赖历史数据",
        "不使用.*未来",
    ]

    features_with_future_test = set()
    features_without_future_test = set()

    config = load_feature_dependencies()
    all_features = set(config.get("features", {}).keys())

    for test_file in test_files:
        try:
            content = test_file.read_text(encoding="utf-8")

            # 检查是否包含未来数据泄露测试
            has_future_test = any(
                re.search(keyword, content, re.IGNORECASE)
                for keyword in future_leak_keywords
            )

            if has_future_test:
                # 提取测试文件中提到的特征
                # 方法1: 从函数调用中提取特征名
                # extract_hurst_features -> hurst_features
                # extract_garch_features -> garch_features
                # extract_hilbert_features -> hilbert_advanced (需要特殊处理)
                func_patterns = [
                    r"extract_(\w+)_features",
                    r"compute_(\w+)",
                ]

                for pattern in func_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        # 直接匹配特征名
                        if match in all_features:
                            features_with_future_test.add(match)
                        # 尝试添加 _features 后缀
                        feat_with_suffix = f"{match}_features"
                        if feat_with_suffix in all_features:
                            features_with_future_test.add(feat_with_suffix)
                        # 特殊处理：hilbert -> hilbert_advanced
                        if match == "hilbert":
                            if "hilbert_advanced" in all_features:
                                features_with_future_test.add("hilbert_advanced")

                # 方法2: 直接匹配特征名（在字符串中）
                for feat_name in all_features:
                    # 检查特征名是否在测试文件中出现
                    if re.search(rf"\b{re.escape(feat_name)}\b", content):
                        features_with_future_test.add(feat_name)
        except Exception as e:
            print(f"   ⚠️  Error reading {test_file}: {e}")

    features_without_future_test = all_features - features_with_future_test

    return {
        "has_tests": sorted(features_with_future_test),
        "missing_tests": sorted(features_without_future_test),
    }


def check_multi_asset_normalization() -> Dict[str, List[str]]:
    """
    检查测试中是否有多资产归一化验证

    Returns:
        dict: {
            "has_tests": [特征名列表],
            "missing_tests": [特征名列表]
        }
    """
    test_files = get_test_files()

    multi_asset_keywords = [
        "multi.*asset",
        "cross.*asset",
        "跨.*资产",
        "跨.*品种",
        "group.*normalize",
        "_symbol",
        "normalize.*group",
        "rank.*transform",
    ]

    features_with_multi_asset_test = set()

    config = load_feature_dependencies()
    all_features = set(config.get("features", {}).keys())

    for test_file in test_files:
        try:
            content = test_file.read_text(encoding="utf-8")

            # 检查是否包含多资产归一化测试
            has_multi_asset_test = any(
                re.search(keyword, content, re.IGNORECASE)
                for keyword in multi_asset_keywords
            )

            if has_multi_asset_test:
                # 提取测试文件中提到的特征
                feature_patterns = [
                    r"extract_(\w+)_features",
                    r"compute_(\w+)",
                    r"(\w+)_features",
                ]

                for pattern in feature_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if match in all_features:
                            features_with_multi_asset_test.add(match)
        except Exception as e:
            print(f"   ⚠️  Error reading {test_file}: {e}")

    all_features_without = all_features - features_with_multi_asset_test

    return {
        "has_tests": sorted(features_with_multi_asset_test),
        "missing_tests": sorted(all_features_without),
    }


def check_simulated_data_tests() -> Dict[str, List[str]]:
    """
    检查测试中是否使用模拟数据

    Returns:
        dict: {
            "has_tests": [特征名列表],
            "missing_tests": [特征名列表]
        }
    """
    test_files = get_test_files()

    simulated_keywords = [
        "simulate",
        "mock",
        "synthetic",
        "random",
        "np.random",
        "create.*test.*data",
        "模拟数据",
    ]

    features_with_simulated_test = set()

    config = load_feature_dependencies()
    all_features = set(config.get("features", {}).keys())

    for test_file in test_files:
        try:
            content = test_file.read_text(encoding="utf-8")

            # 检查是否使用模拟数据
            has_simulated = any(
                re.search(keyword, content, re.IGNORECASE)
                for keyword in simulated_keywords
            )

            if has_simulated:
                # 提取测试文件中提到的特征
                feature_patterns = [
                    r"extract_(\w+)_features",
                    r"compute_(\w+)",
                    r"(\w+)_features",
                ]

                for pattern in feature_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if match in all_features:
                            features_with_simulated_test.add(match)
        except Exception as e:
            print(f"   ⚠️  Error reading {test_file}: {e}")

    all_features_without = all_features - features_with_simulated_test

    return {
        "has_tests": sorted(features_with_simulated_test),
        "missing_tests": sorted(all_features_without),
    }


def main():
    """主函数"""
    print("=" * 80)
    print("特征质量检查")
    print("=" * 80)

    # 1. 检查未来数据泄露测试
    print("\n📊 1. 未来数据泄露检查")
    print("-" * 80)
    future_leak = check_future_data_leakage()
    print(f"   ✅ 有未来数据泄露测试的特征: {len(future_leak['has_tests'])}")
    print(f"   ❌ 缺少未来数据泄露测试的特征: {len(future_leak['missing_tests'])}")

    if future_leak["has_tests"]:
        print(f"\n   有测试的特征（前10个）:")
        for feat in future_leak["has_tests"][:10]:
            print(f"     - {feat}")

    # 2. 检查多资产归一化测试
    print("\n📊 2. 多资产归一化检查")
    print("-" * 80)
    multi_asset = check_multi_asset_normalization()
    print(f"   ✅ 有多资产归一化测试的特征: {len(multi_asset['has_tests'])}")
    print(f"   ❌ 缺少多资产归一化测试的特征: {len(multi_asset['missing_tests'])}")

    if multi_asset["has_tests"]:
        print(f"\n   有测试的特征（前10个）:")
        for feat in multi_asset["has_tests"][:10]:
            print(f"     - {feat}")

    # 3. 检查模拟数据测试
    print("\n📊 3. 模拟数据测试检查")
    print("-" * 80)
    simulated = check_simulated_data_tests()
    print(f"   ✅ 有模拟数据测试的特征: {len(simulated['has_tests'])}")
    print(f"   ❌ 缺少模拟数据测试的特征: {len(simulated['missing_tests'])}")

    if simulated["has_tests"]:
        print(f"\n   有测试的特征（前10个）:")
        for feat in simulated["has_tests"][:10]:
            print(f"     - {feat}")

    # 4. 综合统计
    print("\n" + "=" * 80)
    print("综合统计")
    print("=" * 80)

    config = load_feature_dependencies()
    all_features = set(config.get("features", {}).keys())
    total = len(all_features)

    # 计算覆盖率
    future_coverage = len(future_leak["has_tests"]) / total * 100 if total > 0 else 0
    multi_asset_coverage = (
        len(multi_asset["has_tests"]) / total * 100 if total > 0 else 0
    )
    simulated_coverage = len(simulated["has_tests"]) / total * 100 if total > 0 else 0

    print(f"\n总特征数: {total}")
    print(f"未来数据泄露测试覆盖率: {future_coverage:.1f}%")
    print(f"多资产归一化测试覆盖率: {multi_asset_coverage:.1f}%")
    print(f"模拟数据测试覆盖率: {simulated_coverage:.1f}%")

    # 找出完全没有测试的关键特征
    print("\n" + "=" * 80)
    print("关键特征测试状态")
    print("=" * 80)

    key_features = [
        "wpt_features",
        "extended_volatility_features",
        "garch_features",
        "vpin_features",
        "spectrum_features",
        "hilbert_advanced",
        "evt_features",
        "volume_profile_volatility_features",
    ]

    for feat in key_features:
        has_future = feat in future_leak["has_tests"]
        has_multi = feat in multi_asset["has_tests"]
        has_sim = feat in simulated["has_tests"]

        status = []
        if has_future:
            status.append("未来数据✓")
        if has_multi:
            status.append("多资产✓")
        if has_sim:
            status.append("模拟数据✓")

        if status:
            print(f"   ✅ {feat}: {', '.join(status)}")
        else:
            print(f"   ❌ {feat}: 缺少测试")


if __name__ == "__main__":
    main()
