"""
检查特征测试覆盖情况
分析 feature_dependencies.yaml 中定义的特征是否都有对应的测试
"""

import sys
from pathlib import Path
import yaml
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_feature_dependencies() -> dict:
    """加载特征依赖配置"""
    config_path = PROJECT_ROOT / "config" / "feature_dependencies.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_all_feature_names() -> set:
    """获取所有特征名称"""
    config = load_feature_dependencies()
    features = config.get("features", {})
    return set(features.keys())


def get_test_files() -> list:
    """获取所有测试文件"""
    tests_dir = PROJECT_ROOT / "tests"
    return list(tests_dir.glob("test_*.py"))


def extract_feature_names_from_test(test_file: Path) -> set:
    """从测试文件中提取特征名称"""
    feature_names = set()

    try:
        content = test_file.read_text(encoding="utf-8")

        # 查找特征名称模式
        # 例如：wpt_price_trend, vol_raw_5, garch_volatility 等
        patterns = [
            r'["\'](\w+_\w+_\w+)["\']',  # 三个下划线的特征名
            r'["\'](\w+_\w+)["\']',  # 两个下划线的特征名
            r"compute_(\w+)",  # compute_xxx 函数名
            r"extract_(\w+)_features",  # extract_xxx_features
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # 过滤掉明显不是特征名的（如 test_, assert_ 等）
                if not match.startswith(
                    ("test_", "assert_", "self.", "import", "from")
                ):
                    feature_names.add(match)

    except Exception as e:
        print(f"   ⚠️  Error reading {test_file}: {e}")

    return feature_names


def check_test_coverage():
    """检查测试覆盖情况"""
    print("=" * 80)
    print("特征测试覆盖检查")
    print("=" * 80)

    # 获取所有特征
    all_features = get_all_feature_names()
    print(f"\n📊 总特征数: {len(all_features)}")

    # 获取所有测试文件
    test_files = get_test_files()
    print(f"📁 测试文件数: {len(test_files)}")

    # 从测试文件中提取特征名
    tested_features = set()
    test_file_features = {}

    for test_file in test_files:
        features_in_file = extract_feature_names_from_test(test_file)
        tested_features.update(features_in_file)
        if features_in_file:
            test_file_features[test_file.name] = features_in_file

    print(f"✅ 测试中提到的特征数: {len(tested_features)}")

    # 找出有测试的特征
    features_with_tests = all_features & tested_features
    print(f"✅ 有测试的特征数: {len(features_with_tests)}")

    # 找出没有测试的特征
    features_without_tests = all_features - tested_features
    print(f"❌ 没有测试的特征数: {len(features_without_tests)}")

    # 按类别分组
    config = load_feature_dependencies()
    features = config.get("features", {})

    categories = {}
    for feat_name, feat_config in features.items():
        category = feat_config.get("category", "unknown")
        if category not in categories:
            categories[category] = {"total": 0, "tested": 0, "untested": []}
        categories[category]["total"] += 1
        if feat_name in features_with_tests:
            categories[category]["tested"] += 1
        else:
            categories[category]["untested"].append(feat_name)

    print("\n" + "=" * 80)
    print("按类别统计")
    print("=" * 80)
    for category, stats in sorted(categories.items()):
        coverage = stats["tested"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"\n{category}:")
        print(f"  总数: {stats['total']}")
        print(f"  已测试: {stats['tested']} ({coverage:.1f}%)")
        print(f"  未测试: {len(stats['untested'])}")
        if stats["untested"] and len(stats["untested"]) <= 10:
            print(f"    未测试特征: {', '.join(stats['untested'])}")
        elif stats["untested"]:
            print(
                f"    未测试特征: {', '.join(stats['untested'][:10])} ... (共{len(stats['untested'])}个)"
            )

    # 显示测试文件与特征的映射
    print("\n" + "=" * 80)
    print("测试文件与特征映射（前10个）")
    print("=" * 80)
    for test_file, features in list(test_file_features.items())[:10]:
        print(f"\n{test_file}:")
        print(f"  特征: {', '.join(list(features)[:10])}")
        if len(features) > 10:
            print(f"  ... (共{len(features)}个)")

    # 关键特征检查
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
    ]

    for feat in key_features:
        if feat in all_features:
            status = "✅" if feat in features_with_tests else "❌"
            print(f"{status} {feat}")

    return {
        "total": len(all_features),
        "tested": len(features_with_tests),
        "untested": len(features_without_tests),
        "coverage": (
            len(features_with_tests) / len(all_features) * 100 if all_features else 0
        ),
    }


if __name__ == "__main__":
    stats = check_test_coverage()
    print("\n" + "=" * 80)
    print(
        f"总结: {stats['tested']}/{stats['total']} 特征有测试 ({stats['coverage']:.1f}%)"
    )
    print("=" * 80)
