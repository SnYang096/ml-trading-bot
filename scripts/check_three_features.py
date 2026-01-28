#!/usr/bin/env python3
"""
检查四个策略是否都包含三个特征：
1. vp_boundary_stability_score (来自 volume_profile_volatility_features_f)
2. sr_strength_max (来自 sr_strength_max_f 或 sr_strength_max_close_f)
3. dist_to_nearest_sr (来自 sr_strength_max_f 或 sr_strength_max_close_f)
"""

import yaml
from pathlib import Path

STRATEGIES = [
    "sr_reversal_rr_reg_long",
    "compression_breakout",
    "sr_breakout",
    "trend_following",
]


def check_features(strategy: str):
    """检查策略是否包含所需特征"""
    config_path = Path(f"config/strategies/{strategy}/features.yaml")

    if not config_path.exists():
        print(f"❌ {strategy}: features.yaml 不存在")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    requested = config.get("feature_pipeline", {}).get("requested_features", [])

    # 检查特征
    has_vp_boundary = "volume_profile_volatility_features_f" in requested
    has_sr_strength = (
        "sr_strength_max_f" in requested or "sr_strength_max_close_f" in requested
    )

    # dist_to_nearest_sr 是 sr_strength_max_f/sr_strength_max_close_f 的输出列
    # 所以只要有了 sr_strength_max_f 或 sr_strength_max_close_f，就会有 dist_to_nearest_sr
    has_dist_to_sr = has_sr_strength

    status = []
    if has_vp_boundary:
        status.append(
            "✅ vp_boundary_stability_score (volume_profile_volatility_features_f)"
        )
    else:
        status.append(
            "❌ vp_boundary_stability_score (缺少 volume_profile_volatility_features_f)"
        )

    if has_sr_strength:
        sr_feature = (
            "sr_strength_max_f"
            if "sr_strength_max_f" in requested
            else "sr_strength_max_close_f"
        )
        status.append(f"✅ sr_strength_max ({sr_feature})")
    else:
        status.append(
            "❌ sr_strength_max (缺少 sr_strength_max_f 或 sr_strength_max_close_f)"
        )

    if has_dist_to_sr:
        status.append(
            "✅ dist_to_nearest_sr (自动包含在 sr_strength_max_f/sr_strength_max_close_f)"
        )
    else:
        status.append(
            "❌ dist_to_nearest_sr (缺少 sr_strength_max_f 或 sr_strength_max_close_f)"
        )

    all_ok = has_vp_boundary and has_sr_strength and has_dist_to_sr

    print(f"\n{strategy}:")
    for s in status:
        print(f"  {s}")

    return all_ok


def main():
    print("=" * 80)
    print("检查四个策略是否都包含三个特征")
    print("=" * 80)

    all_ok = True
    for strategy in STRATEGIES:
        if not check_features(strategy):
            all_ok = False

    print(f"\n{'='*80}")
    if all_ok:
        print("✅ 所有策略都包含三个特征，无需重启")
    else:
        print("❌ 有策略缺少特征，需要添加后重启")
    print("=" * 80)

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
