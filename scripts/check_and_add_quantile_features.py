#!/usr/bin/env python3
"""
检查所有策略的特征配置，确保使用分位数/归一化版本
"""

import yaml
from pathlib import Path

STRATEGIES = [
    "sr_reversal_rr_reg_long",
    "compression_breakout",
    "sr_breakout",
    "trend_following",
]

# 需要添加分位数版本的特征映射
QUANTILE_FEATURES = {
    "atr_f": "atr_percentile_f",  # ATR 分位数版本
}


def check_and_update_features(strategy: str):
    """检查并更新策略特征配置"""
    config_path = Path(f"config/strategies/{strategy}/features.yaml")

    if not config_path.exists():
        print(f"❌ {strategy}: features.yaml 不存在")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    requested = config.get("feature_pipeline", {}).get("requested_features", [])

    # 检查是否需要添加分位数特征
    needs_update = False
    missing_quantile = []

    for base_feature, quantile_feature in QUANTILE_FEATURES.items():
        if base_feature in requested and quantile_feature not in requested:
            missing_quantile.append((base_feature, quantile_feature))
            needs_update = True

    if needs_update:
        print(f"\n{strategy}: 需要添加分位数特征")
        for base_feature, quantile_feature in missing_quantile:
            print(f"  - 添加 {quantile_feature} (基于 {base_feature})")
            # 在 base_feature 后面添加 quantile_feature
            idx = requested.index(base_feature)
            requested.insert(idx + 1, quantile_feature)

        config["feature_pipeline"]["requested_features"] = requested

        # 保存更新
        with open(config_path, "w") as f:
            yaml.dump(
                config, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )

        print(f"  ✅ 已更新 {config_path}")
        return True
    else:
        print(f"✅ {strategy}: 特征配置正常")
        return False


def main():
    print("=" * 80)
    print("检查所有策略的特征配置（分位数/归一化版本）")
    print("=" * 80)

    updated_count = 0
    for strategy in STRATEGIES:
        if check_and_update_features(strategy):
            updated_count += 1

    print(f"\n{'='*80}")
    print(f"检查完成: {updated_count}/{len(STRATEGIES)} 个策略需要更新")
    print("=" * 80)


if __name__ == "__main__":
    main()
