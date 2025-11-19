#!/usr/bin/env python3
"""
移除可疑的高相关特征，用于验证数据泄露
"""

import json
from pathlib import Path

# 根据数据泄露检测结果，移除相关性 > 0.1 的特征
SUSPICIOUS_FEATURES = {
    "dl_seq_f43": 0.1840,
    "atr_zscore_w288": 0.1517,
    "volatility_zscore_w288": 0.1455,
    "atr_percentile": 0.1417,
    "dl_seq_f55": -0.1411,
    "dl_seq_f18": -0.1372,
    "dl_seq_f38": -0.1357,
    "atr_compression_ratio": -0.1331,
    "dl_seq_f7": 0.1294,
    "volatility_zscore_w500": 0.1250,
    # 还有其他 14 个特征，但先移除这 10 个最可疑的
}


def remove_suspicious_features(input_path: str, output_path: str):
    """从 top_factors.json 中移除可疑特征"""
    with open(input_path, "r") as f:
        data = json.load(f)

    original_count = len(data["top_factors"])
    suspicious_names = set(SUSPICIOUS_FEATURES.keys())

    # 移除可疑特征
    data["top_factors"] = [
        factor
        for factor in data["top_factors"]
        if factor["name"] not in suspicious_names
    ]

    removed_count = original_count - len(data["top_factors"])

    # 更新 count
    data["count"] = len(data["top_factors"])

    # 保存
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Removed {removed_count} suspicious features")
    print(f"   Original: {original_count} features")
    print(f"   Remaining: {len(data['top_factors'])} features")
    print(f"   Removed features: {sorted(suspicious_names)}")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    import sys

    input_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "results/feature_evaluation/top_factors.json"
    )
    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "results/feature_evaluation/top_factors_clean.json"
    )

    remove_suspicious_features(input_path, output_path)
