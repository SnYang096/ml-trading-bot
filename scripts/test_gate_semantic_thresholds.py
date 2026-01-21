#!/usr/bin/env python3
"""
测试修正后的 Gate 语义分数阈值逻辑（不需要 FeatureStore）。

直接使用现有的 logs 和 regime 文件，只测试语义分数阈值过滤。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Test Gate semantic score thresholds.")
    p.add_argument("--logs", required=True, help="logs_3action parquet")
    p.add_argument("--regime", required=True, help="physics_regime parquet")
    p.add_argument(
        "--semantic-thresholds", required=True, help="semantic_score_floors.json"
    )
    p.add_argument("--output", required=True, help="Output parquet with gate results")
    args = p.parse_args()

    # Load data
    logs_df = pd.read_parquet(args.logs)
    regime_df = pd.read_parquet(args.regime)

    # Load thresholds
    with open(args.semantic_thresholds, "r") as f:
        thresholds = json.load(f)

    print("=" * 80)
    print("测试 Gate 语义分数阈值逻辑")
    print("=" * 80)
    print(f"\n输入:")
    print(f"  Logs: {len(logs_df)} 行")
    print(f"  Regime: {len(regime_df)} 行")
    print(f"\n阈值:")
    print(f"  TC p95 (上限): {thresholds.get('tc_semantic_score_p95', 'N/A')}")
    print(f"  TE p10 (下限): {thresholds.get('te_semantic_score_p10', 'N/A')}")

    # Merge (handle case where logs already has regime column)
    regime_cols = ["symbol", "timestamp", "tc_semantic_score", "te_semantic_score"]
    if "regime" not in logs_df.columns:
        regime_cols.append("regime")

    merged = logs_df.merge(
        regime_df[[c for c in regime_cols if c in regime_df.columns]],
        on=["symbol", "timestamp"],
        how="left",
        suffixes=("", "_regime"),
    )

    # Use regime from regime_df if available
    if "regime_regime" in merged.columns:
        merged["regime"] = merged["regime_regime"].fillna(merged.get("regime", ""))
        merged = merged.drop(columns=["regime_regime"])

    print(f"\n合并后: {len(merged)} 行")

    # Apply semantic score thresholds
    gate_ok = []
    gate_reasons = []
    semantic_veto_count = 0

    tc_ceiling = thresholds.get("tc_semantic_score_p95")
    te_floor = thresholds.get("te_semantic_score_p10")

    for _, row in merged.iterrows():
        mode = str(row.get("mode", "NO_TRADE")).upper()
        regime = str(row.get("regime", "NO_TRADE")).upper()

        if mode == "NO_TRADE":
            gate_ok.append(True)
            gate_reasons.append("")
            continue

        # Apply semantic score thresholds
        vetoed = False
        reason = ""

        if regime == "TC_REGIME" and tc_ceiling is not None:
            score = row.get("tc_semantic_score")
            if pd.notna(score) and float(score) > float(tc_ceiling):
                vetoed = True
                reason = "tc_semantic_ceiling"
                semantic_veto_count += 1

        elif regime == "TE_REGIME" and te_floor is not None:
            score = row.get("te_semantic_score")
            if pd.notna(score) and float(score) < float(te_floor):
                vetoed = True
                reason = "te_semantic_floor"
                semantic_veto_count += 1

        gate_ok.append(not vetoed)
        gate_reasons.append(reason)

    merged["gate_ok_semantic"] = gate_ok
    merged["gate_reason_semantic"] = gate_reasons

    print(f"\n语义分数阈值过滤结果:")
    print(f"  Veto 数量: {semantic_veto_count}")
    print(f"  通过数量: {sum(gate_ok)}")

    # Statistics by regime
    print(f"\n按 Regime 统计:")
    for regime in ["TC_REGIME", "TE_REGIME"]:
        regime_df_sub = merged[merged["regime"] == regime]
        if len(regime_df_sub) > 0:
            vetoed = (~regime_df_sub["gate_ok_semantic"]).sum()
            passed = regime_df_sub["gate_ok_semantic"].sum()
            print(f"  {regime}:")
            print(f"    总行数: {len(regime_df_sub)}")
            print(f"    通过: {passed}")
            print(f"    Veto: {vetoed}")

            # Show score distribution
            if regime == "TC_REGIME":
                scores = pd.to_numeric(
                    regime_df_sub["tc_semantic_score"], errors="coerce"
                )
                vetoed_scores = pd.to_numeric(
                    regime_df_sub[~regime_df_sub["gate_ok_semantic"]][
                        "tc_semantic_score"
                    ],
                    errors="coerce",
                )
                passed_scores = pd.to_numeric(
                    regime_df_sub[regime_df_sub["gate_ok_semantic"]][
                        "tc_semantic_score"
                    ],
                    errors="coerce",
                )
            else:
                scores = pd.to_numeric(
                    regime_df_sub["te_semantic_score"], errors="coerce"
                )
                vetoed_scores = pd.to_numeric(
                    regime_df_sub[~regime_df_sub["gate_ok_semantic"]][
                        "te_semantic_score"
                    ],
                    errors="coerce",
                )
                passed_scores = pd.to_numeric(
                    regime_df_sub[regime_df_sub["gate_ok_semantic"]][
                        "te_semantic_score"
                    ],
                    errors="coerce",
                )

            if scores.notna().any():
                print(f"    分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
            if vetoed_scores.notna().any():
                print(
                    f"    Veto 分数范围: [{vetoed_scores.min():.4f}, {vetoed_scores.max():.4f}]"
                )
            if passed_scores.notna().any():
                print(
                    f"    通过分数范围: [{passed_scores.min():.4f}, {passed_scores.max():.4f}]"
                )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print(f"\n✅ 结果已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
