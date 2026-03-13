#!/usr/bin/env python3
"""
就地 patch bpc_wpt_energy_low 和 bpc_pre_breakout_score。

无需重建整个 layer：直接用 parquet 里已有的 bb_width_normalized_pct 重算这两列。

用法:
    python3 scripts/patch_bpc_wpt_energy_low.py <layer_name>
    python3 scripts/patch_bpc_wpt_energy_low.py features_me_60T_87c2c8c784
    python3 scripts/patch_bpc_wpt_energy_low.py features_bpc_240T_912b2f17be

加 --dry-run 只打印统计不写文件。
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def recompute_wpt_low(
    bb_width_normalized_pct: pd.Series,
    pct_window: int = 100,
) -> pd.Series:
    """
    bb_width_normalized_pct 已是 [0,1] 百分位。
    逻辑与 compute_bpc_compression_state_from_series 完全一致:
      bb_median = rolling(pct_window).median()
      bb_below  = (bb_w < bb_median).astype(float)
      wpt_low   = bb_below.rolling(5).mean().clip(0, 1)
    """
    bb_w = (
        pd.to_numeric(bb_width_normalized_pct, errors="coerce").fillna(0.5).clip(0, 1)
    )
    bb_median = bb_w.rolling(pct_window, min_periods=20).median().fillna(0.5)
    bb_below = (bb_w < bb_median).astype(float)
    wpt_low = bb_below.rolling(5, min_periods=1).mean().clip(0, 1)
    return wpt_low


def patch_layer(layer: str, fs_root: str = "feature_store", dry_run: bool = False):
    root = Path(fs_root)
    layer_dir = root / layer

    if not layer_dir.exists():
        print(f"❌ Layer not found: {layer_dir}")
        sys.exit(1)

    # 按 symbol 分组处理，保持时间顺序（需要 warmup）
    all_paths = sorted(glob.glob(str(layer_dir / "*/*/[0-9]*.parquet")))
    by_sym: dict[str, list[Path]] = defaultdict(list)
    for p in all_paths:
        sym = Path(p).parts[-3]
        by_sym[sym].append(Path(p))
    for sym in by_sym:
        by_sym[sym].sort()

    total_files = sum(len(v) for v in by_sym.values())
    print(f"Layer : {layer}")
    print(f"Files : {total_files}  Symbols: {sorted(by_sym.keys())}")
    if dry_run:
        print("🔍 DRY-RUN mode — no files will be written\n")

    WARMUP_ROWS = 200  # 100-bar rolling median + 5-bar rolling mean + buffer

    patched = 0
    skipped = 0

    for sym, paths in sorted(by_sym.items()):
        prev_tail: pd.DataFrame | None = None

        for path in paths:
            month = path.stem
            df = pd.read_parquet(path)

            # 检查所需列
            if "bb_width_normalized_pct" not in df.columns:
                print(f"  ⚠️  {sym}/{month}: bb_width_normalized_pct missing, skip")
                skipped += 1
                continue

            if "bpc_wpt_energy_low" not in df.columns:
                print(f"  ⚠️  {sym}/{month}: bpc_wpt_energy_low missing, skip")
                skipped += 1
                continue

            # 检查是否已经是常量 0.5（需要修复）
            current_nu = df["bpc_wpt_energy_low"].dropna().nunique()
            current_val = df["bpc_wpt_energy_low"].iloc[0] if len(df) > 0 else None
            needs_fix = current_nu == 1 and abs(float(current_val) - 0.5) < 1e-6

            if not needs_fix:
                print(f"  ✅ {sym}/{month}: already OK (nunique={current_nu}), skip")
                skipped += 1
                prev_tail = df.tail(WARMUP_ROWS)
                continue

            # 拼接 warmup（前一个月的尾部）+ 当前月
            if prev_tail is not None and len(prev_tail) > 0:
                combined = pd.concat(
                    [
                        prev_tail[["bb_width_normalized_pct"]],
                        df[["bb_width_normalized_pct"]],
                    ]
                )
            else:
                combined = df[["bb_width_normalized_pct"]]

            # 重算 wpt_low（在完整序列上，带 warmup 上下文）
            wpt_combined = recompute_wpt_low(combined["bb_width_normalized_pct"])

            # 只取当前月的行
            wpt_new = wpt_combined.loc[df.index]

            # 重算 pre_breakout_score（用已有的其他压缩分量）
            score_cols = {
                "bpc_vol_compression_state": 0.25,
                "bpc_bb_compression_state": 0.45,
                "bpc_garch_compression": 0.15,
            }
            pre_score = pd.Series(0.0, index=df.index)
            for col, w in score_cols.items():
                if col in df.columns:
                    pre_score += pd.to_numeric(df[col], errors="coerce").fillna(0.5) * w
            pre_score += wpt_new * 0.15
            pre_score = pre_score.clip(0, 1)

            if dry_run:
                print(
                    f"  🔍 {sym}/{month}: wpt_low nunique {current_nu}→{wpt_new.nunique()} "
                    f"  pre_score nunique {df['bpc_pre_breakout_score'].dropna().nunique() if 'bpc_pre_breakout_score' in df.columns else '?'}"
                    f"→{pre_score.nunique()}"
                )
            else:
                df["bpc_wpt_energy_low"] = wpt_new
                if "bpc_pre_breakout_score" in df.columns:
                    df["bpc_pre_breakout_score"] = pre_score
                df.to_parquet(path, index=True)
                print(
                    f"  ✏️  {sym}/{month}: patched wpt_low nunique→{wpt_new.nunique()} "
                    f"pre_score nunique→{pre_score.nunique()}"
                )

            patched += 1
            prev_tail = df.tail(WARMUP_ROWS)

    print(f"\n{'DRY-RUN' if dry_run else 'Done'}: patched={patched}  skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patch bpc_wpt_energy_low in FeatureStore layer"
    )
    parser.add_argument("layer", help="Layer name, e.g. features_me_60T_87c2c8c784")
    parser.add_argument(
        "--fs-root", default="feature_store", help="FeatureStore root dir"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print stats without writing"
    )
    args = parser.parse_args()

    patch_layer(args.layer, fs_root=args.fs_root, dry_run=args.dry_run)
