"""比较 staged_2b AB 两臂的入场质量（不仅 total R）。

指标：
  - total_R / n / expectancy（R / trade）
  - win_rate（pnl_r > 0）
  - max_dd_R（按 exit_time 累计 pnl_r 的最大回撤）
  - 分离 **仅新开母仓**（is_add_position=False）的 expectancy / win_rate，代表"入场信号质量"
  - treatment 保留的首仓（symbol, entry_time）在 baseline 中的同单 pnl_r 对比，检验 arm 是否挑到更好入场
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _load_arm(root: Path, arm: str) -> pd.DataFrame:
    frames = []
    for mdir in sorted((root / arm).iterdir()):
        f = mdir / "trades.csv"
        if f.exists():
            try:
                d = pd.read_csv(f)
                d["month"] = mdir.name
                frames.append(d)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    return df


def _metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"arm": label, "n": 0}
    r = df["pnl_r"].astype(float)
    curve = r.sort_values().cumsum()  # 为 DD 用 exit_time 重排
    dfo = df.dropna(subset=["exit_time"]).sort_values("exit_time")
    eq = dfo["pnl_r"].astype(float).cumsum().to_numpy()
    if len(eq) == 0:
        dd = 0.0
    else:
        peak = -1e18
        dd = 0.0
        for v in eq:
            peak = max(peak, v)
            dd = min(dd, v - peak)
    wins = int((r > 0).sum())
    return {
        "arm": label,
        "n": int(len(df)),
        "totalR": round(float(r.sum()), 2),
        "expR": round(float(r.mean()), 3),
        "winRate": round(wins / len(df), 3),
        "maxDD_R": round(float(dd), 2),
        "stdR": round(float(r.std(ddof=0)), 3),
        "sharpe_like": round(
            float(r.mean()) / float(r.std(ddof=0)) if r.std(ddof=0) > 0 else 0.0, 3
        ),
    }


def main() -> int:
    root = Path(
        sys.argv[1] if len(sys.argv) > 1 else "results/reports/srb_fast_ab_staged_2b"
    )
    base = _load_arm(root, "baseline")
    treat = _load_arm(root, "treatment")
    if base.empty or treat.empty:
        print("缺少 baseline/treatment 数据")
        return 2

    def _show(name: str, df_b: pd.DataFrame, df_t: pd.DataFrame) -> None:
        mb = _metrics(df_b, "baseline")
        mt = _metrics(df_t, "treatment")
        print(f"\n=== {name} ===")
        cols = [
            "arm",
            "n",
            "totalR",
            "expR",
            "winRate",
            "maxDD_R",
            "stdR",
            "sharpe_like",
        ]
        print(pd.DataFrame([mb, mt])[cols].to_string(index=False))

    _show("ALL trades (含加仓)", base, treat)
    _show(
        "母仓新开（is_add_position=False）",
        base[~base["is_add_position"].astype(bool)],
        treat[~treat["is_add_position"].astype(bool)],
    )
    _show(
        "加仓（is_add_position=True）",
        base[base["is_add_position"].astype(bool)],
        treat[treat["is_add_position"].astype(bool)],
    )

    # === 关键：treatment 保留的母仓入场 vs baseline 中"同 symbol+同 entry_time"同单 ===
    # 如果 staged 2b 只是"挑子集"，那两臂同一单的 pnl_r 应相等（执行无差异）；
    # 差异来自 arm 改变了母仓后的加仓/退出路径 → 说明入场 timing 没变，只是"筛选"。
    kmb = base[~base["is_add_position"].astype(bool)][
        ["symbol", "entry_time", "pnl_r"]
    ].rename(columns={"pnl_r": "pnl_r_baseline"})
    kmt = treat[~treat["is_add_position"].astype(bool)][
        ["symbol", "entry_time", "pnl_r"]
    ].rename(columns={"pnl_r": "pnl_r_treat"})
    j = kmt.merge(kmb, on=["symbol", "entry_time"], how="left")
    matched = j.dropna(subset=["pnl_r_baseline"])
    unmatched_treat = j[j["pnl_r_baseline"].isna()]
    kept_in_base = kmb.merge(
        kmt[["symbol", "entry_time"]], on=["symbol", "entry_time"], how="inner"
    )
    blocked_by_2b = kmb.merge(
        kmt[["symbol", "entry_time"]],
        on=["symbol", "entry_time"],
        how="left",
        indicator=True,
    )
    blocked_by_2b = blocked_by_2b[blocked_by_2b["_merge"] == "left_only"]

    print("\n=== 母仓入场对齐（用 symbol+entry_time）===")
    print(
        f"baseline 母仓总数       : {len(kmb)}\n"
        f"treatment 母仓总数      : {len(kmt)}\n"
        f"两臂同单（完全重合）    : {len(matched)}\n"
        f"treatment 独有（新入场）: {len(unmatched_treat)}\n"
        f"baseline 独有（被 2b 挡）: {len(blocked_by_2b)}"
    )
    if len(matched) > 0:
        same_r = matched["pnl_r_baseline"].sum()
        print(f"同单 pnl_r 之和（baseline 视角）: {same_r:+.2f}R")
    if len(blocked_by_2b) > 0:
        blk_r = blocked_by_2b["pnl_r_baseline"].astype(float)
        print(
            f"\n== baseline 独有（staged 2b 挡掉的单）n={len(blk_r)} "
            f"totalR={blk_r.sum():+.2f}R  expR={blk_r.mean():+.3f}  "
            f"winRate={(blk_r > 0).mean():.3f}  "
            f"top10: {blk_r.sort_values(ascending=False).head(10).tolist()}\n"
            f"bot10: {blk_r.sort_values().head(10).tolist()}"
        )
    if len(unmatched_treat) > 0:
        nt = unmatched_treat["pnl_r_treat"].astype(float)
        print(
            f"\n== treatment 独有（arm 导致的新入场）n={len(nt)} "
            f"totalR={nt.sum():+.2f}R  expR={nt.mean():+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
