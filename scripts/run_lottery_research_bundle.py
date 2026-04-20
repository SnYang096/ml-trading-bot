"""Run Lottery100 research chain: v4 capacity → B+ backtest → merged summary.

Reads `config/prod_train_pipeline_2h_lottery100.yaml` key `lottery_bundle` (or
`--config` to point to a YAML with the same block).

Also invoked by ``auto_research_pipeline.py`` when ``strategy_family: lottery100``.

Does **not** invoke neural training or SHAP.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_regime_diag(out_v4: Path, out_root: Path) -> None:
    """Emit regime_diag.json for pipeline KPI prefilter (bull_bar_fraction = anchor universe)."""
    diag: dict = {}
    rs = out_v4 / "regime_summary.json"
    if rs.is_file():
        summ = json.loads(rs.read_text(encoding="utf-8"))
        diag["bull_bar_fraction"] = float(summ["bull_bar_fraction"])
        diag["anchor_symbol"] = summ.get("anchor_symbol")
        diag["n_anchor_bars"] = summ.get("n_anchor_bars")
        diag["source"] = "regime_summary.json"
    else:
        try:
            import pandas as pd
        except ImportError:
            return
        if not out_v4.exists():
            return
        cands = sorted(out_v4.glob("BTCUSDT_*_H120_*bull*.parquet"))
        if not cands:
            cands = sorted(out_v4.glob("*.parquet"))
        if not cands:
            return
        df = pd.read_parquet(cands[0])
        if "bull_regime" not in df.columns:
            return
        br = df["bull_regime"]
        diag = {
            "source_parquet": str(cands[0]),
            "n_rows": int(len(df)),
            "bull_bar_fraction": float(br.astype(bool).mean()),
            "source": "parquet_fallback_bull_only_subset",
            "note": "prefer regime_summary.json from v4 for KPI prefilter band",
        }
    (out_root / "regime_diag.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(ROOT / "config/prod_train_pipeline_2h_lottery100.yaml"),
        help="YAML containing a `lottery_bundle` block",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Override lottery_bundle.output_root (e.g. experiment run_dir/reports/...)",
    )
    p.add_argument(
        "--regime-config",
        default=None,
        help="Absolute path to leverage_capacity_v4.yaml (experiment copy)",
    )
    p.add_argument(
        "--backtest-config",
        default=None,
        help="Absolute path to backtest_bplus.yaml (experiment copy)",
    )
    p.add_argument("--skip-v4", action="store_true")
    p.add_argument("--skip-bplus", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    lb = cfg.get("lottery_bundle")
    if not lb:
        raise SystemExit(f"No lottery_bundle in {cfg_path}")

    out_root = Path(
        args.output_root or lb.get("output_root", "results/lottery100_bundle")
    )
    out_v4 = out_root / "v4"
    out_b = out_root / "bplus"
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    v4_script = ROOT / lb.get(
        "analyze_script", "scripts/analyze_leverage_capacity_v4.py"
    )
    _reg = args.regime_config or lb.get(
        "regime_config",
        "config/strategies/bad-candidates/lottery100/leverage_capacity_v4.yaml",
    )
    v4_yml = Path(_reg)
    if not v4_yml.is_absolute():
        v4_yml = ROOT / v4_yml

    if not args.skip_v4:
        cmd_v4 = [
            py,
            str(v4_script),
            "--config",
            str(v4_yml),
            "--bull-only",
            "--output-dir",
            str(out_v4),
        ]
        print("Running:", " ".join(cmd_v4), flush=True)
        subprocess.run(cmd_v4, cwd=str(ROOT), check=True)
        _write_regime_diag(out_v4, out_root)

    samples_glob = str(out_v4 / "BTCUSDT_120T_H120_samples_bull_only.parquet")
    bt_script = ROOT / lb.get("backtest_script", "scripts/lottery_backtest_bplus.py")
    _btc = args.backtest_config or lb.get(
        "backtest_config",
        "config/strategies/bad-candidates/lottery100/backtest_bplus.yaml",
    )
    bt_cfg = Path(_btc)
    if not bt_cfg.is_absolute():
        bt_cfg = ROOT / bt_cfg

    summary_merge: dict = {"config": str(cfg_path), "output_root": str(out_root)}

    rd = out_root / "regime_diag.json"
    if rd.is_file():
        with open(rd, encoding="utf-8") as f:
            summary_merge["regime_diag"] = json.load(f)

    if not args.skip_bplus:
        cmd_bt = [
            py,
            str(bt_script),
            "--config",
            str(bt_cfg),
            "--samples",
            samples_glob,
            "--output-dir",
            str(out_b),
        ]
        print("Running:", " ".join(cmd_bt), flush=True)
        subprocess.run(cmd_bt, cwd=str(ROOT), check=True)

        sj = out_b / "summary.json"
        if sj.is_file():
            with open(sj, encoding="utf-8") as f:
                summary_merge["bplus_summary"] = json.load(f)

    merged_path = out_root / "summary_bundle.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(summary_merge, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {merged_path}")


if __name__ == "__main__":
    main()
