#!/usr/bin/env python3
"""Bar-level add-on regime study and event-backtest validation."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("validate_add_regime_study")


def _copy_bpc_tree(dst_parent: Path) -> Path:
    src = _REPO / "config" / "strategies" / "bpc"
    dst = dst_parent / "bpc"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst_parent


def _patch_execution(dst_parent: Path, mutator: Callable[[dict], None]) -> None:
    p = dst_parent / "bpc" / "archetypes" / "execution.yaml"
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    mutator(doc)
    p.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run(cmd: List[str], *, cwd: Path) -> int:
    logger.info("run: %s", " ".join(cmd))
    return int(subprocess.run(cmd, cwd=str(cwd)).returncode)


def _run_event_backtest(
    *,
    repo: Path,
    strategies_root: Path,
    start_date: str,
    end_date: str,
    symbols_csv: str,
    data_path: Path,
    out_json: Path,
    out_csv: Path,
    cap_dir: Path,
    no_kill_switch: bool,
    inject_ml: Optional[Path],
    fee_rate: float,
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(repo / "scripts" / "event_backtest.py"),
        "--strategy",
        "bpc",
        "--strategies-root",
        str(strategies_root),
        "--data-path",
        str(data_path),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--symbols",
        symbols_csv,
        "--output",
        str(out_json),
        "--export",
        str(out_csv),
        "--capital-report",
        str(cap_dir),
        "--fee-rate",
        str(fee_rate),
    ]
    if no_kill_switch:
        cmd.append("--no-kill-switch")
    if inject_ml is not None:
        cmd.extend(["--inject-add-ml-scores", str(inject_ml)])
    return _run(cmd, cwd=repo)


def _read_metrics(out_json: Path) -> Dict[str, Any]:
    if not out_json.is_file():
        return {"error": "missing_json"}
    obj = json.loads(out_json.read_text(encoding="utf-8"))
    f = obj.get("funnel") or {}
    ap = obj.get("add_position_stats") or {}
    return {
        "n_trades": int(obj.get("n_trades", 0)),
        "total_r": float(obj.get("total_r", 0.0)),
        "max_drawdown_r": float(obj.get("max_drawdown_r", 0.0)),
        "sharpe_r": float(obj.get("sharpe_r", 0.0)),
        "reject_kill_switch": int(f.get("reject_kill_switch", 0) or 0),
        "signals_generated": int(f.get("signals_generated", 0) or 0),
        "add_count": int(ap.get("add_count", 0) or 0),
        "add_mean_r": float(ap.get("add_mean_r", 0.0) or 0.0),
        "add_win_rate": float(ap.get("add_win_rate", 0.0) or 0.0),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default="2022-01-01")
    ap.add_argument("--end-date", default="2026-05-01")
    ap.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT",
    )
    ap.add_argument("--data-path", type=Path, default=_REPO / "data" / "parquet_data")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--fee-rate", type=float, default=0.0004)
    ap.add_argument("--no-kill-switch", action="store_true")
    ap.add_argument("--quick", action="store_true", help="2024-01–2025-12 缩窗")
    ap.add_argument("--skip-panel", action="store_true")
    ap.add_argument("--skip-rule", action="store_true")
    ap.add_argument("--skip-ml", action="store_true")
    ap.add_argument("--skip-event", action="store_true")
    ap.add_argument("--timeframe", default="120T")
    ap.add_argument("--horizon-bars", type=int, default=12)
    ap.add_argument("--good-mfe-atr", type=float, default=1.0)
    ap.add_argument("--max-mae-atr", type=float, default=1.0)
    ap.add_argument("--quality-penalty", type=float, default=1.0)
    ap.add_argument("--current-chop-lte", type=float, default=0.55)
    args = ap.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    if args.quick:
        start_date = "2024-01-01"
        end_date = "2025-12-31"

    out_dir = args.out_dir or (_REPO / "results" / "add_regime_study_latest")
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "strategy_variants"
    work.mkdir(exist_ok=True)
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    panel_path = analysis_dir / "add_bar_panel.parquet"
    panel_summary_path = analysis_dir / "add_bar_panel.summary.json"
    if not args.skip_panel:
        rc = _run(
            [
                sys.executable,
                str(_REPO / "scripts" / "build_add_bar_panel.py"),
                "--strategy",
                "bpc",
                "--strategies-root",
                "config/strategies",
                "--timeframe",
                str(args.timeframe),
                "--symbols",
                ",".join(symbols),
                "--data-path",
                str(args.data_path),
                "--start-date",
                start_date,
                "--end-date",
                end_date,
                "--horizon-bars",
                str(int(args.horizon_bars)),
                "--good-mfe-atr",
                str(float(args.good_mfe_atr)),
                "--max-mae-atr",
                str(float(args.max_mae_atr)),
                "--quality-penalty",
                str(float(args.quality_penalty)),
                "--output",
                str(panel_path),
                "--summary-json",
                str(panel_summary_path),
            ],
            cwd=_REPO,
        )
        if rc != 0:
            return rc

    rule_dir = analysis_dir / "rule_search"
    rule_dir.mkdir(exist_ok=True)
    if not args.skip_rule:
        rc = _run(
            [
                sys.executable,
                str(_REPO / "scripts" / "analyze_add_bar_rules.py"),
                "--panel",
                str(panel_path),
                "--out-dir",
                str(rule_dir),
            ],
            cwd=_REPO,
        )
        if rc != 0:
            return rc

    ml_dir = analysis_dir / "ml_train"
    ml_dir.mkdir(exist_ok=True)
    if not args.skip_ml:
        rc = _run(
            [
                sys.executable,
                str(_REPO / "scripts" / "train_add_bar_ml.py"),
                "--panel",
                str(panel_path),
                "--out-dir",
                str(ml_dir),
            ],
            cwd=_REPO,
        )
        if rc != 0:
            return rc

    best_rule = {}
    best_rule_path = rule_dir / "best_rule.json"
    if best_rule_path.exists():
        best_rule = json.loads(best_rule_path.read_text(encoding="utf-8"))
    best_ml = {}
    best_ml_path = ml_dir / "best_ml_gate.json"
    if best_ml_path.exists():
        best_ml = json.loads(best_ml_path.read_text(encoding="utf-8"))
    ml_score_path = ml_dir / "add_ml_scores.parquet"

    if args.skip_event:
        print(
            json.dumps(
                {
                    "panel": str(panel_path),
                    "best_rule": best_rule,
                    "best_ml": best_ml,
                    "ml_scores": str(ml_score_path),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    scenarios: List[Tuple[str, Path, Optional[Path]]] = []

    baseline_root = work / "baseline_no_gate"
    _copy_bpc_tree(baseline_root)

    def _mut_baseline(doc: dict) -> None:
        doc.setdefault("add_position", {}).pop("add_regime_gate", None)

    _patch_execution(baseline_root, _mut_baseline)
    scenarios.append(("baseline_no_gate", baseline_root, None))

    chop_root = work / "current_chop_055"
    _copy_bpc_tree(chop_root)

    def _mut_chop(doc: dict) -> None:
        ap0 = doc.setdefault("add_position", {})
        ap0["add_regime_gate"] = {
            "enabled": True,
            "allow_if_all": [
                {
                    "feature": "bpc_semantic_chop_ts_q",
                    "lte": float(args.current_chop_lte),
                },
            ],
        }

    _patch_execution(chop_root, _mut_chop)
    scenarios.append(("current_chop_055", chop_root, None))

    if best_rule:
        rule_root = work / "best_rule_gate"
        _copy_bpc_tree(rule_root)

        def _mut_rule(doc: dict) -> None:
            ap0 = doc.setdefault("add_position", {})
            feat = str(best_rule.get("feature", ""))
            is_aligned = feat.endswith("_aligned")
            base_feat = feat[: -len("_aligned")] if is_aligned else feat
            dir_key = str(best_rule.get("direction", "lte")).strip().lower()
            cmp_key = "lte" if dir_key == "lte" else "gte"
            ap0["add_regime_gate"] = {
                "enabled": True,
                "allow_if_all": [
                    {
                        "feature": base_feat,
                        "align_with_side": bool(is_aligned),
                        cmp_key: float(best_rule.get("threshold")),
                    }
                ],
            }

        _patch_execution(rule_root, _mut_rule)
        scenarios.append(("best_rule_gate", rule_root, None))

    if (not args.skip_ml) and best_ml and ml_score_path.exists():
        ml_root = work / "best_ml_gate"
        _copy_bpc_tree(ml_root)

        def _mut_ml(doc: dict) -> None:
            ap0 = doc.setdefault("add_position", {})
            cmp_key = (
                "gte"
                if str(best_ml.get("direction", "gte")).lower() == "gte"
                else "lte"
            )
            ap0["add_regime_gate"] = {
                "enabled": True,
                "allow_if_all": [
                    {
                        "feature_by_side": dict(
                            best_ml.get(
                                "feature_by_side",
                                {
                                    "long": "add_ml_score_long",
                                    "short": "add_ml_score_short",
                                },
                            )
                        ),
                        cmp_key: float(best_ml.get("threshold")),
                    },
                ],
            }

        _patch_execution(ml_root, _mut_ml)
        scenarios.append(("best_ml_gate", ml_root, ml_score_path))

        if best_rule:
            hybrid_root = work / "hybrid_rule_ml_gate"
            _copy_bpc_tree(hybrid_root)

            def _mut_hybrid(doc: dict) -> None:
                ap0 = doc.setdefault("add_position", {})
                feat = str(best_rule.get("feature", ""))
                is_aligned = feat.endswith("_aligned")
                base_feat = feat[: -len("_aligned")] if is_aligned else feat
                dir_key = str(best_rule.get("direction", "lte")).strip().lower()
                cmp_rule = "lte" if dir_key == "lte" else "gte"
                cmp_ml = (
                    "gte"
                    if str(best_ml.get("direction", "gte")).lower() == "gte"
                    else "lte"
                )
                ap0["add_regime_gate"] = {
                    "enabled": True,
                    "allow_if_all": [
                        {
                            "feature": base_feat,
                            "align_with_side": bool(is_aligned),
                            cmp_rule: float(best_rule.get("threshold")),
                        },
                        {
                            "feature_by_side": dict(
                                best_ml.get(
                                    "feature_by_side",
                                    {
                                        "long": "add_ml_score_long",
                                        "short": "add_ml_score_short",
                                    },
                                )
                            ),
                            cmp_ml: float(best_ml.get("threshold")),
                        },
                    ],
                }

            _patch_execution(hybrid_root, _mut_hybrid)
            scenarios.append(("hybrid_rule_ml_gate", hybrid_root, ml_score_path))

    rows_out: List[Dict[str, Any]] = []
    for name, root, inj in scenarios:
        oj = out_dir / f"event_backtest_{name}.json"
        oc = out_dir / f"event_trades_{name}.csv"
        rc = _run_event_backtest(
            repo=_REPO,
            strategies_root=root,
            start_date=start_date,
            end_date=end_date,
            symbols_csv=",".join(symbols),
            data_path=args.data_path,
            out_json=oj,
            out_csv=oc,
            cap_dir=oj.parent,
            no_kill_switch=bool(args.no_kill_switch),
            inject_ml=inj,
            fee_rate=float(args.fee_rate),
        )
        m = _read_metrics(oj)
        m.update({"scenario": name, "exit_code": rc})
        rows_out.append(m)
        print(json.dumps(m, indent=2, ensure_ascii=False))

    summary_path = out_dir / "add_regime_study_summary.json"
    summary_path.write_text(
        json.dumps(rows_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Add Regime Study Report",
        "",
        f"- Window: `{start_date}` → `{end_date}`",
        f"- Symbols: `{','.join(symbols)}`",
        "",
        "## Scenarios",
    ]
    for r in rows_out:
        lines.append(
            f"- `{r.get('scenario')}`: total_r={r.get('total_r'):.4f}, "
            f"max_drawdown_r={r.get('max_drawdown_r'):.4f}, n_trades={r.get('n_trades')}, "
            f"reject_kill_switch={r.get('reject_kill_switch')}, add_count={r.get('add_count')}"
        )
    rep = out_dir / "report.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote report → %s", rep)
    print(f"\nWrote summary → {summary_path}")
    return 0 if all(r.get("exit_code", 1) == 0 for r in rows_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
