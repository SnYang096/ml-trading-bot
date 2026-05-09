from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml


def _pcm_ef_cutoff(
    *,
    pcm_cutoff_date: Optional[str],
    test_start: str,
    holdout_start: str,
) -> str:
    """Entry-filter cutoff used by PCM validation windows."""
    return str(pcm_cutoff_date or test_start or holdout_start)


def _pcm_ef_val_segment_end(
    *,
    pcm_cutoff_date: Optional[str],
    test_start: str,
    holdout_start: str,
    end_date: str,
) -> str:
    """Validation segment end for PCM entry-filter scans."""
    return str(pcm_cutoff_date or test_start or holdout_start or end_date)


def run_prefilter_scan_stage(
    *,
    strategy: str,
    scfg: dict,
    prefilter_gates: dict,
    config_dir: str,
    prepare_dir: str,
    log: Path,
    dry_run: bool,
    locked_prefilter_override: str,
    auto_locked_override: str,
    project_root: Path,
    prod_config_dir: str,
    run_step: Callable[..., Any],
    standardize_method_list_fn: Callable[..., List[str]],
    load_locked_rules_fn: Callable[[Path], List[Dict[str, Any]]],
    merge_locked_rules_fn: Callable[[Path, List[Dict[str, Any]]], Dict[str, int]],
) -> Dict[str, Any]:
    pf_results: Dict[str, Any] = {}
    pf_yaml: Optional[Path] = None
    enforce_pf_locked = False
    locked_prefilter_rules: List[Dict[str, Any]] = []

    if not scfg.get("has_prefilter"):
        return {
            "pf_results": pf_results,
            "pf_yaml": pf_yaml,
            "enforce_pf_locked": enforce_pf_locked,
            "locked_prefilter_rules": locked_prefilter_rules,
        }

    features_prefilter_path = Path(config_dir) / "features_prefilter.yaml"
    if not features_prefilter_path.exists():
        print(f"  ❌ Prefilter: {features_prefilter_path} 不存在, 跳过")
        return {
            "pf_results": pf_results,
            "pf_yaml": pf_yaml,
            "enforce_pf_locked": enforce_pf_locked,
            "locked_prefilter_rules": locked_prefilter_rules,
        }

    enforce_pf_locked = bool(
        prefilter_gates.get("enforce_locked_rules_in_experiment", True)
    )
    pf_locked_source = prefilter_gates.get(
        "locked_rules_source", "production_prefilter"
    )
    if enforce_pf_locked:
        if pf_locked_source != "production_prefilter":
            print(
                f"  ⚠️ Prefilter locked source 不支持: {pf_locked_source}, 已跳过强制注入"
            )
        else:
            prod_pf = Path(
                locked_prefilter_override
                or auto_locked_override
                or str(project_root / prod_config_dir / "archetypes" / "prefilter.yaml")
            )
            locked_prefilter_rules = load_locked_rules_fn(prod_pf)
            if locked_prefilter_rules:
                print(
                    f"  🔒 Prefilter locked 规则已启用: {len(locked_prefilter_rules)} 条 (source={prod_pf})"
                )
            else:
                print("  ℹ️ Prefilter locked 规则为空: 继续按默认流程")

    prefilter_cmd = [
        "python",
        "scripts/analyze_archetype_feature_stratification.py",
        "--logs",
        f"{prepare_dir}/features_labeled.parquet",
        "--strategy",
        strategy,
        "--meta-algorithm",
        "--features-prefilter",
        str(features_prefilter_path),
        "--config",
        str(config_dir),
        "--promote",
    ]

    if prefilter_gates.get("min_pass_rate"):
        prefilter_cmd += [
            "--min-prefilter-pass-rate",
            str(prefilter_gates["min_pass_rate"]),
        ]
    if prefilter_gates.get("min_rows"):
        prefilter_cmd += ["--min-prefilter-rows", str(prefilter_gates["min_rows"])]

    pf_fallbacks = prefilter_gates.get("scoring_method_fallbacks")
    if pf_fallbacks and isinstance(pf_fallbacks, list):
        pf_methods = standardize_method_list_fn(
            pf_fallbacks, default=["distribution_ks"]
        )
    elif prefilter_gates.get("scoring_method"):
        pf_methods = standardize_method_list_fn(
            [prefilter_gates["scoring_method"]], default=["distribution_ks"]
        )
    else:
        pf_methods = ["distribution_ks"]

    def append_pf_kpi_args(cmd: List[str]) -> None:
        if prefilter_gates.get("min_ks_statistic") is not None:
            cmd += [
                "--prefilter-min-ks",
                str(prefilter_gates["min_ks_statistic"]),
            ]
        if prefilter_gates.get("max_ks_pvalue") is not None:
            cmd += [
                "--prefilter-max-ks-pvalue",
                str(prefilter_gates["max_ks_pvalue"]),
            ]
        if prefilter_gates.get("min_lift") is not None:
            cmd += ["--prefilter-min-lift", str(prefilter_gates["min_lift"])]
        if prefilter_gates.get("min_positive_lift") is not None:
            cmd += [
                "--prefilter-positive-lift",
                str(prefilter_gates["min_positive_lift"]),
            ]
        if prefilter_gates.get("deny_rate_max") is not None:
            cmd += [
                "--prefilter-deny-rate-max",
                str(prefilter_gates["deny_rate_max"]),
            ]

    pf_yaml = Path(config_dir) / "archetypes" / "prefilter.yaml"
    for pf_method in pf_methods:
        if pf_yaml.exists():
            pf_yaml.unlink()
        if not dry_run and enforce_pf_locked and locked_prefilter_rules:
            m = merge_locked_rules_fn(pf_yaml, locked_prefilter_rules)
            print(
                f"   🔒 [{pf_method}] 初始注入 locked: +{m['added']} (total={m['total']})"
            )

        cmd = prefilter_cmd + ["--prefilter-scoring-method", pf_method]
        append_pf_kpi_args(cmd)
        step_name = (
            f"Prefilter Analyze [{pf_method}]"
            if len(pf_methods) > 1
            else "Prefilter Analyze"
        )
        run_step(step_name, cmd, log, dry_run=dry_run)

        if pf_yaml.exists() and not dry_run:
            try:
                if enforce_pf_locked and locked_prefilter_rules:
                    m2 = merge_locked_rules_fn(pf_yaml, locked_prefilter_rules)
                    if m2["added"] > 0:
                        print(
                            f"   🔒 [{pf_method}] 回补 locked: +{m2['added']} (total={m2['total']})"
                        )
                pf_data = yaml.safe_load(pf_yaml.read_text(encoding="utf-8")) or {}
                rules = pf_data.get("rules") or []
                n_rules = len(rules)
                tmp = pf_yaml.parent / f"prefilter_{pf_method}.yaml"
                shutil.copy(pf_yaml, tmp)
                pf_results[pf_method] = {"n_rules": n_rules, "path": tmp}
                if len(pf_methods) > 1:
                    print(f"   📊 [{pf_method}] rules={n_rules}")
            except Exception:
                pass

    # 兜底: 分析失败/未产出时，写空 prefilter 以保证后续 Gate 不因缺文件中断
    if not dry_run and pf_yaml is not None and not pf_yaml.exists():
        pf_yaml.parent.mkdir(parents=True, exist_ok=True)
        pf_yaml.write_text("rules: []\n", encoding="utf-8")
        print(f"  ⚠️ Prefilter 未产出，已回退为空规则: {pf_yaml}")

    return {
        "pf_results": pf_results,
        "pf_yaml": pf_yaml,
        "enforce_pf_locked": enforce_pf_locked,
        "locked_prefilter_rules": locked_prefilter_rules,
    }


def run_entry_filter_stage(
    *,
    strategy: str,
    scfg: dict,
    kpi_gates: dict,
    config_dir: str,
    gate_dir: str,
    strategies_root: str,
    holdout_start: str,
    test_start: str,
    end_date: str,
    log: Path,
    dry_run: bool,
    run_step: Callable[..., Any],
    parse_backtest_stdout_fn: Callable[[str], Dict[str, Any]],
    standardize_method_list_fn: Callable[..., List[str]],
) -> Optional[Dict[str, Any]]:
    ef_yaml_path = Path(config_dir) / "features_entry_filter.yaml"
    ef_gates = kpi_gates.get("entry_filter", {})
    ef_methods = standardize_method_list_fn(
        ef_gates.get("scoring_method_fallbacks"),
        default=[
            "distribution_ks",
            "mean_effect",
            "tail_bad_rate_ratio",
            "upside_positive_rate_ratio",
        ],
    )
    if not ef_yaml_path.exists():
        print("\n  ℹ️  Entry Filter: features_entry_filter.yaml 不存在, 跳过")
        return None
    if dry_run:
        return None

    print(f"\n{'='*72}")
    print(
        f"🔬 Entry Filter 多方法 Sharpe 择优 — {len(ef_methods)} methods: {ef_methods}"
    )
    print(f"{'='*72}")

    ef_sharpe: Dict[str, float] = {}
    ef_trades: Dict[str, int] = {}
    ef_n_rules: Dict[str, int] = {}
    ef_arch_dir = Path(config_dir) / "archetypes"
    ef_orig_path = ef_arch_dir / "entry_filters.yaml"
    simple_exec = scfg.get("simple_execution", {})

    for em in ef_methods:
        print(f"\n── Entry Filter method: [{em}] ──")
        ef_cmd = [
            sys.executable,
            "scripts/optimize_entry_filter_plateau.py",
            "--logs",
            f"{gate_dir}/logs_gated.parquet",
            "--strategy",
            strategy,
            "--strategies-root",
            strategies_root,
            "--meta-algorithm",
            "--features-entry-filter",
            str(ef_yaml_path),
            "--scoring-method",
            em,
            "--promote",
            "--simple-execution",
        ]
        if test_start != holdout_start:
            ef_cmd += ["--cutoff-date", test_start]
        rc_ef, out_ef = run_step(f"  EF Scan [{em}]", ef_cmd, log)
        if rc_ef != 0:
            print(f"   ❌ EF [{em}] 失败")
            ef_sharpe[em] = float("-inf")
            ef_trades[em] = 0
            ef_n_rules[em] = 0
            continue

        if ef_orig_path.exists():
            with open(ef_orig_path, "r", encoding="utf-8") as efr:
                ef_cfg = yaml.safe_load(efr) or {}
            n_ef_rules = len(
                [f for f in ef_cfg.get("filters", []) if f.get("enabled", True)]
            )
        else:
            n_ef_rules = 0
        ef_n_rules[em] = n_ef_rules

        if n_ef_rules == 0:
            ef_sharpe[em] = float("-inf")
            ef_trades[em] = 0
            continue

        ef_tmp = ef_arch_dir / f"entry_filters_cmp_{em}.yaml"
        shutil.copy(ef_orig_path, ef_tmp)

        ef_bt = [
            "python",
            "scripts/backtest_execution_layer.py",
            "--logs",
            f"{gate_dir}/logs_gated.parquet",
            "--strategy",
            strategy,
            "--strategies-root",
            strategies_root,
            "--test-start",
            holdout_start,
            "--test-end",
            test_start if test_start != holdout_start else end_date,
            "--simple-execution",
        ]
        if simple_exec.get("sl_r") is not None:
            ef_bt += ["--simple-sl", str(simple_exec["sl_r"])]
        if simple_exec.get("tp_r") is not None:
            ef_bt += ["--simple-tp", str(simple_exec["tp_r"])]
        if simple_exec.get("timeout_bars") is not None:
            ef_bt += ["--simple-timeout", str(simple_exec["timeout_bars"])]
        rc_ebt, out_ebt = run_step(f"  EF Backtest [{em}]", ef_bt, log)
        ebt_m = parse_backtest_stdout_fn(out_ebt)
        ef_sharpe[em] = ebt_m.get("sharpe_per_trade", float("-inf"))
        ef_trades[em] = ebt_m.get("total_trades", 0)
        print(
            f"   📊 EF [{em}] Sharpe={ef_sharpe[em]:+.4f}, Trades={ef_trades[em]}, Rules={n_ef_rules}"
        )

    ef_empty = ef_arch_dir / "entry_filters_cmp_empty.yaml"
    ef_empty.write_text("filters: []\ncombination_mode: or\n", encoding="utf-8")
    shutil.copy(ef_empty, ef_orig_path)
    ef_bt_empty = [
        "python",
        "scripts/backtest_execution_layer.py",
        "--logs",
        f"{gate_dir}/logs_gated.parquet",
        "--strategy",
        strategy,
        "--strategies-root",
        strategies_root,
        "--test-start",
        holdout_start,
        "--test-end",
        test_start if test_start != holdout_start else end_date,
        "--simple-execution",
    ]
    if simple_exec.get("sl_r") is not None:
        ef_bt_empty += ["--simple-sl", str(simple_exec["sl_r"])]
    if simple_exec.get("tp_r") is not None:
        ef_bt_empty += ["--simple-tp", str(simple_exec["tp_r"])]
    if simple_exec.get("timeout_bars") is not None:
        ef_bt_empty += ["--simple-timeout", str(simple_exec["timeout_bars"])]
    rc_ebt_empty, out_ebt_empty = run_step("  EF Backtest [empty]", ef_bt_empty, log)
    ebt_m_empty = parse_backtest_stdout_fn(out_ebt_empty)
    ef_sharpe["empty"] = ebt_m_empty.get("sharpe_per_trade", float("-inf"))
    ef_trades["empty"] = ebt_m_empty.get("total_trades", 0)
    ef_n_rules["empty"] = 0

    for em in ef_methods:
        if ef_n_rules.get(em, 0) == 0 and em not in ["empty"]:
            ef_sharpe[em] = ef_sharpe["empty"]
            ef_trades[em] = ef_trades["empty"]

    best_ef = max(ef_sharpe, key=lambda m: ef_sharpe[m])
    ef_tbl = []
    ef_tbl.append(f"\n{'='*72}")
    ef_tbl.append(f"  {'方法':<25} {'Sharpe':>10} {'Trades':>7} {'Rules':>6}  标记")
    ef_tbl.append(f"  {'-'*68}")
    for m in sorted(ef_sharpe, key=lambda x: -ef_sharpe[x]):
        flag = " ← 最优" if m == best_ef else ""
        s = ef_sharpe[m]
        s_str = f"{s:+.4f}" if s != float("-inf") else "  FAIL"
        ef_tbl.append(
            f"  {m:<25} {s_str:>10} {ef_trades.get(m, 0):>7} {ef_n_rules.get(m, 0):>6}{flag}"
        )
    ef_tbl.append(f"{'='*72}\n")
    ef_tbl_text = "\n".join(ef_tbl)
    print(ef_tbl_text)
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'='*72}\n")
        lf.write("🔬 Entry Filter Sharpe 对比汇总\n")
        lf.write(ef_tbl_text + "\n")

    ef_comparison = {
        "best": best_ef,
        "improvement_vs_empty": ef_sharpe.get(best_ef, float("-inf"))
        - ef_sharpe.get("empty", float("-inf")),
        "candidates": {
            m: {
                "sharpe": ef_sharpe[m],
                "trades": ef_trades.get(m, 0),
                "rules": ef_n_rules.get(m, 0),
            }
            for m in ef_sharpe
        },
    }

    min_ef_improvement = 0.02
    ef_improvement = ef_sharpe.get(best_ef, float("-inf")) - ef_sharpe.get(
        "empty", float("-inf")
    )
    if (
        best_ef != "empty"
        and ef_n_rules.get(best_ef, 0) > 0
        and ef_improvement >= min_ef_improvement
    ):
        best_ef_path = ef_arch_dir / f"entry_filters_cmp_{best_ef}.yaml"
        if best_ef_path.exists():
            shutil.copy(best_ef_path, ef_orig_path)
            print(
                f"   ✅ 最优 Entry Filter [{best_ef}] 已写入, "
                f"Rules={ef_n_rules[best_ef]}, Sharpe={ef_sharpe[best_ef]:+.4f} "
                f"(vs empty {ef_sharpe.get('empty', 0):+.4f}, 提升={ef_improvement:+.4f})"
            )
        else:
            ef_empty.rename(ef_orig_path) if not ef_orig_path.exists() else None
            print("   ⚠️  最优方法临时文件丢失, 使用空 entry filter")
    elif best_ef == "empty" or ef_improvement < min_ef_improvement:
        old_exists = ef_orig_path.exists()
        if old_exists:
            print(
                f"   ℹ️  Entry Filter 提升不足 (best={best_ef} 提升={ef_improvement:+.4f} < {min_ef_improvement}), "
                "保留原有 entry_filters.yaml 不变"
            )
        else:
            shutil.copy(ef_empty, ef_orig_path)
            print(
                f"   ℹ️  empty 最优 且无历史 entry_filters.yaml, 写入空规则 "
                f"(Sharpe={ef_sharpe.get('empty', 0):+.4f})"
            )

    for em in list(ef_methods) + ["empty"]:
        tmp = ef_arch_dir / f"entry_filters_cmp_{em}.yaml"
        if tmp.exists() and tmp != ef_orig_path:
            tmp.unlink()

    return ef_comparison
