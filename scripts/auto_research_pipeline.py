#!/usr/bin/env python3
"""
自动研究流水线 — 一键执行全流程训练 + 结果快照 + 对比决策

功能:
  1. 自动检测最新数据日期, 计算 holdout 窗口 (end - 14 个月)
  2. 下载 + 转换最新月度 aggTrades 数据 (增量, 已有跳过)
  3. 按策略执行完整训练链: DataDownload → FeatureStore → Prepare
     → Prefilter → Direction → Gate → Execution → Backtest
  4. 所有阈值优化步骤带 --promote, 写入实验目录 (不覆盖生产 config)
  5. 保存结构化 report.json 到 results/research_history/{strategy}/{timestamp}/
  6. 与上次研究结果对比, 输出确定性决策: ADOPT / KEEP / ALERT
  7. ADOPT 时自动将实验 archetypes 复制回生产 config

  实验目录隔离:
    每次运行自动复制 config/strategies/{strategy}/ 到实验工作区,
    所有 --promote 写入实验副本, 生产 config 仅在 ADOPT 时更新。

用法:
    # 单策略
    python scripts/auto_research_pipeline.py --strategy fer

    # 全部策略
    python scripts/auto_research_pipeline.py --all

    # 指定 end-date (跳过自动检测)
    python scripts/auto_research_pipeline.py --strategy bpc --end-date 2026-01-01

    # 只运行对比 (不重新训练)
    python scripts/auto_research_pipeline.py --strategy fer --compare-only

    # dry-run (打印命令但不执行)
    python scripts/auto_research_pipeline.py --strategy fer --dry-run

    # 列出历史实验
    python scripts/auto_research_pipeline.py --strategy fer --list

    # 手动采纳某次实验
    python scripts/auto_research_pipeline.py --strategy fer --adopt 20260222_120000

    # 对比两次实验的 archetypes 差异
    python scripts/auto_research_pipeline.py --strategy fer --diff 20260220_100000 20260222_120000
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stat_method_registry import standardize_method_list
from scripts.locked_prefilter_utils import (
    load_locked_prefilter_rules,
    merge_locked_prefilter_rules,
)
from scripts.locked_gate_utils import (
    load_locked_gate_rules,
    merge_locked_gate_rules,
)
from scripts.locked_entry_filter_utils import (
    load_locked_entry_filters,
    merge_locked_entry_filters,
)
from scripts.pipeline import config as pipeline_config
from scripts.pipeline import cli as pipeline_cli
from scripts.pipeline import events as pipeline_events
from scripts.pipeline import strategy_pipeline as pipeline_strategy
from scripts.pipeline import steps as pipeline_steps
from scripts.capital_report import write_capital_report_from_trades
from scripts.multi_leg_trading_map import write_continuous_trading_map

# ====================================================================
# Config
# ====================================================================

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "research_pipeline.yaml"


def load_pipeline_config(path: Path) -> dict:
    return pipeline_config.load_pipeline_config(path)


def resolve_symbols_from_config(cfg: dict) -> str:
    return pipeline_config.resolve_symbols_from_config(cfg)


def _resolve_strategy_direction_scope(cfg: Dict[str, Any]) -> str:
    """读取策略方向范围开关: all/long/short."""
    raw = (cfg.get("strategy_scope", {}) or {}).get("direction", "all") or "all"
    scope = str(raw).strip().lower()
    if scope not in {"all", "long", "short"}:
        print(f"⚠️  未知 strategy_scope.direction={raw}, 回退为 all")
        return "all"
    return scope


def _filter_strategies_by_direction_scope(
    strategies: List[str], scope: str, cfg: Optional[Dict[str, Any]] = None
) -> List[str]:
    """按方向范围过滤策略。优先读配置 side，兼容旧命名后缀。"""
    cfg = cfg or {}
    scfg_all = (
        cfg.get("strategies", {}) if isinstance(cfg.get("strategies"), dict) else {}
    )

    def _resolve_side(strategy_name: str) -> str:
        scfg = scfg_all.get(strategy_name, {}) if isinstance(scfg_all, dict) else {}
        side_raw = ""
        if isinstance(scfg, dict):
            side_raw = str(scfg.get("side", "") or "").strip().lower()
        if side_raw in {"long", "short", "both"}:
            return side_raw
        n = str(strategy_name).lower()
        if "-long-" in n or n.endswith("-long"):
            return "long"
        if "-short-" in n or n.endswith("-short"):
            return "short"
        return "both"

    if scope == "all":
        return list(strategies)
    if scope == "long":
        return [s for s in strategies if _resolve_side(s) in {"long", "both"}]
    if scope == "short":
        return [s for s in strategies if _resolve_side(s) in {"short", "both"}]
    return list(strategies)


def _resolve_strategy_side(strategy_name: str, scfg: Dict[str, Any]) -> str:
    """策略侧向：优先配置 side，其次兼容旧命名。"""
    side_raw = str((scfg or {}).get("side", "") or "").strip().lower()
    if side_raw in {"long", "short", "both"}:
        return side_raw
    n = str(strategy_name).lower()
    if "-long-" in n or n.endswith("-long"):
        return "long"
    if "-short-" in n or n.endswith("-short"):
        return "short"
    return "both"


def _parse_month_token(month_token: str) -> Tuple[int, int]:
    """Parse YYYY-MM month token."""
    token = str(month_token or "").strip()
    try:
        dt = datetime.strptime(token, "%Y-%m")
        return dt.year, dt.month
    except Exception as exc:
        raise ValueError(f"非法月份格式: {month_token}, 期望 YYYY-MM") from exc


def _split_month_list(month_spec: str) -> List[str]:
    """Parse one or more YYYY-MM tokens from comma/space/semicolon-separated string."""
    raw = str(month_spec or "").strip()
    if not raw:
        return []
    parts = re.split(r"[\s,;]+", raw)
    out: List[str] = []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        _parse_month_token(t)
        out.append(t)
    return out


def _month_token_to_range(month_token: str) -> Tuple[str, str]:
    """Convert YYYY-MM to start/end date."""
    y, m = _parse_month_token(month_token)
    last_day = monthrange(y, m)[1]
    return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last_day:02d}"


def _add_months(date_str: str, months: int) -> str:
    """Shift YYYY-MM-DD by month delta; returns first day for shifted month."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    y = dt.year
    m = dt.month + int(months)
    while m > 12:
        y += 1
        m -= 12
    while m <= 0:
        y -= 1
        m += 12
    d = min(dt.day, monthrange(y, m)[1])
    return f"{y:04d}-{m:02d}-{d:02d}"


def _month_start(month_token: str) -> str:
    y, m = _parse_month_token(month_token)
    return f"{y:04d}-{m:02d}-01"


def _month_prev_end(month_token: str) -> str:
    ms = _month_start(month_token)
    prev_month_day = datetime.strptime(ms, "%Y-%m-%d") - timedelta(days=1)
    return prev_month_day.strftime("%Y-%m-%d")


def _calib_and_test_windows(
    *,
    month_token: str,
    calibration_months: int,
) -> Dict[str, str]:
    """For target month M: calib=[M-k, M-1], test=[M, M]."""
    test_start, test_end = _month_token_to_range(month_token)
    calib_end = _month_prev_end(month_token)
    calib_start = _add_months(test_start, -int(calibration_months))
    return {
        "calib_start": calib_start,
        "calib_end": calib_end,
        "test_start": test_start,
        "test_end": test_end,
    }


def _iter_month_tokens(start_date: str, end_date: str) -> List[str]:
    return pipeline_config.iter_month_tokens(start_date, end_date)


def _resolve_stage_strategies_root(cfg: Dict[str, Any], stage: str) -> Path:
    """Best-effort root for config consistency checks."""
    rolling_cfg = cfg.get("rolling", {}) or {}
    mode = str(rolling_cfg.get("mode", "legacy") or "legacy").strip().lower()
    if mode == "turbo_fixed_features":
        turbo_cfg = rolling_cfg.get("turbo_fixed_features", {}) or {}
        root = Path(
            str(
                turbo_cfg.get("fixed_strategies_root", "config/strategies")
                or "config/strategies"
            )
        )
        return root if root.is_absolute() else (PROJECT_ROOT / root)
    return PROJECT_ROOT / "config" / "strategies"


def _read_timeframe_from_meta_yaml_path(meta_path: Path | None) -> str:
    """从策略 meta.yaml 读取周期：优先 strategy.timeframe，其次顶层 timeframe。"""
    if meta_path is None or not meta_path.is_file():
        return ""
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return ""
    st = meta.get("strategy")
    if isinstance(st, dict):
        tf = str(st.get("timeframe", "") or "").strip()
        if tf:
            return tf
    return str(meta.get("timeframe", "") or "").strip()


def _run_config_consistency_checks(
    *,
    cfg: Dict[str, Any],
    strategies: List[str],
    stage: str,
    dry_run: bool,
) -> None:
    """Print high-signal config consistency diagnostics before running."""
    _ = dry_run  # reserved for future strict toggles
    stage_root = _resolve_stage_strategies_root(cfg, stage=stage)
    print(f"   ConfigCheck: strategies_root={stage_root}")
    for strategy in strategies:
        scfg = cfg.get("strategies", {}).get(strategy, {}) or {}
        if not isinstance(scfg, dict):
            scfg = {}
        cfg_tf = str(scfg.get("timeframe", "") or "").strip()
        cfg_dir = str(scfg.get("config", "") or "").strip()
        side = _resolve_strategy_side(strategy, scfg)

        # 优先 stage_root，其次 strategies.*.config
        cand_dirs: List[Path] = [stage_root / strategy]
        if cfg_dir:
            cfg_path = Path(cfg_dir)
            if not cfg_path.is_absolute():
                cfg_path = PROJECT_ROOT / cfg_path
            cand_dirs.append(cfg_path)
        meta_path = next(
            (d / "meta.yaml" for d in cand_dirs if (d / "meta.yaml").exists()), None
        )
        dir_path = next(
            (
                d / "archetypes" / "direction.yaml"
                for d in cand_dirs
                if (d / "archetypes" / "direction.yaml").exists()
            ),
            None,
        )

        meta_tf = _read_timeframe_from_meta_yaml_path(meta_path)
        print(
            f"   ConfigCheck[{strategy}]: timeframe={meta_tf or 'N/A'} (meta.yaml) "
            f"side={side} config={cfg_dir or 'N/A'}"
        )
        if cfg_tf and meta_tf and cfg_tf != meta_tf:
            print(
                f"⚠️  ConfigCheck[{strategy}]: 管线仅使用 meta 的 timeframe={meta_tf}；"
                f"strategies.{strategy}.timeframe={cfg_tf} 将被忽略（请删除 YAML 键或改为与 meta 一致）"
            )
        elif cfg_tf and not meta_tf:
            print(
                f"⚠️  ConfigCheck[{strategy}]: 配置了 strategies.{strategy}.timeframe={cfg_tf}，"
                f"但未从 meta.yaml 读到周期；管线 **只认 meta**，请补全 meta 并去掉 YAML 中 timeframe"
            )

        has_direction = bool(scfg.get("has_direction", False))
        if has_direction and dir_path is None:
            print(
                f"⚠️  ConfigCheck[{strategy}]: has_direction=true 但未找到 "
                "archetypes/direction.yaml (在 stage_root 或 strategies.*.config 下)"
            )

        if side == "both":
            print(
                f"ℹ️  ConfigCheck[{strategy}]: side=both（建议在策略配置显式声明 side: long/short/both）"
            )


def _quality_score_from_event_metrics(
    metrics: Dict[str, Any],
    *,
    history_w: float = 0.55,
    now_w: float = 0.45,
) -> Tuple[float, Dict[str, float]]:
    """Compute V1 quality score and components for fast-month ranking."""
    sharpe = float(metrics.get("sharpe_r", 0.0) or 0.0)
    mean_r = float(metrics.get("mean_r", 0.0) or 0.0)
    n_trades = float(metrics.get("n_trades", 0.0) or 0.0)
    near_stop_rate = float(metrics.get("near_stop_rate", 0.0) or 0.0)
    dd = float(metrics.get("max_drawdown_r", 0.0) or 0.0)
    trade_boost = min(1.0, n_trades / 40.0)
    history_edge = (
        0.70 * sharpe
        + 0.20 * mean_r
        + 0.10 * trade_boost
        - 0.08 * near_stop_rate
        - 0.04 * max(0.0, dd)
    )
    # Optional real-time strength channels (when provided by event metrics)
    cvd_accel = float(metrics.get("cvd_accel_aligned", 0.0) or 0.0)
    price_eff = float(metrics.get("price_efficiency_aligned", 0.0) or 0.0)
    of_strength = float(metrics.get("orderflow_strength_aligned", 0.0) or 0.0)
    eps = 1e-6
    now_strength = float(cvd_accel * price_eff)
    ratio_strength = (
        float(price_eff / max(eps, abs(of_strength))) if of_strength != 0 else 0.0
    )
    # Blend two interpretations; defaults to 0 when channels absent.
    now_strength = 0.7 * now_strength + 0.3 * ratio_strength
    score = history_w * history_edge + now_w * now_strength
    return float(score), {
        "history_edge": float(history_edge),
        "now_strength": float(now_strength),
        "cvd_accel_aligned": float(cvd_accel),
        "price_efficiency_aligned": float(price_eff),
        "orderflow_strength_aligned": float(of_strength),
    }


# ====================================================================


def detect_latest_data_date(data_path: str, symbols: str) -> str:
    """扫描 parquet 文件名, 推断最新可用数据月份 (首日)."""
    import glob

    dp = Path(data_path)
    latest_year, latest_month = 2023, 1
    for sym in symbols.split(","):
        # 文件名格式: BTCUSDT_4h_2025.parquet 或 BTCUSDT/2025-12.parquet 等
        for f in dp.rglob(f"*{sym.strip()}*"):
            name = f.stem
            # 尝试提取 YYYY-MM 或 YYYY
            m = re.search(r"(\d{4})-(\d{2})", name)
            if m:
                y, mo = int(m.group(1)), int(m.group(2))
            else:
                m2 = re.search(r"(\d{4})", name)
                if m2:
                    y, mo = int(m2.group(1)), 12
                else:
                    continue
            if (y, mo) > (latest_year, latest_month):
                latest_year, latest_month = y, mo

    # 返回该月下一个月的第一天 (数据"到"这个月 → end-date = 下月 1 号)
    if latest_month == 12:
        end = datetime(latest_year + 1, 1, 1)
    else:
        end = datetime(latest_year, latest_month + 1, 1)
    return end.strftime("%Y-%m-%d")


def compute_holdout_start(end_date: str, holdout_months: int) -> str:
    return pipeline_config.compute_holdout_start(end_date, holdout_months)


def resolve_strategy_dates(
    cfg: dict,
    strategy: str,
    *,
    default_end_date: str,
    forced_end_date: str = "",
) -> Dict[str, Any]:
    return pipeline_config.resolve_strategy_dates(
        cfg,
        strategy=strategy,
        default_end_date=default_end_date,
        forced_end_date=forced_end_date,
    )


# ====================================================================
# Step runner
# ====================================================================


def run_step(
    name: str,
    cmd: List[str],
    log_file: Path,
    *,
    dry_run: bool = False,
    cwd: Optional[Path] = None,
    env_extra: Optional[Dict[str, str]] = None,
) -> Tuple[int, str]:
    return pipeline_steps.run_step(
        name, cmd, log_file, dry_run=dry_run, cwd=cwd, env_extra=env_extra
    )


def find_output_dir(output: str, strategy: str) -> Optional[str]:
    return pipeline_steps.find_output_dir(output, strategy)


# ====================================================================
# Parse backtest output
# ====================================================================


def parse_backtest_stdout(output: str) -> Dict[str, Any]:
    return pipeline_steps.parse_backtest_stdout(output)


def _looks_like_gate_insufficient_sample(output: str) -> bool:
    """Detect Gate Train failure caused by too few samples after filtering."""
    text = str(output or "")
    tokens = (
        "Prefilter 后 Train 样本量",
        "统计不可信",
        "No objects to concatenate",
        "No valid samples after filtering",
    )
    return any(tok in text for tok in tokens)


def _ensure_timestamp_for_gate_input(parquet_path: Path) -> bool:
    """Ensure gate input parquet has `timestamp` column; add from `datetime` when possible."""
    try:
        if not parquet_path.exists():
            return False
        import pandas as pd  # local import to keep startup light

        df = pd.read_parquet(parquet_path)
        if "timestamp" in df.columns:
            return True
        if "datetime" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")
            df.to_parquet(parquet_path, index=False)
            print(f"   🔧 Gate 输入补列: {parquet_path} 添加 timestamp <- datetime")
            return True
        return False
    except Exception as exc:
        print(f"   ⚠️ Gate 输入 timestamp 补列失败: {exc}")
        return False


def _read_bpc_vwap_band_abs(strategies_root: Path) -> Optional[Tuple[float, float]]:
    """``single_position_band`` + vwap1200 / macro_tp_vwap → (inner_abs, outer_abs)."""
    path = strategies_root / "bpc" / "archetypes" / "direction.yaml"
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rules = data.get("direction_rules") or []
        if not isinstance(rules, list):
            return None
        for r in rules:
            if not isinstance(r, dict):
                continue
            if str(r.get("method", "")).strip() != "single_position_band":
                continue
            fid = str(r.get("id", "") or "")
            feat = str(r.get("feature", "") or "")
            if fid != "vwap1200_band" and "macro_tp_vwap" not in feat:
                continue
            try:
                inn = float(r.get("inner_abs"))
                out = float(r.get("outer_abs"))
            except (TypeError, ValueError):
                return None
            if inn > 0 and inn < 1 and out > inn and out < 1:
                return (inn, out)
            return None
        return None
    except Exception:
        return None


def _ledger_bpc_vwap_band_schedule(
    ledger: List[Dict[str, Any]],
    *,
    fallback_inner: float,
    fallback_outer: float,
) -> List[Dict[str, Any]]:
    """Each rolling month → inner/outer from ``run_root/strategies_calibrated/bpc/.../direction.yaml``."""
    fbi = float(fallback_inner)
    fbo = float(fallback_outer)
    if not (fbi > 0 and fbi < 1 and fbo > fbi and fbo < 1):
        fbi, fbo = 0.005, 0.05
    rows: List[Dict[str, Any]] = []
    for row in ledger:
        mt = str(row.get("month", "") or "").strip()
        if not mt:
            continue
        try:
            d0, d1 = _month_token_to_range(mt)
        except Exception:
            continue
        rr = Path(str(row.get("run_root", "") or ""))
        sc = rr / "strategies_calibrated"
        pair = _read_bpc_vwap_band_abs(sc) if sc.is_dir() else None
        inn, outv = pair if pair else (fbi, fbo)
        if not (inn > 0 and inn < 1 and outv > inn and outv < 1):
            inn, outv = fbi, fbo
        rows.append(
            {
                "month": mt,
                "d0": d0,
                "d1": d1,
                "inner": float(inn),
                "outer": float(outv),
            }
        )
    return rows


def _ledger_stitched_pcm_fallback_totals(
    ledger: List[Dict[str, Any]],
) -> Tuple[float, int]:
    """Sum trades/R when ``pcm_eval`` is off: ``ledger['pcm']`` is empty but event CSVs exist.

    Mirrors the fallback in ``_build_continuous_pcm_trading_map`` so ``stitched_summary.json``
    matches continuous / per-symbol map statistics.
    """
    try:
        import pandas as pd
    except Exception:
        return 0.0, 0
    total_r = 0.0
    n = 0
    for row in ledger:
        run_root = Path(str(row.get("run_root", "") or ""))
        if not run_root.is_dir():
            continue
        end_state_paths = row.get("end_state_paths", {}) or {}
        for strat in end_state_paths.keys():
            strat_name = str(strat or "").strip()
            if not strat_name:
                continue
            ep = run_root / strat_name / f"event_trades_{strat_name}.csv"
            if not ep.exists():
                continue
            try:
                df = pd.read_csv(ep)
            except Exception:
                continue
            if df.empty:
                continue
            n += len(df)
            if "pnl_r" in df.columns:
                total_r += float(
                    pd.to_numeric(df["pnl_r"], errors="coerce").fillna(0.0).sum()
                )
    return total_r, n


def _load_funnel_per_bar_from_ledger(
    ledger: List[Dict[str, Any]], symbols: Set[str]
) -> List[Dict[str, Any]]:
    """Merge ``funnel_per_bar`` from PCM joint JSONs and/or per-strategy event_backtest_*.json.

    Per rolling month, either ``pcm.json_path`` exists (multi-strategy PCM) or we fall back to
    ``<run_root>/<strat>/event_backtest_<strat>.json`` (single-strategy / no PCM eval).
    """
    out: List[Dict[str, Any]] = []
    if not ledger or not symbols:
        return out
    for row in ledger:
        pcm = row.get("pcm") or {}
        jp = str(pcm.get("json_path") or "").strip()
        if jp and Path(jp).is_file():
            try:
                data = json.loads(Path(jp).read_text(encoding="utf-8"))
                for r in data.get("funnel_per_bar") or []:
                    if str(r.get("symbol") or "") in symbols:
                        out.append(dict(r))
            except Exception as exc:
                print(f"   ⚠️  连续地图: 读取 funnel PCM JSON 失败 {jp}: {exc}")
        else:
            run_root = Path(str(row.get("run_root", "") or ""))
            if not run_root.is_dir():
                continue
            for pjson in sorted(run_root.glob("*/event_backtest_*.json")):
                try:
                    data = json.loads(pjson.read_text(encoding="utf-8"))
                    for r in data.get("funnel_per_bar") or []:
                        if str(r.get("symbol") or "") in symbols:
                            out.append(dict(r))
                except Exception as exc:
                    print(f"   ⚠️  连续地图: 读取 funnel JSON 失败 {pjson}: {exc}")
    return out


def _build_continuous_funnel_figures(
    sym: str,
    funnel_rows: List[Dict[str, Any]],
    x_range: Any,
    ref_index: Optional[Any],
    plot_w: int,
) -> List[Any]:
    """Same stacked ladder as ``event_backtest.generate_trading_map_html`` funnel panel."""
    if not funnel_rows:
        return []
    try:
        import pandas as pd
        from bokeh.models import ColumnDataSource, FixedTicker, HoverTool
        from bokeh.plotting import figure as bk_figure
    except Exception as exc:
        print(f"   ⚠️  连续地图: 无法绘制 funnel 附图: {exc}")
        return []

    by_strat: Dict[str, List[Dict[str, Any]]] = {}
    for r in funnel_rows:
        sk = str(r.get("strategy") or "unknown")
        by_strat.setdefault(sk, []).append(dict(r))

    def _pcm_y(rec: Dict[str, Any]) -> float:
        if rec.get("pcm_direction_filter") is False:
            return 0.0
        return 1.0

    def _bool_y(rec: Dict[str, Any], key: str) -> float:
        v = rec.get(key)
        if v is None:
            return float("nan")
        return 1.0 if v else 0.0

    def _dir_y(rec: Dict[str, Any]) -> float:
        dv = rec.get("direction_value")
        if dv is None:
            return float("nan")
        try:
            dvi = int(dv)
        except (TypeError, ValueError):
            return float("nan")
        return {-1: 0.0, 0: 0.5, 1: 1.0}.get(dvi, float("nan"))

    def _step_xy(ts: list, vals: list) -> tuple:
        if not ts:
            return [], []
        xs: list = []
        ys: list = []
        for i in range(len(ts)):
            if i > 0:
                xs.append(ts[i])
                ys.append(vals[i - 1])
            xs.append(ts[i])
            ys.append(vals[i])
        return xs, ys

    _STAGES = [
        ("PCM EMA", _pcm_y, "#64748b"),
        ("Prefilter", lambda rec: _bool_y(rec, "prefilter"), "#3274D9"),
        ("Gate", lambda rec: _bool_y(rec, "gate"), "#7c3aed"),
        ("Entry filter", lambda rec: _bool_y(rec, "entry_filter"), "#ca8a04"),
        ("Direction (−1/0/+1)", _dir_y, "#059669"),
    ]

    def _compact_reason(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return "; ".join(str(x) for x in value[:6])
        return str(value)

    def _block_points(sub_rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        out: Dict[str, List[Any]] = {
            "x": [],
            "y": [],
            "stage": [],
            "reason": [],
            "strategy": [],
            "direction": [],
        }
        for rec in sub_rows:
            stage = ""
            reason = ""
            y = float("nan")
            if rec.get("prefilter") is False:
                stage = "Prefilter"
                reason = _compact_reason(rec.get("prefilter_reason"))
                y = 1.0
            elif rec.get("direction_value") == 0:
                stage = "Direction"
                reason = _compact_reason(
                    rec.get("direction_reason") or rec.get("direction_rule")
                )
                y = 4.0
            elif rec.get("gate") is False:
                stage = "Gate"
                reason = _compact_reason(rec.get("gate_reasons"))
                y = 2.0
            elif rec.get("entry_filter") is False:
                stage = "Entry"
                reason = _compact_reason(rec.get("entry_filter_reason"))
                y = 3.0
            elif int(rec.get("pcm_drop_slot", 0) or 0) > 0:
                stage = "PCM"
                reason = "slot_full"
                y = 0.0
            elif int(rec.get("pcm_drop_direction_policy", 0) or 0) > 0:
                stage = "PCM"
                reason = "direction_policy"
                y = 0.0
            elif int(rec.get("pcm_drop_family_conflict", 0) or 0) > 0:
                stage = "PCM"
                reason = "family_conflict"
                y = 0.0
            elif int(rec.get("pcm_drop_daily_limit", 0) or 0) > 0:
                stage = "PCM"
                reason = "daily_limit"
                y = 0.0
            if not stage:
                continue
            out["x"].append(pd.Timestamp(rec["timestamp"]))
            out["y"].append(y)
            out["stage"].append(stage)
            out["reason"].append(reason)
            out["strategy"].append(str(rec.get("strategy") or ""))
            out["direction"].append(str(rec.get("direction_value")))
        return out

    figs: List[Any] = []
    for strat_name in sorted(by_strat.keys()):
        sub = sorted(
            by_strat[strat_name], key=lambda t: pd.Timestamp(t.get("timestamp"))
        )
        ts = [pd.Timestamp(t["timestamp"]) for t in sub]
        if ref_index is not None and getattr(ref_index, "tz", None) is not None:
            _tz = ref_index.tz
            ts = [
                (
                    x.tz_convert(_tz)
                    if x.tzinfo
                    else x.tz_localize("UTC").tz_convert(_tz)
                )
                for x in ts
            ]
        pf = bk_figure(
            title=f"{sym} · {strat_name} — gate / prefilter / direction",
            x_axis_type="datetime",
            width=plot_w,
            height=200,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            x_range=x_range,
            y_range=(-0.15, 4.65),
        )
        pf.yaxis.ticker = FixedTicker(ticks=[0, 1, 2, 3, 4])
        pf.yaxis.major_label_overrides = {
            0: "PCM",
            1: "Prefilter",
            2: "Gate",
            3: "EntryFlt",
            4: "Dir",
        }
        pf.grid.grid_line_alpha = 0.25
        for _bi, (label, fn, color) in enumerate(_STAGES):
            vals = [float(fn(rec)) for rec in sub]
            xs, ys = _step_xy(ts, [float(_bi) + 0.35 * v for v in vals])
            if xs:
                pf.line(xs, ys, line_color=color, line_width=1.6, legend_label=label)
        block_data = _block_points(sub)
        if block_data["x"]:
            bsrc = ColumnDataSource(block_data)
            blocked = pf.scatter(
                "x",
                "y",
                source=bsrc,
                marker="x",
                size=9,
                line_width=2,
                color="#dc2626",
                legend_label="No-entry reason",
            )
            pf.add_tools(
                HoverTool(
                    renderers=[blocked],
                    tooltips=[
                        ("Time", "@x{%F %H:%M}"),
                        ("Stage", "@stage"),
                        ("Reason", "@reason"),
                        ("Strategy", "@strategy"),
                        ("Dir", "@direction"),
                    ],
                    formatters={"@x": "datetime"},
                )
            )
        pf.legend.click_policy = "hide"
        pf.legend.label_text_font_size = "8pt"
        pf.legend.location = "top_left"
        pf.add_tools(
            HoverTool(
                tooltips=[("Time", "@x{%F %H:%M}"), ("y", "@y{0.2f}")],
                formatters={"@x": "datetime"},
                mode="mouse",
            )
        )
        figs.append(pf)
    return figs


def _build_continuous_pcm_trading_map(
    ledger: List[Dict[str, Any]],
    output_path: Path,
    *,
    data_path: str = "data/parquet_data",
    map_vwap_window_bars: int = 1200,
    map_long_ema_span: int = 1200,
    indicator_lookback_days: int = 140,
    band_inner_abs: float = 0.005,
    band_outer_abs: float = 0.05,
    chart_x_start: str | None = None,
    chart_x_end: str | None = None,
) -> str:
    """Build continuous multi-month map with 2H K-lines and trade overlays.

    VWAP band inner/outer are **piecewise by ledger month** from each
    ``run_root/strategies_calibrated/bpc/archetypes/direction.yaml``; gaps use
    ``band_inner_abs`` / ``band_outer_abs`` as fallback.

    Price overlays: rolling typical-price VWAP over ``map_vwap_window_bars`` bars
    (same window as ``macro_tp_vwap_1200_position`` on 2H). ``map_long_ema_span`` is
    kept for call-site compatibility; EMA overlay was removed as redundant with VWAP.
    Extra history is loaded before ``x_min`` for stable VWAP; by default the X axis
    spans first trade → last trade. If ``chart_x_start`` / ``chart_x_end`` (YYYY-MM-DD)
    are set (typically pipeline ``dates.start_date`` / ``dates.end_date``), the axis
    is widened so K线 shows the full configured sample even when early months have no trades.
    Trade segment legend: solid = primary leg, dashed = add-on (palette first color often blue).
    """
    try:
        import pandas as pd  # local import to keep startup light
    except Exception as exc:
        print(f"   ⚠️ 连续地图生成失败（缺少 pandas）: {exc}")
        return ""
    try:
        from bokeh.plotting import figure as bk_figure
        from bokeh.models import (
            HoverTool,
            Range1d,
            Div,
            ColumnDataSource,
            Legend,
            LegendItem,
        )
        from bokeh.layouts import column as bk_column
        from bokeh.resources import INLINE as BK_RESOURCES
        from bokeh.embed import file_html as bk_file_html
    except Exception as exc:
        print(f"   ⚠️ 连续地图生成失败（缺少 bokeh）: {exc}")
        return ""
    try:
        from src.data_tools.data_handler import DataHandler
    except Exception as exc:
        print(f"   ⚠️ 连续地图生成失败（DataHandler 导入失败）: {exc}")
        return ""

    frames: List[Any] = []
    for row in ledger:
        pcm = row.get("pcm") or {}
        trades_csv = str(pcm.get("trades_csv_path", "") or "")
        month_tag = str(row.get("month", ""))
        if not trades_csv:
            # Fallback: when pcm_eval is disabled, stitch per-strategy event trades.
            run_root = Path(str(row.get("run_root", "") or ""))
            end_state_paths = row.get("end_state_paths", {}) or {}
            for strat in end_state_paths.keys():
                strat_name = str(strat or "").strip()
                if not strat_name:
                    continue
                ep = run_root / strat_name / f"event_trades_{strat_name}.csv"
                if not ep.exists():
                    continue
                try:
                    df = pd.read_csv(ep)
                except Exception:
                    continue
                if df.empty:
                    continue
                df = df.copy()
                df["month"] = month_tag
                df["source"] = f"event:{strat_name}"
                frames.append(df)
            continue

        p = Path(trades_csv)
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                df = df.copy()
                df["month"] = month_tag
                df["source"] = "pcm_joint"
                frames.append(df)

    if not frames:
        return ""

    merged = pd.concat(frames, ignore_index=True)
    need_cols = {"symbol", "entry_time", "exit_time", "pnl_r", "archetype", "side"}
    if not need_cols.issubset(set(merged.columns)):
        return ""

    merged["entry_time"] = pd.to_datetime(
        merged["entry_time"], utc=True, errors="coerce"
    )
    merged["exit_time"] = pd.to_datetime(merged["exit_time"], utc=True, errors="coerce")
    merged = merged.dropna(subset=["entry_time", "exit_time", "symbol"]).copy()
    if merged.empty:
        return ""

    merged["entry_price"] = pd.to_numeric(merged.get("entry_price"), errors="coerce")
    merged["exit_price"] = pd.to_numeric(merged.get("exit_price"), errors="coerce")
    merged["pnl_r"] = pd.to_numeric(merged["pnl_r"], errors="coerce").fillna(0.0)
    if "is_add_position" in merged.columns:
        merged["is_add_position"] = merged["is_add_position"].astype(bool)
    elif "_is_add_position" in merged.columns:
        merged["is_add_position"] = merged["_is_add_position"].astype(bool)
    elif "add_position_seq" in merged.columns:
        merged["is_add_position"] = (
            pd.to_numeric(merged["add_position_seq"], errors="coerce")
            .fillna(0)
            .astype(int)
            > 0
        )
    else:
        merged["is_add_position"] = False
    if "add_position_seq" in merged.columns:
        merged["add_position_seq"] = (
            pd.to_numeric(merged["add_position_seq"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    elif "_add_position_seq" in merged.columns:
        merged["add_position_seq"] = (
            pd.to_numeric(merged["_add_position_seq"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        merged["add_position_seq"] = 0
    merged = merged.sort_values("entry_time").reset_index(drop=True)
    for _c in ("month", "source"):
        if _c not in merged.columns:
            merged[_c] = ""
        else:
            merged[_c] = merged[_c].fillna("").astype(str)
    symbols = sorted(str(s) for s in merged["symbol"].dropna().unique().tolist())
    if not symbols:
        return ""
    funnel_all = _load_funnel_per_bar_from_ledger(ledger, set(symbols))
    if funnel_all:
        print(
            f"   连续地图: 已合并 funnel_per_bar 行数={len(funnel_all)} (PCM/单月 event JSON)"
        )

    merged["archetype"] = (
        merged.get("archetype", "unknown").astype(str).fillna("unknown")
    )
    total_trades = int(len(merged))
    total_r = float(merged["pnl_r"].sum())
    win_rate = float((merged["pnl_r"] > 0).mean()) if total_trades > 0 else 0.0
    source_counts = (
        merged["source"].value_counts().to_dict() if "source" in merged.columns else {}
    )
    x_min = merged["entry_time"].min() - pd.Timedelta(days=2)
    x_max = merged["exit_time"].max() + pd.Timedelta(days=2)
    if chart_x_start:
        try:
            xs = pd.Timestamp(str(chart_x_start).strip(), tz="UTC")
            if xs.tzinfo is None:
                xs = xs.tz_localize("UTC")
            if xs < x_min:
                x_min = xs
        except Exception:
            pass
    if chart_x_end:
        try:
            xe = pd.Timestamp(str(chart_x_end).strip(), tz="UTC")
            if xe.tzinfo is None:
                xe = xe.tz_localize("UTC")
            if xe > x_max:
                x_max = xe
        except Exception:
            pass
    archetypes = sorted(
        str(a) for a in merged["archetype"].dropna().astype(str).unique().tolist()
    )
    _palette = [
        "#2563eb",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#16a34a",
        "#be123c",
        "#0d9488",
        "#7c3aed",
        "#ca8a04",
        "#dc2626",
        "#4f46e5",
        "#0f766e",
    ]
    archetype_colors = {
        a: _palette[i % len(_palette)] for i, a in enumerate(archetypes)
    }
    dh = DataHandler(str(data_path))
    bar_ms = int(2 * 60 * 60 * 1000 * 0.65)  # 2H body width
    # Wider figure; legend overlays inside plot (not right panel) so candle width is not squeezed.
    plot_w = 1900
    fig_list: List[Any] = []

    _ = band_inner_abs
    _ = band_outer_abs

    vw_n = max(2, int(map_vwap_window_bars))
    _ = map_long_ema_span  # API compat; EMA not drawn
    lookback_days = max(int(indicator_lookback_days), int((vw_n * 2 + 24) // 24) + 7)
    try:
        from scripts.event_backtest import _rolling_tp_vwap as _pcm_rolling_tp_vwap
    except Exception as exc:
        print(f"   ⚠️ 连续地图: 无法导入 _rolling_tp_vwap: {exc}")
        return ""

    # ── Top panel: cumulative R (overall + per archetype) ──
    cum_df = merged.sort_values("exit_time").copy()
    cum_df["cum_r_all"] = cum_df["pnl_r"].cumsum()
    p_cum = bk_figure(
        title=f"Cumulative R | total_r={total_r:.2f} | trades={total_trades}",
        x_axis_type="datetime",
        width=plot_w,
        height=630,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p_cum.grid.grid_line_alpha = 0.25
    p_cum.x_range = Range1d(x_min.to_pydatetime(), x_max.to_pydatetime())
    p_cum.yaxis.axis_label = "Cumulative R"
    p_cum.line(
        x=cum_df["exit_time"],
        y=cum_df["cum_r_all"],
        line_color="#111827",
        line_width=2.2,
        alpha=0.95,
        legend_label="ALL (cumulative R)",
    )
    for arch in archetypes:
        g = cum_df.loc[cum_df["archetype"] == arch, ["exit_time", "pnl_r"]].copy()
        if g.empty:
            continue
        g = g.sort_values("exit_time")
        g["cum_r_arch"] = g["pnl_r"].cumsum()
        p_cum.line(
            x=g["exit_time"],
            y=g["cum_r_arch"],
            line_color=archetype_colors.get(arch, "#6b7280"),
            line_width=1.6,
            alpha=0.9,
            legend_label=f"{arch} (cumulative R)",
        )
    p_cum.legend.location = "top_left"
    p_cum.legend.click_policy = "hide"
    p_cum.add_tools(
        HoverTool(
            tooltips=[("time", "@x{%F %T}"), ("cum_r", "@y{0.000}")],
            formatters={"@x": "datetime"},
            mode="vline",
        )
    )
    fig_list.append(p_cum)
    for sym in symbols:
        tdf = merged.loc[merged["symbol"] == sym].copy()
        if tdf.empty:
            continue
        for _c in ("month", "source"):
            if _c not in tdf.columns:
                tdf[_c] = ""
            else:
                tdf[_c] = tdf[_c].fillna("").astype(str)
        arch_renderers: Dict[str, List[Any]] = {}
        r_crf_box: Any = None
        ref_idx_for_funnel: Any = None
        p = bk_figure(
            title=f"{sym} | trades={len(tdf)} | total_r={float(tdf['pnl_r'].sum()):.2f}",
            x_axis_type="datetime",
            width=plot_w,
            height=720,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.grid.grid_line_alpha = 0.25
        p.x_range = Range1d(x_min.to_pydatetime(), x_max.to_pydatetime())
        p.yaxis.axis_label = "Price"

        r_vwap_ref = None
        # 2H K-lines + 与 event 单图一致的 1200-bar VWAP / 1200-span EMA（多取历史只用于计算）
        load_start = (x_min - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        load_end = x_max.strftime("%Y-%m-%d")
        try:
            ohlcv = dh.load_ohlcv(
                symbol=sym,
                timeframe="120T",
                start_date=load_start,
                end_date=load_end,
            )
        except Exception:
            ohlcv = pd.DataFrame()
        if ohlcv is not None and not ohlcv.empty:
            cdf = ohlcv.copy()
            if not isinstance(cdf.index, pd.DatetimeIndex):
                cdf.index = pd.to_datetime(
                    cdf.get("timestamp"), utc=True, errors="coerce"
                )
            _cols = ["open", "high", "low", "close"]
            if "volume" in cdf.columns:
                _cols.append("volume")
            cdf = cdf[_cols].dropna(subset=["open", "high", "low", "close"])
            if not cdf.empty:
                vwap_full = _pcm_rolling_tp_vwap(cdf, vw_n)

                view_start = pd.Timestamp(x_min)
                view_end = pd.Timestamp(x_max)
                idx = cdf.index
                if idx.tz is not None:
                    if view_start.tzinfo is None:
                        view_start = view_start.tz_localize("UTC").tz_convert(idx.tz)
                    else:
                        view_start = view_start.tz_convert(idx.tz)
                    if view_end.tzinfo is None:
                        view_end = view_end.tz_localize("UTC").tz_convert(idx.tz)
                    else:
                        view_end = view_end.tz_convert(idx.tz)
                else:
                    # Trades / merged bounds may be tz-aware while OHLCV index is naive UTC wall-time
                    if view_start.tzinfo is not None:
                        view_start = view_start.tz_convert("UTC").tz_localize(None)
                    if view_end.tzinfo is not None:
                        view_end = view_end.tz_convert("UTC").tz_localize(None)
                plot_mask = (idx >= view_start) & (idx <= view_end)
                cdf_plot = cdf.loc[plot_mask]
                if cdf_plot.empty:
                    cdf_plot = cdf
                ref_idx_for_funnel = cdf_plot.index

                inc = cdf_plot["close"] >= cdf_plot["open"]
                dec = ~inc
                p.segment(
                    cdf_plot.index[inc],
                    cdf_plot["high"][inc],
                    cdf_plot.index[inc],
                    cdf_plot["low"][inc],
                    color="#26a69a",
                    line_width=1,
                )
                p.segment(
                    cdf_plot.index[dec],
                    cdf_plot["high"][dec],
                    cdf_plot.index[dec],
                    cdf_plot["low"][dec],
                    color="#ef5350",
                    line_width=1,
                )
                p.vbar(
                    cdf_plot.index[inc],
                    bar_ms,
                    cdf_plot["open"][inc],
                    cdf_plot["close"][inc],
                    fill_color="#26a69a",
                    line_color="#26a69a",
                    fill_alpha=0.8,
                )
                p.vbar(
                    cdf_plot.index[dec],
                    bar_ms,
                    cdf_plot["open"][dec],
                    cdf_plot["close"][dec],
                    fill_color="#ef5350",
                    line_color="#ef5350",
                    fill_alpha=0.8,
                )
                vp = vwap_full.reindex(cdf_plot.index)
                r_vwap_ref = p.line(
                    cdf_plot.index,
                    vp,
                    line_color="#c026d3",
                    line_width=1.35,
                    line_alpha=0.78,
                )
                # Keep only VWAP reference line on continuous trading map.

                # ── CRF box: rolling 120 lo/hi band (not merged min/max rects) ──
                # 与 prefilter 同条件处着色；每根 bar 用当期的 box_hi/lo，避免长段连绿盖住整图。
                if "crf" in archetypes:
                    try:
                        import numpy as _np
                        from src.features.time_series.box_structure_features import (
                            compute_box_structure_from_series as _box_feat,
                        )

                        _bx = _box_feat(
                            close=cdf["close"],
                            high=cdf["high"],
                            low=cdf["low"],
                        )
                    except Exception as _bex:
                        print(f"   ⚠️ CRF box overlay failed for {sym}: {_bex}")
                        _bx = None
                    if _bx is not None and not _bx.empty:
                        _bx = _bx.reindex(cdf_plot.index)
                        _stab = pd.to_numeric(
                            _bx.get("box_stability_120"), errors="coerce"
                        )
                        _widp = pd.to_numeric(
                            _bx.get("box_width_pct_120"), errors="coerce"
                        )
                        _hi = pd.to_numeric(_bx.get("box_hi_120"), errors="coerce")
                        _lo = pd.to_numeric(_bx.get("box_lo_120"), errors="coerce")
                        _touch_hi = pd.to_numeric(
                            _bx.get("box_touches_hi_120"), errors="coerce"
                        )
                        _touch_lo = pd.to_numeric(
                            _bx.get("box_touches_lo_120"), errors="coerce"
                        )
                        # Keep in sync with config/strategies/crf/archetypes/prefilter.yaml:
                        #   stab>=0.85, 0.04 <= width <= 0.30, hi/lo touches>=5
                        _qual = (
                            (_stab.fillna(0.0) >= 0.85)
                            & (_widp.fillna(0.0) >= 0.04)
                            & (_widp.fillna(1.0) <= 0.30)
                            & (_touch_hi.fillna(0.0) >= 5)
                            & (_touch_lo.fillna(0.0) >= 5)
                        )

                        _pass_rate = float(_qual.mean()) if len(_qual) else 0.0
                        print(
                            f"   CRF prefilter overlay {sym}: pass_rate={_pass_rate:.1%}"
                        )

                        # Use per-bar vbar (one thin rectangle per qualified bar) so
                        # non-qualifying gaps are truly empty — varea would otherwise
                        # bridge NaNs visually and make everything look green.
                        qidx = cdf_plot.index[_qual.values]
                        if len(qidx) > 0:
                            q_lo = _lo.values[_qual.values]
                            q_hi = _hi.values[_qual.values]
                            r_crf_box = p.vbar(
                                x=qidx,
                                width=bar_ms,
                                top=q_hi,
                                bottom=q_lo,
                                fill_color="#22c55e",
                                fill_alpha=0.18,
                                line_color=None,
                            )

                # Trade overlays (split by archetype)
        tdf["pnl_color"] = [
            "#16a34a" if x >= 0 else "#dc2626"
            for x in tdf["pnl_r"].astype(float).tolist()
        ]
        tdf["arch_color"] = [
            archetype_colors.get(str(a), "#6b7280")
            for a in tdf["archetype"].astype(str).tolist()
        ]
        for arch in sorted(tdf["archetype"].astype(str).unique().tolist()):
            adf = tdf.loc[tdf["archetype"].astype(str) == arch].copy()
            if adf.empty:
                continue
            arch_renderers.setdefault(arch, [])
            lines = adf.dropna(subset=["entry_price", "exit_price"]).copy()
            if not lines.empty:
                base_lines = lines.loc[~lines["is_add_position"]].copy()
                add_lines = lines.loc[lines["is_add_position"]].copy()
                if not base_lines.empty:
                    src_l = ColumnDataSource(base_lines)
                    r = p.segment(
                        x0="entry_time",
                        y0="entry_price",
                        x1="exit_time",
                        y1="exit_price",
                        source=src_l,
                        line_color="arch_color",
                        line_alpha=0.42,
                        line_width=1.3,
                    )
                    arch_renderers[arch].append(r)
                if not add_lines.empty:
                    src_la = ColumnDataSource(add_lines)
                    r = p.segment(
                        x0="entry_time",
                        y0="entry_price",
                        x1="exit_time",
                        y1="exit_price",
                        source=src_la,
                        line_color="arch_color",
                        line_dash="dashed",
                        line_alpha=0.75,
                        line_width=2.0,
                    )
                    arch_renderers[arch].append(r)
            entries = adf.dropna(subset=["entry_price"]).copy()
            if not entries.empty:
                base_entries = entries.loc[~entries["is_add_position"]].copy()
                add_entries = entries.loc[entries["is_add_position"]].copy()
                if not base_entries.empty:
                    base_long = base_entries.loc[
                        base_entries["side"]
                        .astype(str)
                        .str.upper()
                        .isin(["LONG", "BUY"])
                    ].copy()
                    base_short = base_entries.loc[
                        base_entries["side"]
                        .astype(str)
                        .str.upper()
                        .isin(["SHORT", "SELL"])
                    ].copy()
                    if not base_long.empty:
                        src_e = ColumnDataSource(base_long)
                        r = p.scatter(
                            x="entry_time",
                            y="entry_price",
                            source=src_e,
                            marker="triangle",
                            size=10,
                            fill_color="arch_color",
                            line_color="pnl_color",
                            line_width=1.3,
                            alpha=0.90,
                        )
                        arch_renderers[arch].append(r)
                    if not base_short.empty:
                        src_es = ColumnDataSource(base_short)
                        r = p.scatter(
                            x="entry_time",
                            y="entry_price",
                            source=src_es,
                            marker="inverted_triangle",
                            size=10,
                            fill_color="arch_color",
                            line_color="pnl_color",
                            line_width=1.3,
                            alpha=0.90,
                        )
                        arch_renderers[arch].append(r)
                if not add_entries.empty:
                    src_ea = ColumnDataSource(add_entries)
                    r = p.scatter(
                        x="entry_time",
                        y="entry_price",
                        source=src_ea,
                        marker="diamond",
                        size=11,
                        fill_color="arch_color",
                        line_color="pnl_color",
                        line_width=1.5,
                        alpha=0.95,
                    )
                    arch_renderers[arch].append(r)
            exits = adf.dropna(subset=["exit_price"]).copy()
            if not exits.empty:
                base_exits = exits.loc[~exits["is_add_position"]].copy()
                add_exits = exits.loc[exits["is_add_position"]].copy()
                if not base_exits.empty:
                    src_x = ColumnDataSource(base_exits)
                    r = p.scatter(
                        x="exit_time",
                        y="exit_price",
                        source=src_x,
                        marker="square",
                        size=7,
                        fill_color="arch_color",
                        line_color="pnl_color",
                        line_width=1.1,
                        alpha=0.78,
                    )
                    arch_renderers[arch].append(r)
                if not add_exits.empty:
                    src_xa = ColumnDataSource(add_exits)
                    r = p.scatter(
                        x="exit_time",
                        y="exit_price",
                        source=src_xa,
                        marker="circle_x",
                        size=9,
                        fill_color="arch_color",
                        line_color="pnl_color",
                        line_width=1.4,
                        alpha=0.9,
                    )
                    arch_renderers[arch].append(r)
        all_trade_renderers: List[Any] = [
            r for rs in arch_renderers.values() for r in rs
        ]
        if r_vwap_ref is not None:
            p.add_tools(
                HoverTool(
                    tooltips=[
                        ("time", "@x{%F %T}"),
                        ("vwap_px", "@y{0.0000}"),
                    ],
                    formatters={"@x": "datetime"},
                    mode="mouse",
                    renderers=[r_vwap_ref],
                )
            )
        if all_trade_renderers:
            p.add_tools(
                HoverTool(
                    tooltips=[
                        ("archetype", "@archetype"),
                        ("side", "@side"),
                        ("month", "@month"),
                        ("source", "@source"),
                        ("pnl_r", "@pnl_r{0.000}"),
                        ("is_add", "@is_add_position"),
                        ("add_seq", "@add_position_seq"),
                        ("entry", "@entry_time{%F %T}"),
                        ("exit", "@exit_time{%F %T}"),
                    ],
                    formatters={"@entry_time": "datetime", "@exit_time": "datetime"},
                    mode="mouse",
                    renderers=all_trade_renderers,
                )
            )
        if r_crf_box is not None:
            p.add_tools(
                HoverTool(
                    tooltips=[
                        ("box_lo_120", "@y1{0.0000}"),
                        ("box_hi_120", "@y2{0.0000}"),
                    ],
                    mode="mouse",
                    renderers=[r_crf_box],
                )
            )
        legend_items: List[Any] = []
        if r_vwap_ref is not None:
            legend_items.append(
                LegendItem(
                    label=f"Rolling TP-VWAP ({vw_n}×2H, local symbol price)",
                    renderers=[r_vwap_ref],
                )
            )
        if r_crf_box is not None:
            legend_items.append(
                LegendItem(
                    label="CRF: rolling 120 lo/hi (where prefilter passes)",
                    renderers=[r_crf_box],
                )
            )
        legend_items.extend(
            [
                LegendItem(label=str(arch), renderers=rs)
                for arch, rs in sorted(arch_renderers.items())
                if rs
            ]
        )
        if legend_items:
            legend = Legend(
                items=legend_items,
                location="top_left",
                orientation="vertical",
                spacing=5,
                padding=8,
                margin=8,
                label_text_font_size="10pt",
                background_fill_alpha=0.88,
                border_line_color="#94a3b8",
                border_line_alpha=0.5,
            )
            legend.click_policy = "hide"
            p.add_layout(legend, "center")
        fig_list.append(p)
        _funnel_rows_sym = [r for r in funnel_all if str(r.get("symbol") or "") == sym]
        for _f_pf in _build_continuous_funnel_figures(
            sym,
            _funnel_rows_sym,
            p.x_range,
            ref_idx_for_funnel,
            plot_w,
        ):
            fig_list.append(_f_pf)

    run_months = sorted(str(r.get("month", "")) for r in ledger if r.get("month"))
    title = "Continuous Trading Map (2H K-line)"
    subtitle = (
        f"months={run_months[0]}~{run_months[-1]} | "
        f"trades={total_trades} | total_r={total_r:.4f} | win_rate={win_rate:.2%}"
        if run_months
        else f"trades={total_trades} | total_r={total_r:.4f} | win_rate={win_rate:.2%}"
    )
    if source_counts:
        src_part = ", ".join(f"{k}:{v}" for k, v in source_counts.items())
        subtitle = f"{subtitle} | source=({src_part})"
    if not fig_list:
        return ""

    for _fi, _fblk in enumerate(fig_list):
        _fblk.xaxis.visible = _fi == len(fig_list) - 1

    header = Div(
        text=(
            f"<h2>{title}</h2><p>{subtitle}</p>"
            f"<p style='font-size:13px;line-height:1.45;max-width:{plot_w}px'>"
            "<b>图例（价格图）</b> 叠在图内左上角。品红实线 = 各 symbol <b>自身</b> 2H K 线上"
            "滚动典型价 VWAP（1200 根 bar，仅价格尺度展示）。"
            "若策略在 meta 中启用了 <code>macro_tp_vwap_anchor</code>，"
            "实盘/事件回测里 <code>macro_tp_vwap_1200_position</code>（gate / direction / "
            "<code>structural_exit vwap1200</code>）用的是<b>锚定品种</b>（默认 BTC）在同一时刻的归一化位置，"
            "与本图各币种的紫色价格线<b>不一定一致</b>。"
            "<b>各 symbol 价格图下方</b> 附 <code>funnel_per_bar</code> 阶梯图"
            "（PCM / prefilter / gate / entry / direction，与单月 event map 同源；数据来自每月经 ledger 的 "
            "PCM JSON 或 <code>event_backtest_*.json</code>，无该字段则该段不显示附图）。"
            "红色 x 标记为<b>未开仓拦截点</b>，悬浮可查看 Prefilter/Gate/Direction/Entry/PCM 的具体原因。"
            "浅绿带 = 各时刻因果 <code>box_lo_120</code>~<code>box_hi_120</code>（与 prefilter 同条件时着色），"
            "是<b>每根 2H</b> 的 rolling 上下沿，<b>不是</b>把长段 min/max 合成一块矩形。"
            " 绿/红 K 线为涨跌。入场→出场：<b>实线</b>=首仓腿，<b>虚线</b>=加仓腿；△ 多 · ▽ 空 · ◇ 加仓 · □ 平仓。"
            "</p>"
        ),
        width=plot_w,
    )
    html = bk_file_html(bk_column([header] + fig_list), BK_RESOURCES, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)


def _resolve_trade_objective(
    layer_cfg: Dict[str, Any],
    *,
    default_min: int,
    default_max: int,
    default_penalty_low: float,
    default_penalty_high: float,
    default_stability: float = 0.0,
) -> Dict[str, float]:
    """Resolve soft trade-range objective with backward compatibility."""
    target_min = int(
        layer_cfg.get(
            "target_trades_min",
            layer_cfg.get("min_trades_target", default_min),
        )
    )
    target_max = int(
        layer_cfg.get("target_trades_max", max(default_max, target_min * 4))
    )
    penalty_low = float(
        layer_cfg.get(
            "trade_penalty_low",
            layer_cfg.get("trade_penalty", default_penalty_low),
        )
    )
    penalty_high = float(layer_cfg.get("trade_penalty_high", default_penalty_high))
    stability_penalty = float(layer_cfg.get("stability_penalty", default_stability))
    return {
        "target_min": float(target_min),
        "target_max": float(max(target_max, target_min)),
        "penalty_low": penalty_low,
        "penalty_high": penalty_high,
        "stability_penalty": stability_penalty,
    }


def _score_with_trade_objective(
    *,
    sharpe: float,
    trades: int,
    objective: Dict[str, float],
) -> Dict[str, float]:
    if sharpe == float("-inf"):
        return {
            "score": float("-inf"),
            "low_gap": 0.0,
            "high_gap": 0.0,
            "low_penalty": 0.0,
            "high_penalty": 0.0,
        }
    t = float(max(0, int(trades or 0)))
    low_gap = max(0.0, objective["target_min"] - t)
    high_gap = max(0.0, t - objective["target_max"])
    low_penalty = objective["penalty_low"] * low_gap
    high_penalty = objective["penalty_high"] * high_gap
    return {
        "score": float(sharpe) - low_penalty - high_penalty,
        "low_gap": low_gap,
        "high_gap": high_gap,
        "low_penalty": low_penalty,
        "high_penalty": high_penalty,
    }


# ====================================================================
# Snapshot & Compare
# ====================================================================


def snapshot_archetypes(strategy: str, strategy_config: dict, dest: Path):
    """复制当前 archetypes/ 配置到快照目录."""
    src = PROJECT_ROOT / strategy_config["config"] / "archetypes"
    if src.exists():
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)


def load_archetype_thresholds(strategy: str, strategy_config: dict) -> Dict[str, Any]:
    """读取当前 archetypes/*.yaml 中的关键阈值."""
    thresholds: Dict[str, Any] = {}
    arch_dir = PROJECT_ROOT / strategy_config["config"] / "archetypes"
    for name in ["gate.yaml", "evidence.yaml", "entry_filters.yaml", "execution.yaml"]:
        f = arch_dir / name
        if f.exists():
            thresholds[name] = yaml.safe_load(f.read_text(encoding="utf-8"))
    return thresholds


def find_previous_report(history_dir: Path, strategy: str) -> Optional[Dict[str, Any]]:
    """找到上一次研究的 report.json."""
    strat_dir = history_dir / strategy
    if not strat_dir.exists():
        return None
    runs = sorted(strat_dir.iterdir(), reverse=True)
    for run_dir in runs:
        report = run_dir / "report.json"
        if report.exists():
            return json.loads(report.read_text(encoding="utf-8"))
    return None


def check_deploy_gate(
    decision: str,
    comparison: Dict[str, Any],
    drift_levels: Optional[Dict[str, str]],
    deploy_cfg: dict,
) -> Dict[str, Any]:
    """检查是否满足 deploy 门禁条件.

    双层逻辑:
      1. 触发条件 (OR): 至少一个满足才「值得」deploy
         - Sharpe 显著提升 (>= trigger_sharpe_improve)
         - 漂移 >= HIGH 且 |Sharpe 变化| > min_sharpe_change (参数过时 + 性能有变)
      2. 安全门禁 (AND): 全部满足才「允许」deploy
         - ADOPT 决策
         - min_trades >= 阈值

    Returns: {"deploy_ready": bool, "triggers": [...], "safety": [...],
              "blocked_by": [...], "skip_reason": str|None}
    """
    triggers: List[Dict[str, Any]] = []
    safety: List[Dict[str, Any]] = []
    blocked: List[str] = []
    skip_reason: Optional[str] = None

    DRIFT_ORDER = {
        "NONE": 0,
        "LOW": 1,
        "STABLE": 1,
        "MONITOR": 2,
        "MEDIUM": 2,
        "REVIEW": 3,
        "HIGH": 3,
        "ADJUST": 4,
    }

    # ── 触发条件 (OR) ─────────────────────────────────────
    triggered = False

    # T1. Sharpe 提升
    sharpe_thresh = deploy_cfg.get("trigger_sharpe_improve", 0.05)
    prev_sharpe = comparison.get("previous_sharpe")
    cur_sharpe = comparison.get("current_sharpe", 0)
    if prev_sharpe is not None and prev_sharpe != 0:
        improve = (cur_sharpe - prev_sharpe) / abs(prev_sharpe)
        t1_ok = improve >= sharpe_thresh
        triggers.append(
            {
                "rule": "sharpe_improve",
                "value": f"{improve:+.1%}",
                "threshold": f">= {sharpe_thresh:.0%}",
                "pass": t1_ok,
            }
        )
        if t1_ok:
            triggered = True
    else:
        # 首次运行, 无对比基准 → 视为触发 (首版本必须 deploy)
        triggers.append({"rule": "sharpe_improve", "value": "首次运行", "pass": True})
        triggered = True

    # T2. 漂移级别 + Sharpe 稳定性保护
    trigger_drift = deploy_cfg.get("trigger_drift_level", "HIGH")
    min_sharpe_chg = deploy_cfg.get("min_sharpe_change", 0.03)
    if drift_levels:
        overall = max(
            drift_levels.values(), key=lambda x: DRIFT_ORDER.get(x, 0), default="NONE"
        )
        drift_ok = DRIFT_ORDER.get(overall, 0) >= DRIFT_ORDER.get(trigger_drift, 3)
        # 稳定性保护: 即使漂移达标, |Sharpe变化| 须 > min_sharpe_change 才触发
        sharpe_changed = True  # 默认有变化
        if prev_sharpe is not None and prev_sharpe != 0:
            abs_chg = abs(cur_sharpe - prev_sharpe) / abs(prev_sharpe)
            sharpe_changed = abs_chg > min_sharpe_chg
        t2_ok = drift_ok and sharpe_changed
        t2_note = f">= {trigger_drift}"
        if drift_ok and not sharpe_changed:
            t2_note += f" (Sharpe稳定, |变化|<={min_sharpe_chg:.0%}, 不触发)"
        triggers.append(
            {
                "rule": "drift_level",
                "value": overall,
                "threshold": t2_note,
                "pass": t2_ok,
            }
        )
        if t2_ok:
            triggered = True
    else:
        triggers.append({"rule": "drift_level", "value": "无历史对比", "pass": False})

    if not triggered:
        skip_reason = "无触发条件: Sharpe 提升不足 且 漂移较小 → 不需要 deploy"

    # ── 安全门禁 (AND) ────────────────────────────────────
    # S1. require_adopt
    if deploy_cfg.get("require_adopt", True):
        s1_ok = decision == "ADOPT"
        safety.append({"rule": "require_adopt", "value": decision, "pass": s1_ok})
        if not s1_ok:
            blocked.append(f"决策={decision}, 需要 ADOPT")

    # S2. min_trades
    min_trades = deploy_cfg.get("min_trades", 50)
    cur_trades = comparison.get("current_trades", 0)
    s2_ok = cur_trades >= min_trades
    safety.append(
        {
            "rule": "min_trades",
            "value": cur_trades,
            "threshold": min_trades,
            "pass": s2_ok,
        }
    )
    if not s2_ok:
        blocked.append(f"trades={cur_trades} < {min_trades}")

    # ── 最终判定 ──────────────────────────────────────────
    # 必须: 有触发 AND 安全门禁全过
    deploy_ready = triggered and len(blocked) == 0
    return {
        "deploy_ready": deploy_ready,
        "triggered": triggered,
        "triggers": triggers,
        "safety": safety,
        "blocked_by": blocked,
        "skip_reason": skip_reason,
        "require_human_confirm": deploy_cfg.get("require_human_confirm", True),
    }


def compare_runs(
    current: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    rules: dict,
) -> Dict[str, Any]:
    """确定性对比, 输出决策."""
    min_trades = rules.get("min_trades", 10)
    adopt_ratio = rules.get("sharpe_adopt_ratio", 0.7)
    reject_floor = rules.get("sharpe_reject_floor", 0.0)

    cur_metrics = current.get("backtest_metrics", {})
    cur_trades = cur_metrics.get("total_trades", 0)
    cur_sharpe = cur_metrics.get("sharpe_per_trade", 0.0)

    result = {
        "current_sharpe": cur_sharpe,
        "current_trades": cur_trades,
        "previous_run": None,
        "previous_sharpe": None,
        "sharpe_ratio": None,
        "decision": "ADOPT",
        "reasons": [],
    }

    # Rule 1: 交易数太少
    if cur_trades < min_trades:
        result["decision"] = "ERROR"
        result["reasons"].append(f"trades={cur_trades} < min={min_trades}")
        return result

    # Rule 2: Sharpe <= 0
    if cur_sharpe <= reject_floor:
        result["decision"] = "ALERT"
        result["reasons"].append(f"sharpe={cur_sharpe:.4f} <= floor={reject_floor}")

    # 首次运行
    if previous is None:
        result["reasons"].append("首次运行, 无历史对比")
        if result["decision"] != "ALERT":
            result["decision"] = "ADOPT"
        return result

    prev_metrics = previous.get("backtest_metrics", {})
    prev_sharpe = prev_metrics.get("sharpe_per_trade", 0.0)
    result["previous_run"] = previous.get("timestamp")
    result["previous_sharpe"] = prev_sharpe

    if prev_sharpe > 0:
        ratio = cur_sharpe / prev_sharpe
        result["sharpe_ratio"] = ratio

        if ratio >= adopt_ratio:
            if result["decision"] != "ALERT":
                result["decision"] = "ADOPT"
            result["reasons"].append(f"sharpe_ratio={ratio:.2f} >= {adopt_ratio}")
        else:
            result["decision"] = "ALERT"
            result["reasons"].append(
                f"sharpe_ratio={ratio:.2f} < {adopt_ratio} (显著衰减)"
            )
    else:
        result["reasons"].append(f"prev_sharpe={prev_sharpe:.4f} <= 0, 跳过比值")

    return result


# ====================================================================
# Data download & convert (Step 0)
# ====================================================================


def run_data_download(
    cfg: dict,
    *,
    start_date: str,
    end_date: str,
    symbols: str,
    log: Path,
    dry_run: bool = False,
) -> int:
    """Step 0: 下载 + 转换最新月度 aggTrades 数据 (增量).

    已有的月份自动跳过, 只下载新增月份.
    """
    dl_cfg = cfg.get("download", {})
    if not dl_cfg.get("enabled", True):
        print("\n⏭️  数据下载已禁用 (download.enabled=false), 跳过")
        return 0

    data_dir = dl_cfg.get("data_dir", "data/agg_data")
    parquet_dir = dl_cfg.get("parquet_dir", "data/parquet_data")

    # 从 start_date / end_date 推算 year-month
    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")

    # Step 0a: Download
    # mlbot data download --symbols 为「逗号分隔的单个 TEXT」，不可拆成多个 argv
    sym_csv = ",".join(s.strip() for s in symbols.split(",") if s.strip())
    rc, _ = run_step(
        "Data Download",
        [
            "mlbot",
            "data",
            "download",
            "--no-docker",
            "--symbols",
            sym_csv,
            "--start-year",
            str(sd.year),
            "--start-month",
            str(sd.month),
            "--end-year",
            str(ed.year),
            "--end-month",
            str(ed.month),
            "--data-dir",
            data_dir,
            "--parquet-dir",
            parquet_dir,
        ],
        log,
        dry_run=dry_run,
    )

    if rc != 0 and not dry_run:
        print("  ⚠️  下载步骤失败, 尝试继续使用本地数据...")

    # Step 0b: Convert (ZIP → Parquet, 增量)
    rc, _ = run_step(
        "Data Convert",
        [
            "mlbot",
            "data",
            "convert",
            "--no-docker",
            "--input-dir",
            data_dir,
            "--output-dir",
            parquet_dir,
        ],
        log,
        dry_run=dry_run,
    )

    if rc != 0 and not dry_run:
        print("  ⚠️  转换步骤失败, 尝试继续使用已有数据...")

    return 0  # 不中断流水线, 即使下载失败也尝试用本地数据


def _maybe_auto_tune_locked_prefilter(
    *,
    strategy: str,
    cfg: Dict[str, Any],
    scfg: Dict[str, Any],
    config_path: str,
    end_date: str,
    disable_auto_locked_tuning: bool,
) -> str:
    """可选自动调优 locked prefilter 阈值，返回 override prefilter 路径（空串表示不启用）."""
    if disable_auto_locked_tuning:
        print("  ⏭️  locked tuning: disabled by flag")
        return ""

    global_toggles = cfg.get("global_toggles", {}) or {}
    if not bool(global_toggles.get("locked_threshold_tuning_enabled", True)):
        print("  ⏭️  locked tuning: globally disabled by config")
        return ""

    prefilter_gates = (scfg.get("kpi_gates", {}) or {}).get("prefilter", {}) or {}
    tcfg = prefilter_gates.get("locked_threshold_tuning", {}) or {}
    if not tcfg.get("enabled", False):
        return ""
    if tcfg.get("mode", "auto_if_locked") not in {"auto_if_locked", "always"}:
        return ""

    prod_prefilter = (
        PROJECT_ROOT / scfg["config"] / "archetypes" / "prefilter.yaml"
    ).resolve()
    locked_rules = load_locked_prefilter_rules(prod_prefilter)
    if not locked_rules:
        print("  ℹ️  locked tuning: no locked rules, skip")
        return ""

    # 管线 strategies.* 键名与配置统一为短名（bpc / me），不再使用 bpc-long 等前缀。
    if strategy == "me":
        template = "me"
    elif strategy == "bpc":
        template = "bpc"
    else:
        template = "fer"
    key_payload = {
        "strategy": strategy,
        "end_date": end_date,
        "template": template,
        "locked_rules": locked_rules,
        "tcfg": tcfg,
    }
    cache_key = hashlib.sha1(
        json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = PROJECT_ROOT / "results" / "locked_tuning" / "cache" / strategy
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_prefilter = cache_root / f"{cache_key}.yaml"
    cache_meta = cache_root / f"{cache_key}.json"

    if tcfg.get("skip_if_cached", True) and cache_prefilter.exists():
        print(f"  ✅ locked tuning cache hit: {cache_prefilter}")
        return str(cache_prefilter)

    print(f"  🔧 [locked-tune:auto] strategy={strategy} template={template}")
    out_root = PROJECT_ROOT / "results" / "locked_tuning" / strategy
    before = {p.name for p in out_root.iterdir()} if out_root.exists() else set()
    cmd = [
        sys.executable,
        "scripts/tune_locked_prefilter_thresholds.py",
        "--strategy",
        strategy,
        "--config",
        config_path or str(DEFAULT_CONFIG),
        "--template",
        template,
        "--end-dates",
        end_date,
        "--max-cases",
        str(int(tcfg.get("max_cases", 0) or 0)),
        "--min-trades-target",
        str(
            int(tcfg.get("target_trades_min", tcfg.get("min_trades_target", 60)) or 60)
        ),
        "--max-trades-target",
        str(int(tcfg.get("target_trades_max", 0) or 0)),
        "--trade-penalty-low",
        str(
            float(
                tcfg.get(
                    "trade_penalty_low",
                    tcfg.get("trade_penalty", 0.002),
                )
                or 0.002
            )
        ),
        "--trade-penalty-high",
        str(float(tcfg.get("trade_penalty_high", 0.001) or 0.001)),
        "--stability-penalty",
        str(float(tcfg.get("stability_penalty", 0.0) or 0.0)),
        "--skip-shap",
    ]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        print(f"  ⚠️  locked tuning failed, fallback to production locked rules\n{tail}")
        return ""

    after = {p.name for p in out_root.iterdir()} if out_root.exists() else set()
    new_dirs = sorted(after - before)
    run_dir = out_root / new_dirs[-1] if new_dirs else None
    if run_dir is None or not run_dir.exists():
        print("  ⚠️  locked tuning output not found, skip override")
        return ""
    summary_json = run_dir / "summary.json"
    if not summary_json.exists():
        print("  ⚠️  locked tuning summary missing, skip override")
        return ""
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    rows = summary.get("rows") or []
    if not rows:
        print("  ⚠️  locked tuning rows empty, skip override")
        return ""
    best = rows[0]
    case_id = int(best.get("case_id", 0))
    best_override = run_dir / f"case_{case_id:03d}" / "prefilter_locked_override.yaml"
    if not best_override.exists():
        print("  ⚠️  best override file missing, skip override")
        return ""

    shutil.copy(best_override, cache_prefilter)
    cache_meta.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "strategy": strategy,
                "end_date": end_date,
                "template": template,
                "source_run": str(run_dir),
                "best_case_id": case_id,
                "best_score": best.get("score"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _best_score = best.get("score", float("-inf"))
    try:
        _best_score = float(_best_score)
    except Exception:
        _best_score = float("-inf")
    print(f"  ✅ locked tuning selected case={case_id:03d}, score={_best_score:+.4f}")
    return str(cache_prefilter)


# ====================================================================
# Pipeline: single strategy
# ====================================================================


def _yaml_rolling_feature_search_enabled(cfg: dict) -> bool:
    """与 rolling_sim 一致: turbo + disable_feature_search 时视为关闭特征搜索."""
    rolling_cfg = cfg.get("rolling", {}) or {}
    mode = str(rolling_cfg.get("mode", "legacy") or "legacy")
    if mode != "turbo_fixed_features":
        return True
    turbo = rolling_cfg.get("turbo_fixed_features", {}) or {}
    return not bool(turbo.get("disable_feature_search", True))


def _apply_symbol_exclude(strategy: str, scfg: dict, symbols: str) -> str:
    """Filter out symbols listed in meta.yaml strategy.symbol_exclude."""
    prod_rel = str(scfg.get("config", "") or "").strip()
    if not prod_rel:
        return symbols
    meta_path = (PROJECT_ROOT / prod_rel / "meta.yaml").resolve()
    if not meta_path.is_file():
        return symbols
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return symbols
    strat_block = meta.get("strategy") or meta
    exclude = strat_block.get("symbol_exclude") or []
    if not exclude:
        return symbols
    exclude_set = {s.strip().upper() for s in exclude if s}
    filtered = [s for s in symbols.split(",") if s.strip().upper() not in exclude_set]
    if exclude_set:
        removed = [s for s in symbols.split(",") if s.strip().upper() in exclude_set]
        if removed:
            print(
                f"   🚫 symbol_exclude[{strategy}]: 移除 {removed} "
                f"(meta.yaml), 保留 {len(filtered)} symbols"
            )
    return ",".join(filtered)


def _resolve_pipeline_strategy_timeframe(strategy: str, scfg: dict) -> str:
    """策略周期仅以 strategies.*.config 下 meta.yaml 为准（忽略 pipeline YAML 的 timeframe 键）。"""
    prod_rel = str(scfg.get("config", "") or "").strip()
    if not prod_rel:
        raise KeyError(
            f"{strategy}: 未配置 strategies.{strategy}.config，无法从 meta.yaml 读取 timeframe"
        )
    meta_path = (PROJECT_ROOT / prod_rel / "meta.yaml").resolve()
    tf = _read_timeframe_from_meta_yaml_path(meta_path)
    if not tf:
        if not meta_path.is_file():
            raise FileNotFoundError(f"{strategy}: 找不到 meta.yaml: {meta_path}")
        raise ValueError(
            f"{strategy}: meta.yaml 未设置 strategy.timeframe（或顶层 timeframe）: {meta_path}"
        )
    return tf


def run_strategy_pipeline(
    strategy: str,
    cfg: dict,
    *,
    end_date: str,
    holdout_start: str,
    holdout_months: int,
    validation_months: int,
    start_date: str,
    symbols: str,
    data_path: str,
    run_dir: Path,
    seed: int = 42,
    dry_run: bool = False,
    use_1min: bool = False,
    live_root: str = "live/highcap",
    skip_shap: bool = False,
    feature_search_enabled: bool = True,
    threshold_calibration_enabled: bool = True,
    prefilter_optimization_enabled: bool = True,
    source_strategies_root: str = "",
    config_path: str = "",
    locked_prefilter_override: str = "",
    disable_auto_locked_tuning: bool = False,
    stage_stop: str = "full",
    disable_model_training: bool = False,
    skip_direction_tuning: bool = False,
) -> Dict[str, Any]:
    """执行单个策略训练链，可在指定 stage 提前停止."""
    scfg = cfg["strategies"][strategy]
    prod_config_dir = scfg["config"]
    timeframe = _resolve_pipeline_strategy_timeframe(strategy, scfg)
    log = run_dir / "pipeline.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    # ── 实验目录隔离: config 副本到实验工作区 ──────────────────
    # 源目录优先级（高→低）:
    #   1) scfg["config"] 被显式指向非默认路径（如 fbf_strict）→ 以它为准
    #   2) source_strategies_root / strategy（根级 override）存在 → 用它
    #   3) scfg["config"]（兜底）
    # 旧逻辑只检查 (2)+(3)，忽略 scfg["config"] 的显式 per-strategy 覆盖，
    # 结果 config/strategies/fbf 存在时永远遮蔽 config/strategies/fbf_strict。
    exp_strategies_root = run_dir / "strategies"
    exp_config_dir = exp_strategies_root / strategy
    _src_root = (
        Path(source_strategies_root)
        if str(source_strategies_root or "").strip()
        else (PROJECT_ROOT / "config" / "strategies")
    )
    if not _src_root.is_absolute():
        _src_root = PROJECT_ROOT / _src_root
    _src_strategy_dir = _src_root / strategy
    _prod_config_abs = (PROJECT_ROOT / prod_config_dir).resolve()
    _default_config_abs = (PROJECT_ROOT / "config" / "strategies" / strategy).resolve()
    if _prod_config_abs != _default_config_abs and _prod_config_abs.exists():
        _copy_from = _prod_config_abs
    elif _src_strategy_dir.exists():
        _copy_from = _src_strategy_dir
    else:
        _copy_from = _prod_config_abs
    shutil.copytree(_copy_from, exp_config_dir, dirs_exist_ok=True)
    config_dir = str(exp_config_dir)  # 后续命令全部用实验目录
    strategies_root = str(exp_strategies_root)
    print(f"\n📦 实验配置隔离: {exp_config_dir} (source={_copy_from})")
    if bool(scfg.get("has_direction", False)):
        _dir_path = Path(config_dir) / "archetypes" / "direction.yaml"
        if not _dir_path.exists():
            raise ValueError(
                f"{strategy}: has_direction=true 但缺少配置文件 {_dir_path}. "
                "方向来源必须来自配置，不再使用 default/fallback。"
            )
        print(f"   🧭 Direction source: {_dir_path}")
    stage_stop = str(stage_stop or "full").strip().lower()
    disable_model_training = bool(disable_model_training)
    if disable_model_training:
        print("⚡ Threshold-only calibration mode: skip model training")
        print("   NO_MODEL_TUNING: prefilter/gate/entry_filter")

    def _calibrate_locked_prefilter_thresholds_no_model() -> int:
        """No-model prefilter threshold calibration.

        Re-anchor locked prefilter thresholds to keep similar pass-rate on the
        current calibration window (features_labeled.parquet), without any model training.
        """
        import pandas as pd

        if bool(scfg.get("skip_locked_prefilter_reanchor", False)):
            print(
                "   ⏭️  skip_locked_prefilter_reanchor: 跳过 locked prefilter 重锚定（保留手写语义阈值）"
            )
            return 0

        pf_path = Path(config_dir) / "archetypes" / "prefilter.yaml"
        logs_path = Path(prepare_dir) / "features_labeled.parquet"
        if not pf_path.exists() or not logs_path.exists():
            return 0
        try:
            raw = yaml.safe_load(pf_path.read_text(encoding="utf-8")) or {}
            rules = raw.get("rules") or []
            if not isinstance(rules, list) or not rules:
                return 0
            dfp = pd.read_parquet(logs_path)
            if dfp is None or len(dfp) == 0:
                return 0

            def _reanchor_series(s, op_str, old_v):
                s = s.astype(float)
                s = s[s.notna()]
                if len(s) < 30:
                    return old_v
                if op_str == ">=":
                    pass_rate = float((s >= old_v).mean())
                    q = max(0.0, min(1.0, 1.0 - pass_rate))
                    return float(s.quantile(q))
                if op_str == ">":
                    pass_rate = float((s > old_v).mean())
                    q = max(0.0, min(1.0, 1.0 - pass_rate))
                    return float(s.quantile(q))
                if op_str == "<=":
                    pass_rate = float((s <= old_v).mean())
                    q = max(0.0, min(1.0, pass_rate))
                    return float(s.quantile(q))
                if op_str == "<":
                    pass_rate = float((s < old_v).mean())
                    q = max(0.0, min(1.0, pass_rate))
                    return float(s.quantile(q))
                return old_v

            updated = 0
            for r in rules:
                if not isinstance(r, dict):
                    continue
                if r.get("any_of") and isinstance(r.get("any_of"), list):
                    # 仅当父规则 locked 时，对子条件重标定
                    if not r.get("locked"):
                        continue
                    for sub in r.get("any_of", []):
                        if not isinstance(sub, dict):
                            continue
                        feat = str(sub.get("feature", ""))
                        op_str = str(sub.get("operator", ""))
                        old_v = sub.get("value")
                        if (
                            feat
                            and feat in dfp.columns
                            and isinstance(old_v, (int, float))
                            and op_str in {">=", ">", "<=", "<"}
                        ):
                            new_v = _reanchor_series(dfp[feat], op_str, float(old_v))
                            if float(new_v) != float(old_v):
                                sub["value"] = float(round(new_v, 6))
                                updated += 1
                    continue
                if not r.get("locked"):
                    continue
                feat = str(r.get("feature", ""))
                op_str = str(r.get("operator", ""))
                old_v = r.get("value")
                if (
                    feat
                    and feat in dfp.columns
                    and isinstance(old_v, (int, float))
                    and op_str in {">=", ">", "<=", "<"}
                ):
                    new_v = _reanchor_series(dfp[feat], op_str, float(old_v))
                    if float(new_v) != float(old_v):
                        r["value"] = float(round(new_v, 6))
                        updated += 1

            if updated > 0:
                raw["rules"] = rules
                pf_path.write_text(
                    yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                print(f"   ✅ Prefilter locked 阈值重标定完成: {updated} 项")
            else:
                print("   ℹ️  Prefilter locked 阈值无变更")
            return updated
        except Exception as exc:
            print(f"   ⚠️  Prefilter no-model 阈值重标定失败: {exc}")
            return 0

    # ── 加载 per-strategy KPI gates ──
    kpi_gates = scfg.get("kpi_gates", {})
    prefilter_gates = kpi_gates.get("prefilter", {})
    gate_gates = kpi_gates.get("gate", {})
    backtest_gates = kpi_gates.get("backtest", {})
    execution_gates = kpi_gates.get("execution", {})
    auto_locked_override = ""

    # ── Validation / Test 三段分离 ──
    # holdout_months = 总 OOS 窗口, validation_months = 前 N 个月用于 Gate 调阈值
    # Train [start, holdout_start) → Val [holdout_start, test_start) → Test [test_start, end]
    # predictions.parquet 覆盖完整 holdout [holdout_start, end], 训练集不变
    if validation_months > 0 and validation_months < holdout_months:
        # test_start = holdout 内部切分点 = end_date - (holdout_months - validation_months)
        test_start = compute_holdout_start(end_date, holdout_months - validation_months)
    else:
        test_start = holdout_start  # 不分离, 兼容旧行为

    symbols = _apply_symbol_exclude(strategy, scfg, symbols)

    common_train_args = [
        "--symbol",
        symbols,
        "--timeframe",
        timeframe,
        "--data-path",
        data_path,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--holdout-start-date",
        holdout_start,  # predictions 覆盖完整 holdout, 训练集不受影响
        "--holdout-end-date",
        end_date,
        "--seed",
        str(seed),
        "--non-deterministic",  # multi-thread for speed (CLI default=True would force single-thread)
    ]

    # ── Step 0: Data Download + Convert (增量) ──
    run_data_download(
        cfg,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        log=log,
        dry_run=dry_run,
    )

    # ── Step 1: Feature Store (增量, 已有月份自动跳过) ──
    rc, _ = run_step(
        "Feature Store",
        [
            "mlbot",
            "feature-store",
            "build",
            "--no-docker",
            "--config",
            config_dir,
            "--symbols",
            symbols,
            "--timeframe",
            timeframe,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--warmup-months",
            "6",
        ],
        log,
        dry_run=dry_run,
    )
    if rc != 0 and not dry_run:
        return {"error": "feature_store_build_failed"}

    # ── Step 2: Prepare-only (features_labeled.parquet) ──
    rc, out = run_step(
        "Prepare Only",
        [
            "mlbot",
            "train",
            "final",
            "--no-docker",
            "--prepare-only",
            "--config",
            config_dir,
            "--features",
            f"{config_dir}/{scfg['features_gate']}",
            "--labels",
            f"{config_dir}/{scfg['labels_gate']}",
            *common_train_args,
        ],
        log,
        dry_run=dry_run,
    )

    prepare_dir = find_output_dir(out, strategy)
    if not prepare_dir and not dry_run:
        return {"error": "prepare_dir_not_found"}
    prepare_dir = prepare_dir or f"results/train_final_DRYRUN/{strategy}"

    # ── Step 2.5: SHAP Feature Selection (可选, 默认开启) ──
    # SHAP --promote 输出 features_gate_shap.yaml (基于 features_gate 配置生成)
    # 原始 features.yaml 永远不动（保留完整候选池）
    shap_cfg = cfg.get("shap_feature_selection", {})
    _fs_effective = bool(
        feature_search_enabled
    ) and _yaml_rolling_feature_search_enabled(cfg)
    # Default off: gate 特征以 features_gate.yaml 人工/策略白名单为准；需实验时再在
    # research_pipeline.yaml / prod yaml 中设 shap_feature_selection.enabled: true。
    _skip_shap = skip_shap or (not _fs_effective) or not shap_cfg.get("enabled", False)
    shap_active = False  # 标记 SHAP 是否成功生成了 _shap.yaml
    if not _skip_shap:
        shap_cmd = [
            "python",
            "scripts/shap_feature_selection.py",
            "--logs",
            f"{prepare_dir}/features_labeled.parquet",
            "--strategy",
            strategy,
            "--strategies-root",
            strategies_root,
            "--pipeline-config",
            config_path or str(DEFAULT_CONFIG),
            "--output",
            f"{prepare_dir}/shap",
            "--promote",
        ]
        # Per-strategy SHAP override (e.g. ME 需要更宽松的 stability_threshold)
        _shap_ov = scfg.get("shap_override", {})
        if _shap_ov:
            if "stability_threshold" in _shap_ov:
                shap_cmd += [
                    "--stability-threshold",
                    str(_shap_ov["stability_threshold"]),
                ]
            if "top_k" in _shap_ov:
                shap_cmd += ["--top-k", str(_shap_ov["top_k"])]
            if "n_folds" in _shap_ov:
                shap_cmd += ["--n-folds", str(_shap_ov["n_folds"])]
            print(f"   📋 SHAP override for {strategy}: {_shap_ov}")
        # Val/Test 分离: SHAP 稳定性仅看 Val 段 [holdout_start, test_start).
        # 使 gate/prefilter 共享同一 cutoff 语义, 避免 Test 段特征泄漏进特征选择.
        if test_start and holdout_start and test_start != holdout_start:
            shap_cmd += ["--cutoff-date", test_start]
        rc_shap, _ = run_step(
            "SHAP Feature Selection",
            shap_cmd,
            log,
            dry_run=dry_run,
        )
        # 检查 _shap.yaml 是否生成
        gate_shap = Path(config_dir) / scfg["features_gate"].replace(
            ".yaml", "_shap.yaml"
        )
        if dry_run or gate_shap.exists():
            shap_active = True
            print(f"\u2705 SHAP: 使用 {gate_shap.name}")
        else:
            print("\u26a0\ufe0f  SHAP: _shap.yaml 未生成, 回退到原始特征文件")
    else:
        print("\u23ed\ufe0f  SHAP Feature Selection: skipped")

    # 决定 Gate/Evidence 用哪个特征文件
    features_gate_file = (
        scfg["features_gate"].replace(".yaml", "_shap.yaml")
        if shap_active
        else scfg["features_gate"]
    )

    # ── Auto locked tuning (optional) ──
    if not dry_run:
        auto_locked_override = _maybe_auto_tune_locked_prefilter(
            strategy=strategy,
            cfg=cfg,
            scfg=scfg,
            config_path=config_path or str(DEFAULT_CONFIG),
            end_date=end_date,
            disable_auto_locked_tuning=disable_auto_locked_tuning,
        )

    # ── Step 3: Direction (--promote) ── (前置于 Prefilter, 使 meta-algorithm 能在方向已知的子集上学习)
    _fast_loop_cfg = cfg.get("fast_loop") or {}
    if scfg.get("has_direction") and not skip_direction_tuning:
        _dir_tune_cmd = _fast_loop_cfg.get("direction_tuning") or {}
        _compare_feats = _dir_tune_cmd.get("compare_features")
        if _compare_feats is None:
            _compare_feats = True
        _promote_dir = _dir_tune_cmd.get("promote_after_validate")
        if _promote_dir is None:
            _promote_dir = True
        _dir_argv: List[str] = [
            "python",
            "scripts/direction_strict_validation.py",
            "--logs",
            f"{prepare_dir}/features_labeled.parquet",
            "--strategy",
            strategy,
            "--strategies-root",
            strategies_root,
            "--direction-workspace",
            "features_direction.yaml",
        ]
        if _compare_feats:
            _dir_argv.extend(["--compare-features", "--temporal"])
        if _promote_dir:
            _dir_argv.append("--promote")
        run_step("Direction Validate", _dir_argv, log, dry_run=dry_run)
    elif scfg.get("has_direction") and skip_direction_tuning:
        print(
            "⏭️  Direction 调优跳过: fast_loop.direction_tuning 关闭或未到 cadence 月份"
        )

    if not threshold_calibration_enabled:
        print("⏭️  阈值校准已禁用: 跳过 prefilter/gate/entry_filter 优化步骤")
        return {
            "stage": "entry_filter",
            "gate_dir": prepare_dir,
            "evidence_dir": prepare_dir,
            "backtest_metrics": {
                "total_trades": 0,
                "mean_r": 0.0,
                "win_rate": 0.0,
                "sharpe_per_trade": 0.0,
            },
            "exp_config_dir": str(exp_config_dir),
            "prod_config_dir": prod_config_dir,
            "prefilter_comparison": None,
            "validation_end": test_start,
        }

    # ── Step 4: Prefilter (--promote) ── (方向已确定后, meta-algorithm 按方向分裂学习)
    _pf_results: Dict[str, Any] = {}  # 提前初始化, 供 Sharpe 对比块使用
    _pf_yaml: Optional[Path] = None
    _pf_comparison: Optional[Dict[str, Any]] = None  # Sharpe 对比结果, 用于汇总输出
    _enforce_pf_locked = bool(
        prefilter_gates.get("enforce_locked_rules_in_experiment", True)
    )
    _locked_prefilter_rules: List[Dict[str, Any]] = []
    _locked_prefilter_source = (
        locked_prefilter_override
        or auto_locked_override
        or str(PROJECT_ROOT / prod_config_dir / "archetypes" / "prefilter.yaml")
    )
    if _enforce_pf_locked and not dry_run:
        _locked_prefilter_rules = load_locked_prefilter_rules(
            Path(_locked_prefilter_source)
        )
        if _locked_prefilter_rules:
            print(
                f"  🔒 Prefilter locked 规则已启用: {len(_locked_prefilter_rules)} 条 "
                f"(source={_locked_prefilter_source})"
            )
    if scfg.get("has_prefilter") and prefilter_optimization_enabled:
        if disable_model_training:
            _calibrate_locked_prefilter_thresholds_no_model()
        else:
            _features_prefilter_path = Path(config_dir) / "features_prefilter.yaml"
            if not _features_prefilter_path.exists():
                print(f"  ❌ Prefilter: {_features_prefilter_path} 不存在, 跳过")
            else:
                prefilter_cmd = [
                    "python",
                    "scripts/analyze_archetype_feature_stratification.py",
                    "--logs",
                    f"{prepare_dir}/features_labeled.parquet",
                    "--strategy",
                    strategy,
                    "--meta-algorithm",
                    "--features-prefilter",
                    str(_features_prefilter_path),
                    "--config",
                    str(config_dir),
                    "--promote",
                ]

                # 从 kpi_gates 注入 prefilter 约束
                if prefilter_gates.get("min_pass_rate"):
                    prefilter_cmd += [
                        "--min-prefilter-pass-rate",
                        str(prefilter_gates["min_pass_rate"]),
                    ]
                if prefilter_gates.get("min_rows"):
                    prefilter_cmd += [
                        "--min-prefilter-rows",
                        str(prefilter_gates["min_rows"]),
                    ]
                # Per-strategy KPI 覆盖: scoring_method / min_ks / max_ks_pvalue / min_lift
                # 支持 scoring_method_fallbacks: 多方法并行跑, 供候选发现模式人审.
                # 2026-04-23: 原 Wave 2-E 单方法锁定已 revert —— 慢管线定位为
                # "候选发现器" (wave3/02), 多方法共识矩阵是给人审的核心信号,
                # 不再在 Val 上自动择冠军 (决策权交给人审).
                _pf_fallbacks = prefilter_gates.get("scoring_method_fallbacks")
                if _pf_fallbacks and isinstance(_pf_fallbacks, list):
                    _pf_methods = standardize_method_list(
                        _pf_fallbacks, default=["distribution_ks"]
                    )
                elif prefilter_gates.get("scoring_method"):
                    _pf_methods = standardize_method_list(
                        [prefilter_gates["scoring_method"]], default=["distribution_ks"]
                    )
                else:
                    _pf_methods = ["distribution_ks"]  # 默认

                def _append_pf_kpi_args(cmd):
                    """追加非 scoring_method 的 KPI 参数."""
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
                        cmd += [
                            "--prefilter-min-lift",
                            str(prefilter_gates["min_lift"]),
                        ]
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

                _pf_yaml = Path(config_dir) / "archetypes" / "prefilter.yaml"

                # 每次从空 prefilter 出发: 每个 method 在全量数据上独立搜索
                # 不依赖已有 prefilter.yaml 内容, 保证多次运行结果一致
                for _pf_method in _pf_methods:
                    # 清空 prefilter.yaml (空规则 = 全量数据), 每个 method 独立从全量出发
                    if _pf_yaml.exists():
                        _pf_yaml.unlink()
                    if _enforce_pf_locked and _locked_prefilter_rules and not dry_run:
                        _m0 = merge_locked_prefilter_rules(
                            _pf_yaml, _locked_prefilter_rules
                        )
                        print(
                            f"   🔒 [{_pf_method}] 初始注入 locked: +{_m0['added']} (total={_m0['total']})"
                        )
                    _cmd = prefilter_cmd + ["--prefilter-scoring-method", _pf_method]
                    _append_pf_kpi_args(_cmd)
                    _step_name = (
                        f"Prefilter Analyze [{_pf_method}]"
                        if len(_pf_methods) > 1
                        else "Prefilter Analyze"
                    )
                    run_step(_step_name, _cmd, log, dry_run=dry_run)

                    if _pf_yaml.exists() and not dry_run:
                        try:
                            if _enforce_pf_locked and _locked_prefilter_rules:
                                _m1 = merge_locked_prefilter_rules(
                                    _pf_yaml, _locked_prefilter_rules
                                )
                                if _m1["added"] > 0:
                                    print(
                                        f"   🔒 [{_pf_method}] 回补 locked: +{_m1['added']} (total={_m1['total']})"
                                    )
                            _pf_data = (
                                yaml.safe_load(_pf_yaml.read_text(encoding="utf-8"))
                                or {}
                            )
                            _rules = _pf_data.get("rules") or []
                            _n_rules = len(_rules)
                            # 保存当前结果到临时文件 (供 Sharpe 对比使用)
                            _tmp = _pf_yaml.parent / f"prefilter_{_pf_method}.yaml"
                            shutil.copy(_pf_yaml, _tmp)
                            # 同时落盘到 _candidates/method=<name>/ 给 T4 共识矩阵读取
                            # (scripts/slow_candidate_report.py consensus 消费此目录).
                            _cand_dir = (
                                Path(config_dir)
                                / "_candidates"
                                / f"method={_pf_method}"
                            )
                            _cand_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy(_pf_yaml, _cand_dir / "prefilter.yaml")
                            _pf_results[_pf_method] = {
                                "n_rules": _n_rules,
                                "path": _tmp,
                            }
                            if len(_pf_methods) > 1:
                                print(f"   📊 [{_pf_method}] rules={_n_rules}")
                        except Exception:
                            pass
    elif scfg.get("has_prefilter") and not prefilter_optimization_enabled:
        print("⏭️  Prefilter 规则搜索已禁用: 复用当前 archetypes/prefilter.yaml")

    # ── Step 5: Gate 训练 ──
    # Prefilter 兜底: 即使上一步没产出规则文件, 也写入空规则继续后续流程.
    _gate_prefilter_path = Path(config_dir) / "archetypes" / "prefilter.yaml"
    if not dry_run:
        _need_empty_pf = False
        if not _gate_prefilter_path.exists():
            _need_empty_pf = True
        else:
            try:
                _pf_raw = yaml.safe_load(
                    _gate_prefilter_path.read_text(encoding="utf-8")
                )
                if not isinstance(_pf_raw, dict) or _pf_raw.get("rules") is None:
                    _need_empty_pf = True
            except Exception:
                _need_empty_pf = True
        if _need_empty_pf:
            _gate_prefilter_path.parent.mkdir(parents=True, exist_ok=True)
            _gate_prefilter_path.write_text("rules: []\n", encoding="utf-8")
            if _enforce_pf_locked and _locked_prefilter_rules:
                _mfb = merge_locked_prefilter_rules(
                    _gate_prefilter_path, _locked_prefilter_rules
                )
                print(
                    "  ⚠️  Prefilter Analyze 未产出有效 prefilter.yaml, "
                    f"已回退并注入 locked 规则 (total={_mfb['total']})"
                )
            else:
                print(
                    "  ⚠️  Prefilter Analyze 未产出有效 prefilter.yaml, 已自动回退为空规则"
                )

    # ⚠️ Gate Train 现在使用 prefilter (见 BPC pipeline 文档):
    #   - 只在 archetype 适用样本上学习 → 专注策略特有特征
    #   - 避免学习 "archetype vs 非 archetype" 而不是 "好 archetype vs 坏 archetype"
    gate_train_args = [
        "mlbot",
        "train",
        "final",
        "--no-docker",
        "--config",
        config_dir,
        "--features",
        f"{config_dir}/{features_gate_file}",
        "--labels",
        f"{config_dir}/{scfg['labels_gate']}",
        "--archetype-prefilter",
        str(_gate_prefilter_path),
        *common_train_args,
        "--seed",
        "42",  # A.7.1: gate 规则确定性，固定 seed 不受外层 seed 影响
    ]
    if _skip_shap:
        gate_train_args.append("--skip-gate-shap")

    # 缓存“本月起始”的 locked gate 规则，避免被候选对比中的 Gate Opt --promote 覆盖后丢失。
    # 后续所有 gate_draft 注入都优先使用该快照，保证语义锁定在整个月流程可追踪。
    _month_locked_gate_rules: List[Dict[str, Any]] = []
    if not dry_run:
        _month_locked_gate_rules = load_locked_gate_rules(
            Path(f"{config_dir}/archetypes/gate.yaml")
        )

    # ── Prefilter Score 对比: 每个候选 prefilter 跑 mini-pipeline, 按 Score 择优 ──
    # 空 prefilter (empty) 仅在显式允许时参与候选。
    # 关键保护：如果开启了 locked 规则强制注入，默认不允许 empty 覆盖。
    _pf_include_empty = prefilter_gates.get(
        "prefilter_search_include_empty", (not _enforce_pf_locked)
    )
    if (
        not dry_run
        and (not disable_model_training)
        and _pf_results  # 确认多算法分析已运行且有结果
        and _pf_yaml is not None
    ):
        _cand_paths: Dict[str, Any] = {
            m: r["path"]
            for m, r in _pf_results.items()
            if r.get("n_rules", 0) > 0 and r.get("path") and r["path"].exists()
        }
        if _pf_include_empty:
            _empty_pf_p = _pf_yaml.parent / "prefilter_cmp_empty.yaml"
            _empty_pf_p.write_text("rules: []\n", encoding="utf-8")
            _cand_paths["empty"] = _empty_pf_p
        if len(_cand_paths) >= 2:
            _pf_obj = _resolve_trade_objective(
                prefilter_gates,
                default_min=30,
                default_max=200,
                default_penalty_low=0.002,
                default_penalty_high=0.001,
            )
            _cmp_sharpe: Dict[str, float] = {}
            _cmp_trades: Dict[str, int] = {}
            _cmp_rules: Dict[str, int] = {}
            _cmp_score: Dict[str, float] = {}
            _cmp_low_gap: Dict[str, float] = {}
            _cmp_high_gap: Dict[str, float] = {}
            _cmp_low_penalty: Dict[str, float] = {}
            _cmp_high_penalty: Dict[str, float] = {}
            _simple_exec = scfg.get("simple_execution", {})
            print(f"\n{'='*72}")
            print(
                f"🔬 Prefilter Sharpe 对比模式 — {len(_cand_paths)} 候选: {list(_cand_paths)}"
            )
            print(
                "   "
                f"trade_target=[{int(_pf_obj['target_min'])},{int(_pf_obj['target_max'])}], "
                f"penalty_low={_pf_obj['penalty_low']}, penalty_high={_pf_obj['penalty_high']}"
            )
            print(f"{'='*72}")
            for _cm, _cp in _cand_paths.items():
                print(f"\n── 候选 [{_cm}] ──")
                shutil.copy(_cp, _pf_yaml)
                # Gate Train
                _rc_cg, _out_cg = run_step(f"  Gate [{_cm}]", gate_train_args, log)
                _cg_dir = find_output_dir(_out_cg, strategy) or prepare_dir
                if _rc_cg != 0 or not Path(f"{_cg_dir}/predictions.parquet").exists():
                    print(f"   ❌ Gate Train 失败, 跳过")
                    _cmp_sharpe[_cm] = float("-inf")
                    _cmp_trades[_cm] = 0
                    _cmp_score[_cm] = float("-inf")
                    _cmp_low_gap[_cm] = 0.0
                    _cmp_high_gap[_cm] = 0.0
                    _cmp_low_penalty[_cm] = 0.0
                    _cmp_high_penalty[_cm] = 0.0
                    continue
                # gate_draft 由 Gate Train 写入 {config_dir}/（与 --config 同目录的实验隔离树）
                # Ensure candidate gate_draft always carries month locked gate rules.
                if _month_locked_gate_rules:
                    merge_locked_gate_rules(
                        Path(f"{config_dir}/gate_draft.yaml"), _month_locked_gate_rules
                    )
                # Gate Apply (draft)
                run_step(
                    f"  Gate Apply [{_cm}]",
                    [
                        "mlbot",
                        "gate",
                        "apply-archetype",
                        "--logs",
                        f"{_cg_dir}/predictions.parquet",
                        "--strategy",
                        strategy,
                        "--gate-path",
                        f"{config_dir}/gate_draft.yaml",
                    ],
                    log,
                )
                # Gate Optimize
                _cg_opt = [
                    "python",
                    "scripts/optimize_gate_unified.py",
                    "--strategy",
                    strategy,
                    "--strategies-root",
                    strategies_root,
                    "--logs",
                    f"{_cg_dir}/logs_gated.parquet",
                    "--output",
                    f"{_cg_dir}/gate_optimization.json",
                    "--gate-path",
                    f"{config_dir}/gate_draft.yaml",
                    "--promote",
                ]
                if gate_gates.get("min_combined_pass_rate"):
                    _cg_opt += [
                        "--min-combined-pass-rate",
                        str(gate_gates["min_combined_pass_rate"]),
                    ]
                if gate_gates.get("max_hard_gates") is not None:
                    _cg_opt += [
                        "--max-hard-gates",
                        str(int(gate_gates["max_hard_gates"])),
                    ]
                if gate_gates.get("max_system_safety") is not None:
                    _cg_opt += [
                        "--max-system-safety",
                        str(int(gate_gates["max_system_safety"])),
                    ]
                if bool(gate_gates.get("require_positive_effect", False)):
                    _cg_opt += ["--require-positive-effect"]
                    _tol = gate_gates.get("positive_effect_tol")
                    if _tol is not None:
                        _cg_opt += ["--positive-effect-tol", str(float(_tol))]
                # Val/Test 分离: mini-pipeline Gate Opt 在完整 Val 上调阈值 (不需要 cutoff)
                # 预筛选选择也在 Val 上评估, Test 保留给最终 Backtest
                run_step(f"  Gate Opt [{_cm}]", _cg_opt, log)
                # Gate Re-Apply
                run_step(
                    f"  Gate Re-Apply [{_cm}]",
                    [
                        "mlbot",
                        "gate",
                        "apply-archetype",
                        "--logs",
                        f"{_cg_dir}/predictions.parquet",
                        "--strategy",
                        strategy,
                        "--gate-path",
                        f"{config_dir}/archetypes/gate.yaml",
                    ],
                    log,
                )
                # Vector Backtest
                _cg_bt = [
                    "python",
                    "scripts/backtest_execution_layer.py",
                    "--logs",
                    f"{_cg_dir}/logs_gated.parquet",
                    "--strategy",
                    strategy,
                    "--strategies-root",
                    strategies_root,
                    "--test-start",
                    holdout_start,  # Val/Test 分离: prefilter 选择在 Val 段评估
                    "--test-end",
                    test_start if test_start != holdout_start else end_date,
                    "--simple-execution",
                ]
                if _simple_exec.get("sl_r") is not None:
                    _cg_bt += ["--simple-sl", str(_simple_exec["sl_r"])]
                if _simple_exec.get("tp_r") is not None:
                    _cg_bt += ["--simple-tp", str(_simple_exec["tp_r"])]
                if _simple_exec.get("timeout_bars") is not None:
                    _cg_bt += ["--simple-timeout", str(_simple_exec["timeout_bars"])]
                _rc_cbt, _out_cbt = run_step(f"  Backtest [{_cm}]", _cg_bt, log)
                _cbt_m = parse_backtest_stdout(_out_cbt)
                _cmp_sharpe[_cm] = _cbt_m.get("sharpe_per_trade", float("-inf"))
                _cmp_trades[_cm] = _cbt_m.get("total_trades", 0)
                _cmp_rules[_cm] = _pf_results.get(_cm, {}).get("n_rules", 0)
                _sc = _score_with_trade_objective(
                    sharpe=_cmp_sharpe[_cm],
                    trades=_cmp_trades[_cm],
                    objective=_pf_obj,
                )
                _cmp_score[_cm] = _sc["score"]
                _cmp_low_gap[_cm] = _sc["low_gap"]
                _cmp_high_gap[_cm] = _sc["high_gap"]
                _cmp_low_penalty[_cm] = _sc["low_penalty"]
                _cmp_high_penalty[_cm] = _sc["high_penalty"]
                _s = _cmp_sharpe[_cm]
                print(
                    f"   📊 [{_cm}] Sharpe={_s:+.4f}, Trades={_cmp_trades[_cm]}, "
                    f"Score={_cmp_score[_cm]:+.4f}"
                )
            # 汇总对比表
            _best_cm = max(_cmp_score, key=lambda m: _cmp_score[m])
            # 补充 0 规则方法 (等价于 empty, 不需跑 pipeline)
            _empty_sharpe = _cmp_sharpe.get("empty", float("-inf"))
            _empty_trades = _cmp_trades.get("empty", 0)
            _empty_score = _cmp_score.get("empty", float("-inf"))
            _empty_low_gap = _cmp_low_gap.get("empty", 0.0)
            _empty_high_gap = _cmp_high_gap.get("empty", 0.0)
            _empty_low_penalty = _cmp_low_penalty.get("empty", 0.0)
            _empty_high_penalty = _cmp_high_penalty.get("empty", 0.0)
            _zero_rule_methods = [
                m
                for m, r in _pf_results.items()
                if r.get("n_rules", 0) == 0 and m not in _cmp_sharpe
            ]
            for _zrm in _zero_rule_methods:
                _cmp_sharpe[_zrm] = _empty_sharpe
                _cmp_trades[_zrm] = _empty_trades
                _cmp_rules[_zrm] = 0
                _cmp_score[_zrm] = _empty_score
                _cmp_low_gap[_zrm] = _empty_low_gap
                _cmp_high_gap[_zrm] = _empty_high_gap
                _cmp_low_penalty[_zrm] = _empty_low_penalty
                _cmp_high_penalty[_zrm] = _empty_high_penalty
            _tbl_lines: list = []
            _tbl_lines.append(f"\n{'='*72}")
            _tbl_lines.append(
                f"  {'方法':<25} {'Score':>10} {'Sharpe':>10} {'Trades':>7} {'Rules':>6}  标记"
            )
            _tbl_lines.append(f"  {'-'*68}")
            for _m in sorted(_cmp_score, key=lambda m: -_cmp_score[m]):
                _flag = " ← 最优" if _m == _best_cm else ""
                _score = _cmp_score[_m]
                _score_str = f"{_score:+.4f}" if _score != float("-inf") else "  FAIL"
                _s = _cmp_sharpe[_m]
                _s_str = f"{_s:+.4f}" if _s != float("-inf") else "  FAIL"
                _note = "  (0规则=empty)" if _m in _zero_rule_methods else ""
                _tbl_lines.append(
                    f"  {_m:<25} {_score_str:>10} {_s_str:>10} "
                    f"{_cmp_trades.get(_m, 0):>7} {_cmp_rules.get(_m, 0):>6}{_flag}{_note}"
                )
            _tbl_lines.append(f"{'='*72}\n")
            _tbl_text = "\n".join(_tbl_lines)
            print(_tbl_text)
            with open(log, "a", encoding="utf-8") as _lf:
                _lf.write(f"\n{'='*72}\n")
                _lf.write(f"🔬 Prefilter Score 对比汇总\n")
                _lf.write(_tbl_text + "\n")
            # 设置最优 prefilter, 让完整 pipeline 使用
            shutil.copy(_cand_paths[_best_cm], _pf_yaml)
            print(f"   ✅ 最优 prefilter [{_best_cm}] 已写入, 继续完整 pipeline")
            # 清理临时文件
            for _cp in _cand_paths.values():
                if _cp and _cp != _pf_yaml and _cp.exists():
                    _cp.unlink()
            # 存储对比结果, 供最终汇总显示
            _pf_comparison = {
                "best": _best_cm,
                "zero_rule_methods": _zero_rule_methods,
                "objective": _pf_obj,
                "candidates": {
                    m: {
                        "score": _cmp_score[m],
                        "sharpe": _cmp_sharpe[m],
                        "trades": _cmp_trades.get(m, 0),
                        "rules": _cmp_rules.get(m, 0),
                        "low_gap": _cmp_low_gap.get(m, 0.0),
                        "high_gap": _cmp_high_gap.get(m, 0.0),
                        "low_penalty": _cmp_low_penalty.get(m, 0.0),
                        "high_penalty": _cmp_high_penalty.get(m, 0.0),
                    }
                    for m in _cmp_score
                },
            }

    if stage_stop == "prefilter":
        return {
            "stage": "prefilter",
            "gate_dir": prepare_dir,
            "evidence_dir": prepare_dir,
            "backtest_metrics": {
                "total_trades": 0,
                "mean_r": 0.0,
                "win_rate": 0.0,
                "sharpe_per_trade": 0.0,
            },
            "exp_config_dir": str(exp_config_dir),
            "prod_config_dir": prod_config_dir,
            "prefilter_comparison": _pf_comparison,
            "validation_end": test_start,
        }

    if disable_model_training:
        rc, out = 0, ""
        gate_dir = prepare_dir
        gate_pred = Path(f"{gate_dir}/predictions.parquet")
        gate_train_ok = False
        gate_stat_fallback_used = False
        if not dry_run:
            fallback_src = Path(f"{prepare_dir}/features_labeled.parquet")
            if fallback_src.exists():
                gate_pred.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fallback_src, gate_pred)
                gate_train_ok = True
                gate_stat_fallback_used = True
                print(
                    "   ⏭️  Gate Train skipped (disable_model_training=true), "
                    "using features_labeled as gate input"
                )
        exp_gate_draft = Path(f"{config_dir}/gate_draft.yaml")
        if not dry_run and not exp_gate_draft.exists():
            archetype_gate = Path(f"{config_dir}/archetypes/gate.yaml")
            if archetype_gate.exists():
                shutil.copy2(archetype_gate, exp_gate_draft)
    else:
        rc, out = run_step("Gate Train", gate_train_args, log, dry_run=dry_run)
        gate_dir = find_output_dir(out, strategy) or prepare_dir

        # Gate Train 已将 gate_draft.yaml 写到 {config_dir}/（--config 指向的实验策略目录）

        # ── Early termination / statistical fallback on Gate Train failure ──
        gate_pred = Path(f"{gate_dir}/predictions.parquet")
        gate_train_ok = rc == 0 and (dry_run or gate_pred.exists())
        gate_stat_fallback_used = False
    if not gate_train_ok and not dry_run and _looks_like_gate_insufficient_sample(out):
        fallback_src = Path(f"{prepare_dir}/features_labeled.parquet")
        if fallback_src.exists():
            print(
                "\n⚠️  Gate Train 样本不足，触发统计法 fallback: "
                "使用 features_labeled 继续生成 gate。"
            )
            gate_pred.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback_src, gate_pred)

            gate_draft = f"{config_dir}/gate_draft.yaml"
            gate_draft_path = Path(gate_draft)
            if not gate_draft_path.exists():
                archetype_gate = Path(f"{config_dir}/archetypes/gate.yaml")
                if archetype_gate.exists():
                    shutil.copy2(archetype_gate, gate_draft_path)
            gate_opt_fallback_cmd = [
                "python",
                "scripts/optimize_gate_unified.py",
                "--strategy",
                strategy,
                "--strategies-root",
                strategies_root,
                "--logs",
                str(gate_pred),
                "--output",
                f"{gate_dir}/gate_optimization_fallback.json",
                "--gate-path",
                gate_draft,
                "--promote",
                "--stat-fallback-on-empty",
            ]
            if gate_gates.get("min_combined_pass_rate"):
                gate_opt_fallback_cmd += [
                    "--min-combined-pass-rate",
                    str(gate_gates["min_combined_pass_rate"]),
                ]
            if gate_gates.get("max_hard_gates") is not None:
                gate_opt_fallback_cmd += [
                    "--max-hard-gates",
                    str(int(gate_gates["max_hard_gates"])),
                ]
            if gate_gates.get("max_system_safety") is not None:
                gate_opt_fallback_cmd += [
                    "--max-system-safety",
                    str(int(gate_gates["max_system_safety"])),
                ]
            if bool(gate_gates.get("require_positive_effect", False)):
                gate_opt_fallback_cmd += ["--require-positive-effect"]
                _tol = gate_gates.get("positive_effect_tol")
                if _tol is not None:
                    gate_opt_fallback_cmd += ["--positive-effect-tol", str(float(_tol))]
            # Keep Val/Test causality: tune gate only on validation segment.
            if test_start != holdout_start:
                gate_opt_fallback_cmd += ["--cutoff-date", test_start]
            _rc_fallback, _ = run_step(
                "Gate Stat Fallback Optimize",
                gate_opt_fallback_cmd,
                log,
                dry_run=dry_run,
            )
            if _rc_fallback == 0:
                gate_train_ok = True
                gate_stat_fallback_used = True
                print("   ✅ 统计法 fallback 完成，继续后续 Gate/EF/Backtest 流程")

    if not gate_train_ok:
        print(
            f"\n\u274c Gate Train 失败或未产出 predictions.parquet"
            f" (rc={rc}, exists={gate_pred.exists() if not dry_run else 'N/A'})"
        )
        print(
            "   可能原因: prefilter 过滤过严 (样本量不足), "
            "或训练参数错误. 请检查上方日志."
        )
        if not dry_run:
            return {"error": "gate_train_failed", "gate_dir": gate_dir}
    elif gate_stat_fallback_used:
        print("   ℹ️ 本轮 Gate Train 使用统计法 fallback 继续执行")

    if not dry_run:
        _ensure_timestamp_for_gate_input(gate_pred)

    if not dry_run:
        # Ensure month gate_draft always carries locked semantic gate rules
        # from the month archetype gate, even if Gate Train rewrote gate_draft.
        gate_draft_path = Path(f"{config_dir}/gate_draft.yaml")
        archetype_gate_path = Path(f"{config_dir}/archetypes/gate.yaml")
        _locked_gate_rules = (
            _month_locked_gate_rules
            if _month_locked_gate_rules
            else load_locked_gate_rules(archetype_gate_path)
        )
        if _locked_gate_rules:
            _m_gate = merge_locked_gate_rules(gate_draft_path, _locked_gate_rules)
            if int(_m_gate.get("added", 0) or 0) > 0:
                print(
                    f"   🔒 Gate locked 规则已注入 gate_draft: +{_m_gate['added']} "
                    f"(total={_m_gate['total']})"
                )

    # Gate apply (用 gate_draft 作为中间件)
    gate_draft = f"{config_dir}/gate_draft.yaml"
    run_step(
        "Gate Apply",
        [
            "mlbot",
            "gate",
            "apply-archetype",
            "--logs",
            f"{gate_dir}/predictions.parquet",
            "--strategy",
            strategy,
            "--gate-path",
            gate_draft,
        ],
        log,
        dry_run=dry_run,
    )

    # Gate optimize (--promote, 在 gate 应用后的数据上做 plateau 验证)
    # 注: logs_gated.parquet 含完整 holdout (Val+Test)
    #     通过 --cutoff-date 只用 Val 段 [holdout_start, test_start) 调阈值
    gate_optimize_cmd = [
        "python",
        "scripts/optimize_gate_unified.py",
        "--strategy",
        strategy,
        "--strategies-root",
        strategies_root,
        "--logs",
        f"{gate_dir}/logs_gated.parquet",
        "--output",
        f"{gate_dir}/gate_optimization.json",
        "--gate-path",
        gate_draft,
        "--promote",
    ]
    # 从 kpi_gates 注入 gate 约束
    if gate_gates.get("min_combined_pass_rate"):
        gate_optimize_cmd += [
            "--min-combined-pass-rate",
            str(gate_gates["min_combined_pass_rate"]),
        ]
    if gate_gates.get("max_hard_gates") is not None:
        gate_optimize_cmd += [
            "--max-hard-gates",
            str(int(gate_gates["max_hard_gates"])),
        ]
    if gate_gates.get("max_system_safety") is not None:
        gate_optimize_cmd += [
            "--max-system-safety",
            str(int(gate_gates["max_system_safety"])),
        ]
    if bool(gate_gates.get("require_positive_effect", False)):
        gate_optimize_cmd += ["--require-positive-effect"]
        _tol = gate_gates.get("positive_effect_tol")
        if _tol is not None:
            gate_optimize_cmd += ["--positive-effect-tol", str(float(_tol))]
    # Val/Test 分离: Gate Optimize 只用 Val 段 [holdout_start, test_start)
    if test_start != holdout_start:
        gate_optimize_cmd += ["--cutoff-date", test_start]
    run_step(
        "Gate Optimize",
        gate_optimize_cmd,
        log,
        dry_run=dry_run,
    )

    # ── Gate Subset Selection (可选, 诊断报告: 显示哪些规则组合最优) ──
    _gate_sel_method = gate_gates.get("selection_method", "all")
    if _gate_sel_method != "all":
        gate_sel_cmd = [
            sys.executable,
            "scripts/select_gate_subset.py",
            "--strategy",
            strategy,
            "--strategies-root",
            str(exp_strategies_root),
            "--method",
            _gate_sel_method,
            "--start-date",
            holdout_start,
            "--end-date",
            end_date,
            "--data-path",
            data_path,
            "--min-trades",
            str(kpi_gates.get("backtest", {}).get("min_trades", 30)),
            # 不加 --promote: 仅诊断报告, 不自动写入 gate.yaml (避免单 holdout 过拟合)
            "--output",
            f"{run_dir}/gate_subset_selection.json",
        ]
        run_step(
            "Gate Subset Selection",
            gate_sel_cmd,
            log,
            dry_run=dry_run,
        )

    # Re-apply with optimized gate
    if not dry_run:
        _ensure_timestamp_for_gate_input(gate_pred)
    run_step(
        "Gate Re-Apply",
        [
            "mlbot",
            "gate",
            "apply-archetype",
            "--logs",
            f"{gate_dir}/predictions.parquet",
            "--strategy",
            strategy,
            "--gate-path",
            f"{config_dir}/archetypes/gate.yaml",
        ],
        log,
        dry_run=dry_run,
    )

    if stage_stop == "gate":
        return {
            "stage": "gate",
            "gate_dir": gate_dir,
            "evidence_dir": gate_dir,
            "backtest_metrics": {
                "total_trades": 0,
                "mean_r": 0.0,
                "win_rate": 0.0,
                "sharpe_per_trade": 0.0,
            },
            "exp_config_dir": str(exp_config_dir),
            "prod_config_dir": prod_config_dir,
            "prefilter_comparison": _pf_comparison,
            "validation_end": test_start,
        }

    # ── Val/Test 说明 ──
    # logs_gated.parquet 含完整 holdout [holdout_start, end_date]
    # 当 validation_months > 0 时:
    #   Gate Optimize 已通过 --cutoff-date 只用 Val 段 [holdout_start, test_start)
    #   Backtest 使用 Test 段 [test_start, end_date] (纯 OOS)

    # ── Step 6 (Evidence) + Step 7 (Entry Filter): 统计方法重构 ──
    # Evidence 层已删除. Entry Filter 从 features_entry_filter.yaml 读候选,
    # 多方法统计扫描 + Sharpe 择优.
    evidence_dir = gate_dir  # 后续步骤直接使用 gate_dir

    _ef_yaml_path = Path(config_dir) / "features_entry_filter.yaml"
    _ef_gates = kpi_gates.get("entry_filter", {})
    _ef_arch_dir = Path(config_dir) / "archetypes"
    _ef_orig_path = _ef_arch_dir / "entry_filters.yaml"
    _ef_meta_algorithm = bool(_ef_gates.get("meta_algorithm", True))
    _ef_archetype_plateau = bool(_ef_gates.get("archetype_plateau", False))
    _ef_filter_ids_raw = _ef_gates.get("archetype_filter_ids", [])
    _ef_filter_ids: List[str] = []
    if isinstance(_ef_filter_ids_raw, str):
        _ef_filter_ids = [s.strip() for s in _ef_filter_ids_raw.split(",") if s.strip()]
    elif isinstance(_ef_filter_ids_raw, (list, tuple, set)):
        _ef_filter_ids = [str(s).strip() for s in _ef_filter_ids_raw if str(s).strip()]
    _ef_filter_ids = list(dict.fromkeys(_ef_filter_ids))
    # Entry Filter 多方法候选: distribution_ks + mean_effect + tail_bad_rate_ratio
    # + upside_positive_rate_ratio 各跑一遍, 供候选发现模式下人审看共识矩阵.
    # 2026-04-23: 原 Wave 2-E 单方法锁定已 revert —— 详见上文 Prefilter 同名
    # 守护的 2026-04-23 备注.
    _ef_methods = standardize_method_list(
        _ef_gates.get("scoring_method_fallbacks"),
        default=[
            "distribution_ks",
            "mean_effect",
            "tail_bad_rate_ratio",
            "upside_positive_rate_ratio",
        ],
    )
    if _ef_archetype_plateau and not dry_run:
        _ef_arch_method = str(
            _ef_gates.get("archetype_scoring_method")
            or (_ef_methods[0] if _ef_methods else "distribution_ks")
        ).strip()
        if not _ef_orig_path.exists():
            print(
                "\n  ⚠️  Entry archetype plateau 已启用, 但 archetypes/entry_filters.yaml 不存在, 跳过"
            )
        else:
            _targets = list(_ef_filter_ids)
            if not _targets:
                _locked_filters = load_locked_entry_filters(_ef_orig_path)
                _targets = [
                    str(f.get("id", "")).strip()
                    for f in _locked_filters
                    if str(f.get("id", "")).strip()
                    and not bool(f.get("skip_plateau", False))
                ]
                _targets = list(dict.fromkeys(_targets))
            if not _targets:
                _targets = [None]
            _target_hint = (
                ",".join(_targets)
                if _targets != [None]
                else "all enabled/locked filters"
            )
            print(
                f"\n  ⚙️ Entry Archetype Plateau: method={_ef_arch_method}, targets={_target_hint}"
            )
            _ef_failed_targets: List[str] = []
            for _ef_target in _targets:
                _ef_cmd = [
                    sys.executable,
                    "scripts/optimize_entry_filter_plateau.py",
                    "--logs",
                    f"{gate_dir}/logs_gated.parquet",
                    "--strategy",
                    strategy,
                    "--strategies-root",
                    strategies_root,
                    "--scoring-method",
                    _ef_arch_method,
                    "--promote",
                    "--simple-execution",
                ]
                if _ef_target:
                    _ef_cmd += ["--filter", _ef_target]
                if test_start != holdout_start:
                    _ef_cmd += ["--cutoff-date", test_start]
                _ef_sp = _ef_gates.get("significance_p")
                if _ef_sp is not None:
                    _ef_cmd += ["--significance-p", str(float(_ef_sp))]
                _ef_smt = _ef_gates.get("significance_min_trades")
                if _ef_smt is not None:
                    _ef_cmd += [
                        "--significance-min-trades",
                        str(int(_ef_smt)),
                    ]
                _ef_pw = _ef_gates.get("plateau_window")
                if _ef_pw is not None:
                    _ef_cmd += ["--plateau-window", str(int(_ef_pw))]
                _ef_label = _ef_target or "all_enabled"
                _rc_ap, _ = run_step(
                    f"  EF Archetype Plateau [{_ef_label}]",
                    _ef_cmd,
                    log,
                )
                if _rc_ap != 0:
                    _ef_failed_targets.append(_ef_label)
            if _ef_failed_targets:
                print(
                    "   ⚠️  Entry archetype plateau 部分失败: "
                    + ",".join(_ef_failed_targets)
                )
            else:
                print("   ✅ Entry archetype plateau 完成")

    if _ef_meta_algorithm and _ef_yaml_path.exists() and not dry_run:
        print(f"\n{'='*72}")
        print(
            f"🔬 Entry Filter 多方法 Sharpe 择优 — {len(_ef_methods)} methods: {_ef_methods}"
        )
        print(f"{'='*72}")

        _ef_sharpe: Dict[str, float] = {}
        _ef_trades: Dict[str, int] = {}
        _ef_n_rules: Dict[str, int] = {}
        _ef_score: Dict[str, float] = {}
        _ef_low_gap: Dict[str, float] = {}
        _ef_high_gap: Dict[str, float] = {}
        _ef_low_penalty: Dict[str, float] = {}
        _ef_high_penalty: Dict[str, float] = {}
        _ef_obj = _resolve_trade_objective(
            _ef_gates,
            default_min=30,
            default_max=200,
            default_penalty_low=0.0015,
            default_penalty_high=0.001,
        )
        _simple_exec = scfg.get("simple_execution", {})
        print(
            "   "
            f"entry trade_target=[{int(_ef_obj['target_min'])},{int(_ef_obj['target_max'])}], "
            f"penalty_low={_ef_obj['penalty_low']}, penalty_high={_ef_obj['penalty_high']}"
        )

        for _em in _ef_methods:
            print(f"\n── Entry Filter method: [{_em}] ──")
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
                str(_ef_yaml_path),
                "--scoring-method",
                _em,
                "--promote",
                "--simple-execution",
            ]
            # Val/Test 分离: entry filter 也只用 Val 段
            if test_start != holdout_start:
                ef_cmd += ["--cutoff-date", test_start]
            _rc_ef, _out_ef = run_step(f"  EF Scan [{_em}]", ef_cmd, log)
            if _rc_ef != 0:
                print(f"   ❌ EF [{_em}] 失败")
                _ef_sharpe[_em] = float("-inf")
                _ef_trades[_em] = 0
                _ef_n_rules[_em] = 0
                _ef_score[_em] = float("-inf")
                _ef_low_gap[_em] = 0.0
                _ef_high_gap[_em] = 0.0
                _ef_low_penalty[_em] = 0.0
                _ef_high_penalty[_em] = 0.0
                continue

            # 读取产出的 entry_filters.yaml 规则数
            import yaml as _ef_yaml_mod

            if _ef_orig_path.exists():
                with open(_ef_orig_path, "r", encoding="utf-8") as _efr:
                    _ef_cfg = _ef_yaml_mod.safe_load(_efr) or {}
                _n_ef_rules = len(
                    [f for f in _ef_cfg.get("filters", []) if f.get("enabled", True)]
                )
            else:
                _n_ef_rules = 0
            _ef_n_rules[_em] = _n_ef_rules

            if _n_ef_rules == 0:
                # 没有规则 = 无条件入场, 后续用 empty 的 Sharpe
                _ef_sharpe[_em] = float("-inf")
                _ef_trades[_em] = 0
                _ef_score[_em] = float("-inf")
                _ef_low_gap[_em] = 0.0
                _ef_high_gap[_em] = 0.0
                _ef_low_penalty[_em] = 0.0
                _ef_high_penalty[_em] = 0.0
                continue

            # 保存临时规则文件
            _ef_tmp = _ef_arch_dir / f"entry_filters_cmp_{_em}.yaml"
            shutil.copy(_ef_orig_path, _ef_tmp)
            # 同时落盘到 _candidates/method=<name>/ 给 T4 共识矩阵读取
            # (scripts/slow_candidate_report.py consensus 消费此目录).
            _cand_dir = Path(config_dir) / "_candidates" / f"method={_em}"
            _cand_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(_ef_orig_path, _cand_dir / "entry_filters.yaml")

            # mini-backtest on Val 段
            _ef_bt = [
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
            if _simple_exec.get("sl_r") is not None:
                _ef_bt += ["--simple-sl", str(_simple_exec["sl_r"])]
            if _simple_exec.get("tp_r") is not None:
                _ef_bt += ["--simple-tp", str(_simple_exec["tp_r"])]
            if _simple_exec.get("timeout_bars") is not None:
                _ef_bt += ["--simple-timeout", str(_simple_exec["timeout_bars"])]
            _rc_ebt, _out_ebt = run_step(f"  EF Backtest [{_em}]", _ef_bt, log)
            _ebt_m = parse_backtest_stdout(_out_ebt)
            _ef_sharpe[_em] = _ebt_m.get("sharpe_per_trade", float("-inf"))
            _ef_trades[_em] = _ebt_m.get("total_trades", 0)
            _sc = _score_with_trade_objective(
                sharpe=_ef_sharpe[_em], trades=_ef_trades[_em], objective=_ef_obj
            )
            _ef_score[_em] = _sc["score"]
            _ef_low_gap[_em] = _sc["low_gap"]
            _ef_high_gap[_em] = _sc["high_gap"]
            _ef_low_penalty[_em] = _sc["low_penalty"]
            _ef_high_penalty[_em] = _sc["high_penalty"]
            print(
                f"   📊 EF [{_em}] Sharpe={_ef_sharpe[_em]:+.4f}, "
                f"Trades={_ef_trades[_em]}, Rules={_n_ef_rules}, Score={_ef_score[_em]:+.4f}"
            )

        # ── 对比表: 添加 empty baseline ──
        # empty = 无条件入场 (无 entry filter)
        _ef_empty = _ef_arch_dir / "entry_filters_cmp_empty.yaml"
        _ef_empty.write_text("filters: []\ncombination_mode: or\n", encoding="utf-8")
        shutil.copy(_ef_empty, _ef_orig_path)
        _ef_bt_empty = [
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
        if _simple_exec.get("sl_r") is not None:
            _ef_bt_empty += ["--simple-sl", str(_simple_exec["sl_r"])]
        if _simple_exec.get("tp_r") is not None:
            _ef_bt_empty += ["--simple-tp", str(_simple_exec["tp_r"])]
        if _simple_exec.get("timeout_bars") is not None:
            _ef_bt_empty += ["--simple-timeout", str(_simple_exec["timeout_bars"])]
        _rc_ebt_empty, _out_ebt_empty = run_step(
            "  EF Backtest [empty]", _ef_bt_empty, log
        )
        _ebt_m_empty = parse_backtest_stdout(_out_ebt_empty)
        _ef_sharpe["empty"] = _ebt_m_empty.get("sharpe_per_trade", float("-inf"))
        _ef_trades["empty"] = _ebt_m_empty.get("total_trades", 0)
        _ef_n_rules["empty"] = 0
        _sc_empty = _score_with_trade_objective(
            sharpe=_ef_sharpe["empty"], trades=_ef_trades["empty"], objective=_ef_obj
        )
        _ef_score["empty"] = _sc_empty["score"]
        _ef_low_gap["empty"] = _sc_empty["low_gap"]
        _ef_high_gap["empty"] = _sc_empty["high_gap"]
        _ef_low_penalty["empty"] = _sc_empty["low_penalty"]
        _ef_high_penalty["empty"] = _sc_empty["high_penalty"]

        # 0 规则的方法 = empty
        for _em in _ef_methods:
            if _ef_n_rules.get(_em, 0) == 0 and _em not in ["empty"]:
                _ef_sharpe[_em] = _ef_sharpe["empty"]
                _ef_trades[_em] = _ef_trades["empty"]
                _ef_score[_em] = _ef_score["empty"]
                _ef_low_gap[_em] = _ef_low_gap["empty"]
                _ef_high_gap[_em] = _ef_high_gap["empty"]
                _ef_low_penalty[_em] = _ef_low_penalty["empty"]
                _ef_high_penalty[_em] = _ef_high_penalty["empty"]

        # 汇总对比表
        _best_ef = max(_ef_score, key=lambda m: _ef_score[m])
        _ef_tbl = []
        _ef_tbl.append(f"\n{'='*72}")
        _ef_tbl.append(
            f"  {'方法':<25} {'Score':>10} {'Sharpe':>10} {'Trades':>7} {'Rules':>6}  标记"
        )
        _ef_tbl.append(f"  {'-'*68}")
        for _m in sorted(_ef_score, key=lambda m: -_ef_score[m]):
            _flag = " ← 最优" if _m == _best_ef else ""
            _score = _ef_score[_m]
            _score_str = f"{_score:+.4f}" if _score != float("-inf") else "  FAIL"
            _s = _ef_sharpe[_m]
            _s_str = f"{_s:+.4f}" if _s != float("-inf") else "  FAIL"
            _ef_tbl.append(
                f"  {_m:<25} {_score_str:>10} {_s_str:>10} "
                f"{_ef_trades.get(_m, 0):>7} {_ef_n_rules.get(_m, 0):>6}{_flag}"
            )
        _ef_tbl.append(f"{'='*72}\n")
        _ef_tbl_text = "\n".join(_ef_tbl)
        print(_ef_tbl_text)
        with open(log, "a", encoding="utf-8") as _lf:
            _lf.write(f"\n{'='*72}\n")
            _lf.write(f"🔬 Entry Filter Score 对比汇总\n")
            _lf.write(_ef_tbl_text + "\n")

        # 设置最优 entry filter
        # 只有当新规则对比 empty 有足够提升时才覆盖已有 entry_filters.yaml
        # 防止 Val 段负志时「最不差」方法覆盖原有的合理规则
        _MIN_EF_IMPROVEMENT = float(
            _ef_gates.get("min_score_improvement", 0.005)
        )  # 最小提升幅度 (Score 差值)
        _ef_improvement = _ef_score.get(_best_ef, float("-inf")) - _ef_score.get(
            "empty", float("-inf")
        )
        if (
            _best_ef != "empty"
            and _ef_n_rules.get(_best_ef, 0) > 0
            and _ef_improvement >= _MIN_EF_IMPROVEMENT
        ):
            _best_ef_path = _ef_arch_dir / f"entry_filters_cmp_{_best_ef}.yaml"
            if _best_ef_path.exists():
                shutil.copy(_best_ef_path, _ef_orig_path)
                print(
                    f"   ✅ 最优 Entry Filter [{_best_ef}] 已写入, "
                    f"Rules={_ef_n_rules[_best_ef]}, Score={_ef_score[_best_ef]:+.4f}, "
                    f"Sharpe={_ef_sharpe[_best_ef]:+.4f} "
                    f"(vs empty score {_ef_score.get('empty', 0):+.4f}, 提升={_ef_improvement:+.4f})"
                )
            else:
                # fallback: 写空
                _ef_empty.rename(_ef_orig_path) if not _ef_orig_path.exists() else None
                print(f"   ⚠️  最优方法临时文件丢失, 使用空 entry filter")
        elif _best_ef == "empty" or _ef_improvement < _MIN_EF_IMPROVEMENT:
            # 提升不足 → 保留已有 entry_filters.yaml 不变
            _old_exists = _ef_orig_path.exists()
            if _old_exists:
                print(
                    f"   ℹ️  Entry Filter 提升不足 (best={_best_ef} 提升={_ef_improvement:+.4f} < {_MIN_EF_IMPROVEMENT}), "
                    f"保留原有 entry_filters.yaml 不变"
                )
            else:
                # 无历史文件 → 写空
                shutil.copy(_ef_empty, _ef_orig_path)
                print(
                    f"   ℹ️  empty 最优 且无历史 entry_filters.yaml, 写入空规则 "
                    f"(Score={_ef_score.get('empty', 0):+.4f}, Sharpe={_ef_sharpe.get('empty', 0):+.4f})"
                )

        # 清理临时文件
        for _em in list(_ef_methods) + ["empty"]:
            _tmp = _ef_arch_dir / f"entry_filters_cmp_{_em}.yaml"
            if _tmp.exists() and _tmp != _ef_orig_path:
                _tmp.unlink()
    elif _ef_meta_algorithm and not _ef_yaml_path.exists():
        print(f"\n  ℹ️  Entry Filter: features_entry_filter.yaml 不存在, 跳过")
    elif not _ef_meta_algorithm:
        print("\n  ⏭️  Entry Filter meta-algorithm 已关闭: 跳过特征搜索路径")

    if stage_stop == "entry_filter":
        return {
            "stage": "entry_filter",
            "gate_dir": gate_dir,
            "evidence_dir": evidence_dir,
            "backtest_metrics": {
                "total_trades": 0,
                "mean_r": 0.0,
                "win_rate": 0.0,
                "sharpe_per_trade": 0.0,
            },
            "exp_config_dir": str(exp_config_dir),
            "prod_config_dir": prod_config_dir,
            "prefilter_comparison": _pf_comparison,
            "validation_end": test_start,
        }

    # ── Step 8: Execution Optimize (跳过) ──
    # 默认使用 execution.yaml 中的 2ATR 止损, 快速出结果.
    # Execution 参数精调后续用事件回测 + --use-1min 手动优化.
    print("\n  ⏭️  Step 8 SKIP: Execution Optimize 跳过 (默认 2ATR, 后续事件回测精调)")

    # ── Step 9: 向量回测 (快速, 简单执行模式) ──
    # 使用 --simple-execution: 固定 SL/TP/timeout, 可在 research_pipeline.yaml simple_execution 块按策略定制
    # 默认: SL=1.5R, TP=3R, 50bar timeout
    # 目的: 中性评估 Gate/Evidence/Entry Filter 信号质量
    # Execution 参数精调 (trailing/structural) 放到事件回测阶段
    _simple_exec_cfg = scfg.get("simple_execution", {})
    bt_cmd = [
        "python",
        "scripts/backtest_execution_layer.py",
        "--logs",
        f"{evidence_dir}/logs_gated.parquet",
        "--strategy",
        strategy,
        "--strategies-root",
        strategies_root,
        "--test-start",
        test_start,  # Val/Test 分离: Backtest 只用 Test 段
        "--test-end",
        end_date,
        "--simple-execution",
    ]
    if _simple_exec_cfg.get("sl_r") is not None:
        bt_cmd += ["--simple-sl", str(_simple_exec_cfg["sl_r"])]
    if _simple_exec_cfg.get("tp_r") is not None:
        bt_cmd += ["--simple-tp", str(_simple_exec_cfg["tp_r"])]
    if _simple_exec_cfg.get("timeout_bars") is not None:
        bt_cmd += ["--simple-timeout", str(_simple_exec_cfg["timeout_bars"])]
    rc, bt_out = run_step(
        "Vector Backtest",
        bt_cmd,
        log,
        dry_run=dry_run,
    )

    # ── Step 9b: 交易地图 (Full Execution, trailing stop 真实行为) ──
    # 使用 execution.yaml 真实配置生成交易地图 (trailing/structural 可见)
    # 注意: 此步骤仅生成地图, 不影响 ADOPT 决策指标 (仍用 Step 9 simple-execution 结果)
    map_cmd = [
        "python",
        "scripts/backtest_execution_layer.py",
        "--logs",
        f"{evidence_dir}/logs_gated.parquet",
        "--strategy",
        strategy,
        "--strategies-root",
        strategies_root,
        "--test-start",
        test_start,  # Val/Test 分离: 交易地图也只用 Test 段
        "--test-end",
        end_date,
        "--output",
        f"{evidence_dir}/trading_map_{strategy}_exec.html",
    ]
    run_step(
        "Trading Map (Full Exec)",
        map_cmd,
        log,
        dry_run=dry_run,
    )

    # ── 收集指标 ──
    backtest_metrics = (
        parse_backtest_stdout(bt_out)
        if not dry_run
        else {
            "total_trades": 0,
            "mean_r": 0,
            "win_rate": 0,
            "sharpe_per_trade": 0,
        }
    )

    # ── Step 10: 导出训练基线 JSON ──
    if not dry_run:
        try:
            import importlib
            import sys as _sys

            # 确保项目根目录在 sys.path 中
            root_str = str(PROJECT_ROOT)
            if root_str not in _sys.path:
                _sys.path.insert(0, root_str)
            mod = importlib.import_module("scripts.export_training_baseline")
            mod.export_training_baseline(
                strategy=strategy,
                result_dir=Path(evidence_dir),
                gate_dir=Path(gate_dir),
                evidence_dir=Path(evidence_dir),
                backtest_metrics=backtest_metrics,
                config_root=strategies_root,
                training_period={"start": start_date, "end": holdout_start},
                holdout_period={"start": holdout_start, "end": end_date},
            )
        except Exception as exc:
            print(f"\n⚠️  Baseline export failed: {exc}")
    else:
        print("\n  Step 10: Export Training Baseline (dry-run, 跳过)")

    return {
        "gate_dir": gate_dir,
        "evidence_dir": evidence_dir,
        "backtest_metrics": backtest_metrics,
        "exp_config_dir": str(exp_config_dir),
        "prod_config_dir": prod_config_dir,
        "prefilter_comparison": _pf_comparison,
        "validation_end": test_start,
    }


# ====================================================================
# Save report
# ====================================================================


def save_report(
    strategy: str,
    cfg: dict,
    run_dir: Path,
    pipeline_result: Dict[str, Any],
    comparison: Dict[str, Any],
    *,
    start_date: str,
    end_date: str,
    holdout_start: str,
    holdout_months: int,
    validation_months: int,
    validation_end: str = "",  # test_start (原 holdout_start)
) -> Path:
    """保存结构化 report.json + archetypes 快照."""
    scfg = cfg["strategies"][strategy]
    timestamp = run_dir.name

    # 从实验目录读取 archetypes (已 promote 的版本)
    exp_config_dir = pipeline_result.get("exp_config_dir")
    if exp_config_dir:
        thresholds = {}
        arch_dir = Path(exp_config_dir) / "archetypes"
        for name in [
            "gate.yaml",
            "evidence.yaml",
            "entry_filters.yaml",
            "execution.yaml",
        ]:
            f = arch_dir / name
            if f.exists():
                thresholds[name] = yaml.safe_load(f.read_text(encoding="utf-8"))
    else:
        thresholds = load_archetype_thresholds(strategy, scfg)

    report = {
        "version": 2,
        "strategy": strategy,
        "timestamp": timestamp,
        "data_range": {
            "start_date": start_date,
            "end_date": end_date,
            "holdout_start": holdout_start,
            "holdout_months": holdout_months,
            "validation_end": validation_end or holdout_start,
            "validation_months": validation_months,
        },
        "backtest_metrics": pipeline_result.get("backtest_metrics", {}),
        "thresholds": thresholds,
        "comparison": comparison,
        "artifacts": {
            "gate_dir": pipeline_result.get("gate_dir"),
            "evidence_dir": pipeline_result.get("evidence_dir"),
            "exp_config_dir": exp_config_dir,
        },
    }

    # Save report.json
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )

    # 实验 archetypes 已在 run_dir/strategies/{strategy}/archetypes/ 中
    # 不再复制快照副本到 run_dir/archetypes/ (冗余)

    # Save comparison
    comp_path = run_dir / "comparison.json"
    comp_path.write_text(
        json.dumps(comparison, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # 整合: 将 train_final_* 产物目录移动到 run_dir/results/ 下统一管理
    _gate_dir_str = pipeline_result.get("gate_dir")
    if _gate_dir_str:
        _src = PROJECT_ROOT / _gate_dir_str
        _dst = run_dir / "results"
        if _src.exists() and not _dst.exists():
            import shutil as _shutil

            try:
                _shutil.move(str(_src), str(_dst))
                # 更新 report.json 中的 artifact 路径
                _new_rel = str(_dst.relative_to(PROJECT_ROOT))
                report["artifacts"]["gate_dir"] = _new_rel
                report["artifacts"]["evidence_dir"] = _new_rel
                report_path.write_text(
                    json.dumps(report, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8",
                )
                # 同步内存中的路径，否则后续 event_backtest 仍把 trading_map 写到已搬走的 train_final_*
                pipeline_result["gate_dir"] = _new_rel
                pipeline_result["evidence_dir"] = _new_rel
                # 清理空的 train_final_* 父目录
                _parent = _src.parent
                if _parent.exists() and not any(_parent.iterdir()):
                    _parent.rmdir()
                print(f"   📂 产物整合: {_gate_dir_str} → {_new_rel}")
            except Exception as _e:
                print(f"   ⚠️  产物整合失败 (non-critical): {_e}")

    return report_path


def _patch_report_deploy(report_path: Path, deploy_result: Dict[str, Any]):
    """将 deploy 门禁结果追加到已保存的 report.json."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["deploy_gate"] = {
            "deploy_ready": deploy_result["deploy_ready"],
            "triggered": deploy_result.get("triggered", False),
            "triggers": deploy_result.get("triggers", []),
            "safety": deploy_result.get("safety", []),
            "blocked_by": deploy_result.get("blocked_by", []),
            "skip_reason": deploy_result.get("skip_reason"),
        }
        report_path.write_text(
            json.dumps(report, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass  # 非关键路径, 不影响主流程


def _patch_report_event(report_path: Path, event_result: Dict[str, Any]):
    """将事件回测结果追加到已保存的 report.json."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["event_backtest"] = event_result
        report_path.write_text(
            json.dumps(report, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _parse_event_stdout(output: str) -> Dict[str, Any]:
    """从事件回测 stdout 提取指标."""
    m = re.search(r"交易数:\s*(\d+)", output)
    n_trades = int(m.group(1)) if m else None
    m = re.search(r"胜率:\s*([\d.]+)%", output)
    win_rate = float(m.group(1)) / 100 if m else None
    m = re.search(r"Sharpe \(R\):\s*([\-\d.]+)", output)
    sharpe = float(m.group(1)) if m else None
    m = re.search(r"Mean R:\s*([\-\d.]+)", output)
    mean_r = float(m.group(1)) if m else None
    m = re.search(r"Total R:\s*([\-\d.]+)", output)
    total_r = float(m.group(1)) if m else None
    m = re.search(r"Max DD \(R\):\s*([\-\d.]+)", output)
    max_dd = float(m.group(1)) if m else None
    return {
        k: v
        for k, v in {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "sharpe_r": sharpe,
            "mean_r": mean_r,
            "total_r": total_r,
            "max_drawdown_r": max_dd,
        }.items()
        if v is not None
    }


def _event_trading_map_extra_months(cfg: Dict[str, Any]) -> int:
    """交易地图：向前多取月数（仅用于 VWAP/EMA 计算，见 event_backtest.map_extra_months）。"""
    ev = cfg.get("event_backtest", {}) or {}
    try:
        return max(0, int(ev.get("map_extra_months", 12)))
    except (TypeError, ValueError):
        return 12


def _capital_report_initial_capital(cfg: Dict[str, Any]) -> float:
    cap_cfg = cfg.get("capital_report", {}) or {}
    try:
        return float(cap_cfg.get("initial_capital", 10000.0))
    except (TypeError, ValueError):
        return 10000.0


def _capital_report_risk_per_r(cfg: Dict[str, Any]) -> float:
    cap_cfg = cfg.get("capital_report", {}) or {}
    try:
        return float(cap_cfg.get("risk_per_r", 0.01))
    except (TypeError, ValueError):
        return 0.01


def _write_event_capital_report(
    *,
    cfg: Dict[str, Any],
    trades_path: str | Path,
    out_dir: str | Path,
    title: str,
    start_date: str,
    end_date: str,
    total_r: Optional[float] = None,
) -> Dict[str, Any]:
    return write_capital_report_from_trades(
        trades_path=trades_path,
        out_dir=out_dir,
        unit="r_multiple",
        title=title,
        initial_capital=_capital_report_initial_capital(cfg),
        risk_per_r=_capital_report_risk_per_r(cfg),
        start_date=start_date,
        end_date=end_date,
        total_r=total_r,
    )


def _run_event_backtest_step(
    strategy: str,
    evidence_dir: str,
    run_dir: Path,
    *,
    holdout_start: str,
    end_date: str,
    strategies_root: str,
    data_path: str,
    dry_run: bool = False,
    sym_r: str = "1.0:0.5:4.0",
    exec_grid: Optional[Dict[str, str]] = None,
    promote: bool = True,
    objective: str = "sharpe",
    near_stop_threshold_r: float = -0.9,
    near_stop_penalty: float = 0.0,
    max_dd_penalty: float = 0.0,
    min_trades_soft: int = 0,
    undertrade_penalty: float = 0.0,
    resume_state_path: str = "",
    dump_end_state_path: str = "",
    keep_open_positions: bool = False,
    run_execution_opt: bool = True,
    opt_start_date: str = "",
    opt_end_date: str = "",
    event_start_date: str = "",
    event_end_date: str = "",
    map_extra_months: int = 12,
    no_kill_switch: bool = False,
) -> Dict[str, Any]:
    """Step E1: 事件回测 execution 参数优化 + 交易地图生成."""
    log = run_dir / "pipeline.log"
    results_dir = Path(evidence_dir)

    print(f"\n{'='*70}")
    print(f"🎯 Event Backtest Execution Opt: {strategy}")
    print(f"{'='*70}")

    _opt_start = str(opt_start_date or holdout_start)
    _opt_end = str(opt_end_date or end_date)
    _ev_start = str(event_start_date or holdout_start)
    _ev_end = str(event_end_date or end_date)
    rc_opt = 0

    # Step E1a: Execution 参数优化 (sym-r grid search or independent grid)
    opt_output = str(run_dir / "event_exec_opt.json")
    if run_execution_opt:
        opt_cmd = [
            "python",
            "scripts/optimize_event_execution.py",
            "--strategy",
            strategy,
            "--start-date",
            _opt_start,
            "--end-date",
            _opt_end,
        ]
        if exec_grid:
            if "initial_r" in exec_grid:
                opt_cmd.extend(["--initial-r", exec_grid["initial_r"]])
            if "activation_r" in exec_grid:
                opt_cmd.extend(["--activation-r", exec_grid["activation_r"]])
            if "trail_r" in exec_grid:
                opt_cmd.extend(["--trail-r", exec_grid["trail_r"]])
            opt_cmd.append("--trailing")
        else:
            opt_cmd.extend(["--sym-r", sym_r])
        opt_cmd.extend(
            [
                "--strategies-root",
                strategies_root,
                "--data-path",
                data_path,
                "--output",
                opt_output,
                "--objective",
                objective,
                "--near-stop-threshold-r",
                str(near_stop_threshold_r),
                "--near-stop-penalty",
                str(near_stop_penalty),
                "--max-dd-penalty",
                str(max_dd_penalty),
                "--min-trades-soft",
                str(min_trades_soft),
                "--undertrade-penalty",
                str(undertrade_penalty),
            ]
        )
        if promote:
            opt_cmd.append("--promote")
        rc_opt, _ = run_step("Event Execution Optimize", opt_cmd, log, dry_run=dry_run)
        if rc_opt != 0:
            print("   ⚠️  Execution 优化有异常, 继续使用当前 execution.yaml")
    else:
        print("   ⏭️  跳过 execution 优化, 直接执行事件回测")

    # Step E1b: 事件回测 + 交易地图
    map_path = str(results_dir / f"trading_map_{strategy}_event.html")
    export_path = str(results_dir / f"event_trades_{strategy}.csv")
    event_json_path = str(results_dir / f"event_backtest_{strategy}.json")
    ev_cmd = [
        "python",
        "scripts/event_backtest.py",
        "--strategy",
        strategy,
        "--start-date",
        _ev_start,
        "--end-date",
        _ev_end,
        "--strategies-root",
        strategies_root,
        "--data-path",
        data_path,
        "--trading-map",
        map_path,
        "--export",
        export_path,
        "--output",
        event_json_path,
        "--fast",
        "--map-extra-months",
        str(int(map_extra_months)),
    ]
    if resume_state_path:
        ev_cmd.extend(["--resume-state", resume_state_path])
    if dump_end_state_path:
        ev_cmd.extend(["--dump-end-state", dump_end_state_path])
    if keep_open_positions:
        ev_cmd.append("--keep-open-positions")
    if no_kill_switch:
        ev_cmd.append("--no-kill-switch")
    rc_ev, ev_out = run_step("Event Backtest", ev_cmd, log, dry_run=dry_run)

    event_metrics = _parse_event_stdout(ev_out) if not dry_run else {}
    if not dry_run:
        event_metrics.update(_load_pcm_metrics_from_json(event_json_path))
    event_metrics["sym_r"] = sym_r
    event_metrics["objective"] = objective
    event_metrics["opt_window_start"] = _opt_start
    event_metrics["opt_window_end"] = _opt_end
    event_metrics["event_window_start"] = _ev_start
    event_metrics["event_window_end"] = _ev_end
    event_metrics["trading_map"] = map_path
    event_metrics["json_path"] = event_json_path
    event_metrics["exec_opt_rc"] = rc_opt
    if not dry_run:
        cap = write_capital_report_from_trades(
            trades_path=export_path,
            out_dir=results_dir,
            unit="r_multiple",
            title=f"{strategy} Capital Report",
            initial_capital=10000.0,
            risk_per_r=0.01,
            start_date=_ev_start,
            end_date=_ev_end,
            total_r=event_metrics.get("total_r"),
        )
        event_metrics["capital_report"] = str(results_dir / "capital_report.json")
        event_metrics["capital_report_metrics"] = cap
    return {
        "rc": rc_ev,
        "metrics": event_metrics,
        "map_path": map_path,
        "json_path": event_json_path,
        "end_state_path": dump_end_state_path,
    }


def _run_event_execution_opt_only(
    strategy: str,
    run_dir: Path,
    *,
    holdout_start: str,
    end_date: str,
    strategies_root: str,
    data_path: str,
    dry_run: bool = False,
    sym_r: str = "1.0:0.5:4.0",
    exec_grid: Optional[Dict[str, str]] = None,
    promote: bool = True,
    objective: str = "sharpe",
    near_stop_threshold_r: float = -0.9,
    near_stop_penalty: float = 0.0,
    max_dd_penalty: float = 0.0,
    min_trades_soft: int = 0,
    undertrade_penalty: float = 0.0,
) -> Dict[str, Any]:
    """仅执行 execution 参数网格优化，不跑事件回测."""
    log = run_dir / "pipeline.log"
    opt_output = str(run_dir / "event_exec_opt.json")
    opt_cmd = [
        "python",
        "scripts/optimize_event_execution.py",
        "--strategy",
        strategy,
        "--start-date",
        holdout_start,
        "--end-date",
        end_date,
    ]
    if exec_grid:
        if "initial_r" in exec_grid:
            opt_cmd.extend(["--initial-r", exec_grid["initial_r"]])
        if "activation_r" in exec_grid:
            opt_cmd.extend(["--activation-r", exec_grid["activation_r"]])
        if "trail_r" in exec_grid:
            opt_cmd.extend(["--trail-r", exec_grid["trail_r"]])
        opt_cmd.append("--trailing")
    else:
        opt_cmd.extend(["--sym-r", sym_r])
    opt_cmd.extend(
        [
            "--strategies-root",
            strategies_root,
            "--data-path",
            data_path,
            "--output",
            opt_output,
            "--objective",
            objective,
            "--near-stop-threshold-r",
            str(near_stop_threshold_r),
            "--near-stop-penalty",
            str(near_stop_penalty),
            "--max-dd-penalty",
            str(max_dd_penalty),
            "--min-trades-soft",
            str(min_trades_soft),
            "--undertrade-penalty",
            str(undertrade_penalty),
        ]
    )
    if promote:
        opt_cmd.append("--promote")
    rc_opt, _ = run_step(
        f"Event Execution Optimize ({strategy})",
        opt_cmd,
        log,
        dry_run=dry_run,
    )
    return {
        "rc": rc_opt,
        "objective": objective,
        "sym_r": sym_r,
        "output": opt_output,
    }


def _resolve_event_exec_grid_for_strategy(
    cfg: Dict[str, Any],
    strategy: str,
) -> Optional[Dict[str, str]]:
    """Resolve independent execution grid (initial_r/activation_r/trail_r).

    Returns dict like {"initial_r": "2:1:6", "activation_r": "2:1:6", "trail_r": "1:0.5:3"}
    or None if not configured (caller should fall back to sym_r).
    """
    ev_cfg = cfg.get("event_backtest", {}) or {}
    by_family = ev_cfg.get("exec_grid_by_family", {}) or {}
    family = str(strategy).split("-")[0].lower().strip()
    grid = by_family.get(strategy) or by_family.get(family)
    if isinstance(grid, dict) and any(
        k in grid for k in ("initial_r", "activation_r", "trail_r")
    ):
        return {k: str(v) for k, v in grid.items()}
    return None


def _resolve_event_sym_r_for_strategy(
    cfg: Dict[str, Any], strategy: str, cli_default_sym_r: str
) -> str:
    """Resolve per-strategy event execution sym_r from pipeline config.

    Priority:
      1) event_backtest.sym_r_by_strategy.<strategy>
      2) event_backtest.sym_r_by_family.<family>
      3) event_backtest.sym_r_default
      4) CLI --event-sym-r
    """
    ev_cfg = cfg.get("event_backtest", {}) or {}
    by_strategy = ev_cfg.get("sym_r_by_strategy", {}) or {}
    if isinstance(by_strategy, dict):
        val = by_strategy.get(strategy)
        if val:
            return str(val)

    family = str(strategy).split("-")[0].lower().strip()
    by_family = ev_cfg.get("sym_r_by_family", {}) or {}
    if isinstance(by_family, dict):
        val = by_family.get(family)
        if val:
            return str(val)

    val = ev_cfg.get("sym_r_default")
    if val:
        return str(val)
    return str(cli_default_sym_r)


def _resolve_event_exec_objective_for_strategy(
    cfg: Dict[str, Any], strategy: str
) -> Dict[str, Any]:
    """Resolve per-strategy execution optimization objective from config.

    Priority:
      1) event_backtest.exec_objective_by_strategy.<strategy>
      2) event_backtest.exec_objective_by_family.<family>
      3) event_backtest.exec_objective_default + exec_objective_params
      4) built-in defaults
    """
    ev_cfg = cfg.get("event_backtest", {}) or {}
    payload: Dict[str, Any] = {
        "objective": str(ev_cfg.get("exec_objective_default", "sharpe")),
        "near_stop_threshold_r": -0.9,
        "near_stop_penalty": 0.0,
        "max_dd_penalty": 0.0,
        "min_trades_soft": 0,
        "undertrade_penalty": 0.0,
    }
    base_params = ev_cfg.get("exec_objective_params", {}) or {}
    if isinstance(base_params, dict):
        payload.update({k: base_params[k] for k in payload.keys() if k in base_params})

    family = str(strategy).split("-")[0].lower().strip()
    by_family = ev_cfg.get("exec_objective_by_family", {}) or {}
    if isinstance(by_family, dict) and family in by_family:
        fam_cfg = by_family.get(family)
        if isinstance(fam_cfg, str):
            payload["objective"] = fam_cfg
        elif isinstance(fam_cfg, dict):
            payload.update({k: fam_cfg[k] for k in payload.keys() if k in fam_cfg})

    by_strategy = ev_cfg.get("exec_objective_by_strategy", {}) or {}
    if isinstance(by_strategy, dict) and strategy in by_strategy:
        st_cfg = by_strategy.get(strategy)
        if isinstance(st_cfg, str):
            payload["objective"] = st_cfg
        elif isinstance(st_cfg, dict):
            payload.update({k: st_cfg[k] for k in payload.keys() if k in st_cfg})

    payload["objective"] = str(payload.get("objective", "sharpe"))
    payload["near_stop_threshold_r"] = float(payload.get("near_stop_threshold_r", -0.9))
    payload["near_stop_penalty"] = float(payload.get("near_stop_penalty", 0.0))
    payload["max_dd_penalty"] = float(payload.get("max_dd_penalty", 0.0))
    payload["min_trades_soft"] = int(payload.get("min_trades_soft", 0))
    payload["undertrade_penalty"] = float(payload.get("undertrade_penalty", 0.0))
    return payload


def _patch_report_pcm(report_path: Path, pcm_result: Dict[str, Any]):
    """将 PCM 联合回测结果追加到已保存的 report.json."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["pcm_joint"] = {
            "pcm_decision": pcm_result.get("pcm_decision"),
            "sharpe_daily": pcm_result.get("sharpe_daily"),
            "conflict_rate": pcm_result.get("conflict_rate"),
            "strategies_count": pcm_result.get("strategies_count"),
            "strategies": pcm_result.get("strategies", []),
            "total_trades": pcm_result.get("total_trades"),
        }
        report_path.write_text(
            json.dumps(report, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _patch_report_pcm_slot_grid(report_path: Path, slot_grid_result: Dict[str, Any]):
    """将 PCM slot grid 对比结果追加到已保存的 report.json."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["pcm_slot_grid"] = slot_grid_result
        report_path.write_text(
            json.dumps(report, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _sanitize_case_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name).strip())
    return cleaned.strip("_") or "case"


def _apply_slot_case_to_constitution(
    base_obj: Dict[str, Any], case_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    obj = copy.deepcopy(base_obj)
    slots = obj.setdefault("slots", {})
    slot_cfg = case_cfg.get("slots", {}) or {}
    if "slot_count" in slot_cfg:
        slots["slot_count"] = int(slot_cfg["slot_count"])
    if "risk_per_slot" in slot_cfg:
        slots["risk_per_slot"] = float(slot_cfg["risk_per_slot"])

    ra = obj.setdefault("resource_allocation", {})
    psl = ra.setdefault("per_strategy_limits", {})
    case_psl = case_cfg.get("per_strategy_limits", {}) or {}
    for strategy_name, updates in case_psl.items():
        if not isinstance(updates, dict):
            continue
        dst = psl.setdefault(str(strategy_name), {})
        for k, v in updates.items():
            dst[k] = v
    return obj


def _score_pcm_slot_case(
    case_result: Dict[str, Any], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    penalties = cfg.get("penalties", {}) or {}
    max_dd_penalty = float(penalties.get("max_dd_penalty", 0.0))
    slot_full_rate_penalty = float(penalties.get("slot_full_rate_penalty", 0.0))
    undertrade_penalty = float(penalties.get("undertrade_penalty", 0.0))
    min_trades_soft = int(cfg.get("min_trades_soft", 0) or 0)

    sharpe = float(case_result.get("sharpe_daily", 0.0) or 0.0)
    max_dd = float(case_result.get("max_drawdown_r", 0.0) or 0.0)
    slot_full_rate = float(case_result.get("slot_full_rate", 0.0) or 0.0)
    total_trades = int(case_result.get("total_trades", 0) or 0)
    undertrade_gap = max(0, min_trades_soft - total_trades)

    score = (
        sharpe
        - max_dd_penalty * max_dd
        - slot_full_rate_penalty * slot_full_rate
        - undertrade_penalty * undertrade_gap
    )
    return {
        "score": float(score),
        "score_components": {
            "sharpe": sharpe,
            "max_dd_penalty": max_dd_penalty * max_dd,
            "slot_full_rate_penalty": slot_full_rate_penalty * slot_full_rate,
            "undertrade_penalty": undertrade_penalty * undertrade_gap,
            "undertrade_gap": undertrade_gap,
        },
    }


def _run_pcm_slot_grid_backtest(
    *,
    cfg_slot: Dict[str, Any],
    results_summary: List[Dict[str, Any]],
    history_dir: Path,
    timestamp: str,
    dry_run: bool,
    use_1min: bool,
    live_root: str,
    data_path: str,
    holdout_start: str,
    end_date: str,
) -> Optional[Dict[str, Any]]:
    """在 pipeline 中运行 slot 网格 PCM 回测，并选择平坦稳定参数."""
    if not bool(cfg_slot.get("enabled", False)):
        return None

    raw_cases = cfg_slot.get("cases", []) or []
    if not isinstance(raw_cases, list) or not raw_cases:
        print("\n[Step 9.6] PCM Slot Grid: ⏭️  SKIP (未配置 cases)")
        return None

    constitution_path = PROJECT_ROOT / "config/constitution/constitution.yaml"
    if not constitution_path.exists():
        print("\n[Step 9.6] PCM Slot Grid: ⏭️  SKIP (constitution.yaml 不存在)")
        return None

    first_strat = next(
        (r["strategy"] for r in results_summary if r.get("evidence_dir")), None
    )
    if not first_strat:
        return None
    out_dir = history_dir / "_pcm_slot_grid" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("🧪 PCM Slot Grid Backtest (Step 9.6)")
    print(f"{'='*70}")

    base_text = constitution_path.read_text(encoding="utf-8")
    base_obj = yaml.safe_load(base_text) or {}

    case_results: List[Dict[str, Any]] = []
    try:
        for i, case in enumerate(raw_cases, 1):
            if not isinstance(case, dict):
                continue
            case_name = str(case.get("name", f"case_{i:02d}"))
            case_safe = _sanitize_case_name(case_name)
            print(f"\n   ▶️  Slot Case {i}: {case_name}")

            patched = _apply_slot_case_to_constitution(base_obj, case)
            constitution_path.write_text(
                yaml.safe_dump(patched, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            case_pcm = pipeline_events.run_pcm_joint_backtest(
                results_summary,
                history_dir,
                timestamp,
                dry_run=dry_run,
                use_1min=use_1min,
                live_root=live_root,
                data_path=data_path,
                holdout_start=holdout_start,
                end_date=end_date,
                output_stem=f"pcm_slot_grid_{case_safe}",
                step_name=f"PCM Slot Case: {case_name}",
            )
            if not case_pcm:
                case_results.append(
                    {
                        "name": case_name,
                        "case": case,
                        "error": "pcm backtest skipped/failed",
                    }
                )
                continue

            scored = _score_pcm_slot_case(case_pcm, cfg_slot)
            row = {
                "name": case_name,
                "case": case,
                **case_pcm,
                **scored,
            }
            case_results.append(row)
            print(
                "      "
                f"score={row.get('score', 0):+.4f} "
                f"sharpe={row.get('sharpe_daily', 0):+.4f} "
                f"dd={row.get('max_drawdown_r', 0):.2f} "
                f"slot_full={row.get('slot_full_rate', 0):.2%}"
            )
    finally:
        # 无论成功失败都恢复 constitution
        constitution_path.write_text(base_text, encoding="utf-8")

    valid = [r for r in case_results if "error" not in r]
    if not valid:
        return {"enabled": True, "cases": case_results, "recommended_case": None}

    best_score = max(float(r.get("score", -1e9)) for r in valid)
    plateau_delta = float(cfg_slot.get("plateau_delta", 0.02))
    plateau = [
        r for r in valid if float(r.get("score", -1e9)) >= best_score - plateau_delta
    ]
    plateau_sorted = sorted(
        plateau,
        key=lambda r: (
            float(r.get("max_drawdown_r", 1e9) or 1e9),
            float(r.get("slot_full_rate", 1e9) or 1e9),
            -float(r.get("sharpe_daily", -1e9) or -1e9),
            -int(r.get("total_trades", 0) or 0),
        ),
    )
    recommended = (
        plateau_sorted[0]
        if plateau_sorted
        else max(valid, key=lambda r: float(r.get("score", -1e9)))
    )

    report = {
        "enabled": True,
        "plateau_delta": plateau_delta,
        "best_score": best_score,
        "cases": case_results,
        "recommended_case": recommended.get("name") if recommended else None,
        "recommended_metrics": {
            "score": recommended.get("score") if recommended else None,
            "sharpe_daily": recommended.get("sharpe_daily") if recommended else None,
            "total_trades": recommended.get("total_trades") if recommended else None,
            "total_r": recommended.get("total_r") if recommended else None,
            "max_drawdown_r": (
                recommended.get("max_drawdown_r") if recommended else None
            ),
            "slot_full_rate": (
                recommended.get("slot_full_rate") if recommended else None
            ),
        },
    }

    json_path = out_dir / "pcm_slot_grid_report.json"
    md_path = out_dir / "pcm_slot_grid_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    md_lines = [
        "# PCM Slot Grid Report",
        "",
        f"- Recommended case: `{report['recommended_case']}`",
        f"- Plateau delta: `{plateau_delta}`",
        "",
        "| case | score | sharpe | total_r | max_dd_r | trades | slot_full_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(valid, key=lambda x: float(x.get("score", -1e9)), reverse=True):
        md_lines.append(
            "| "
            + f"{r.get('name')} | "
            + f"{float(r.get('score', 0)):+.4f} | "
            + f"{float(r.get('sharpe_daily', 0)):+.4f} | "
            + f"{float(r.get('total_r', 0)):+.4f} | "
            + f"{float(r.get('max_drawdown_r', 0)):.4f} | "
            + f"{int(r.get('total_trades', 0))} | "
            + f"{float(r.get('slot_full_rate', 0)):.2%} |"
        )
    md_path.write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)

    print(f"\n   ✅ Slot Grid 推荐: {report['recommended_case']}")
    print(f"   📄 报告: {json_path}")
    return report


def _run_grid_backtest_stage(
    *,
    cfg: Dict[str, Any],
    strategies: List[str],
    history_dir: Path,
    timestamp: str,
    dry_run: bool,
    data_path: str,
    symbols: str,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """Run standalone grid strategies without the single-position event path."""
    grid_cfg = cfg.get("grid_backtest", {}) or {}
    if not bool(grid_cfg.get("enabled", True)):
        print("\n[Grid Backtest] ⏭️  SKIP (grid_backtest.enabled=false)")
        return []

    out_root = PROJECT_ROOT / str(
        grid_cfg.get("output_dir", f"results/chop_grid/pipeline/{timestamp}")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    print(f"\n{'='*70}")
    print("🧪 Chop Grid Backtest Stage")
    print(f"{'='*70}")
    print(f"   Output: {out_root}")

    for strat in strategies:
        scfg = (cfg.get("strategies", {}) or {}).get(strat, {}) or {}
        strategy_type = str(scfg.get("strategy_type", "") or "").lower()
        if strategy_type != "grid":
            print(
                f"   ⏭️  skip {strat}: strategy_type={strategy_type or 'single_position'}"
            )
            continue
        strat_cfg_dir = PROJECT_ROOT / str(
            scfg.get("config", f"config/strategies/{strat}")
        )
        grid_yaml = strat_cfg_dir / "grid.yaml"
        if not grid_yaml.exists():
            raise FileNotFoundError(f"grid strategy config missing: {grid_yaml}")

        out_dir = out_root / strat
        cmd = [
            sys.executable,
            "scripts/chop_grid_backtest.py",
            "--config",
            str(grid_yaml),
            "--data-dir",
            data_path,
            "--symbols",
            symbols,
            "--start",
            start_date,
            "--end",
            end_date,
            "--timeframe",
            str(scfg.get("timeframe", "2h")),
            "--out-dir",
            str(out_dir),
        ]
        map_symbols = str(grid_cfg.get("map_symbols", "") or "").strip()
        if map_symbols:
            cmd.extend(["--map-symbols", map_symbols])
        if "map_months" in grid_cfg:
            cmd.extend(["--map-months", str(int(grid_cfg.get("map_months", 12) or 12))])
        continuous_map_symbols = str(
            grid_cfg.get("continuous_map_symbols", "") or ""
        ).strip()
        if continuous_map_symbols:
            cmd.extend(["--continuous-map-symbols", continuous_map_symbols])
        if "continuous_map_months" in grid_cfg:
            cmd.extend(
                [
                    "--continuous-map-months",
                    str(int(grid_cfg.get("continuous_map_months", 0) or 0)),
                ]
            )
        print(f"\n   ▶️  {strat}: {' '.join(cmd)}")
        if dry_run:
            summaries.append(
                {"strategy": strat, "out_dir": str(out_dir), "dry_run": True}
            )
            continue
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(f"grid_backtest failed for {strat}\n{tail}")
        if proc.stdout:
            print(proc.stdout[-2000:])
        metrics_path = out_dir / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summaries.append(
            {
                "strategy": strat,
                "out_dir": str(out_dir),
                "report": str(out_dir / "report.html"),
                "metrics": metrics.get("metrics", {}),
            }
        )
    (out_root / "grid_backtest_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summaries


def _run_dual_add_backtest_stage(
    *,
    cfg: Dict[str, Any],
    strategies: List[str],
    history_dir: Path,
    timestamp: str,
    dry_run: bool,
    data_path: str,
    symbols: str,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """Run standalone dual-add trend strategies with multi-leg inventory accounting."""
    _ = history_dir
    dual_cfg = cfg.get("dual_add_backtest", {}) or {}
    if not bool(dual_cfg.get("enabled", True)):
        print("\n[Dual Add Backtest] ⏭️  SKIP (dual_add_backtest.enabled=false)")
        return []

    out_root = PROJECT_ROOT / str(
        dual_cfg.get("output_dir", f"results/dual_add_trend/pipeline/{timestamp}")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    print(f"\n{'='*70}")
    print("🧪 Dual Add Trend Backtest Stage")
    print(f"{'='*70}")
    print(f"   Output: {out_root}")

    for strat in strategies:
        scfg = (cfg.get("strategies", {}) or {}).get(strat, {}) or {}
        strategy_type = str(scfg.get("strategy_type", "") or "").lower()
        if strategy_type != "dual_add_trend":
            print(
                f"   ⏭️  skip {strat}: strategy_type={strategy_type or 'single_position'}"
            )
            continue
        strat_cfg_dir = PROJECT_ROOT / str(
            scfg.get("config", f"config/strategies/{strat}")
        )
        dual_yaml = strat_cfg_dir / "dual_add.yaml"
        if not dual_yaml.exists():
            raise FileNotFoundError(f"dual_add strategy config missing: {dual_yaml}")

        out_dir = out_root / strat
        cmd = [
            sys.executable,
            "scripts/diagnose_dual_add_trend.py",
            "--config",
            str(dual_yaml),
            "--data-dir",
            data_path,
            "--symbols",
            str(dual_cfg.get("symbols", symbols) or symbols),
            "--start",
            start_date,
            "--end",
            end_date,
            "--timeframe",
            str(dual_cfg.get("timeframe", scfg.get("timeframe", "2h"))),
            "--regime",
            str(dual_cfg.get("regime", "trend")),
            "--add-mode",
            str(dual_cfg.get("add_mode", "trend")),
            "--flip-action",
            str(dual_cfg.get("flip_action", "close_offside_all")),
            "--step-atr-mult",
            str(float(dual_cfg.get("step_atr_mult", 0.50))),
            "--tp-atr-mult",
            str(float(dual_cfg.get("tp_atr_mult", 0.25))),
            "--tp-pct",
            str(float(dual_cfg.get("tp_pct", 0.0005))),
            "--max-loss-per-segment",
            str(float(dual_cfg.get("max_loss_per_segment", 0.01))),
            "--max-gross-exposure",
            str(int(dual_cfg.get("max_gross_exposure", 4))),
            "--max-net-exposure",
            str(int(dual_cfg.get("max_net_exposure", 2))),
            "--max-adds-per-side",
            str(int(dual_cfg.get("max_adds_per_side", 3))),
            "--fee-bps",
            str(float(dual_cfg.get("fee_bps", 4.0))),
            "--map-symbols",
            str(dual_cfg.get("map_symbols", "BTCUSDT")),
            "--map-months",
            str(int(dual_cfg.get("map_months", 12))),
            "--continuous-map-symbols",
            str(dual_cfg.get("continuous_map_symbols", "")),
            "--continuous-map-months",
            str(int(dual_cfg.get("continuous_map_months", 0))),
            "--out-dir",
            str(out_dir),
        ]
        if bool(dual_cfg.get("exclude_box", True)):
            cmd.append("--exclude-box")
        print(f"\n   ▶️  {strat}: {' '.join(cmd)}")
        if dry_run:
            summaries.append(
                {"strategy": strat, "out_dir": str(out_dir), "dry_run": True}
            )
            continue
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(f"dual_add_backtest failed for {strat}\n{tail}")
        if proc.stdout:
            print(proc.stdout[-2000:])
        summary_path = out_dir / "summary.csv"
        summary: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                import pandas as pd

                df_summary = pd.read_csv(summary_path)
                if not df_summary.empty:
                    summary = df_summary.iloc[0].to_dict()
            except Exception as exc:  # pragma: no cover - report best effort only
                summary = {"summary_parse_error": str(exc)}
        summaries.append(
            {
                "strategy": strat,
                "out_dir": str(out_dir),
                "summary": summary,
            }
        )
    (out_root / "dual_add_backtest_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summaries


def _parse_pcm_stdout(output: str) -> Dict[str, Any]:
    """从事件回测 (PCM 多策略联合) stdout 提取指标."""
    # 复用 parse_backtest_stdout (事件回测输出格式)
    result = parse_backtest_stdout(output)
    # 映射字段名以匹配 PCM 决策逻辑
    mapped: Dict[str, Any] = {}
    mapped["total_trades"] = result.get("total_trades", 0)
    mapped["mean_r"] = result.get("mean_r")
    mapped["win_rate"] = result.get("win_rate")
    # 事件回测无 conflict_rate (per-strategy 独占 slot, 无跨策略冲突)
    mapped["conflict_rate"] = 0.0
    # sharpe_daily: 从 sharpe_per_trade 近似 (trades/holdout_days * 252)
    sharpe_r = result.get("sharpe_per_trade", 0)
    mapped["sharpe_per_trade"] = sharpe_r
    # 保守估计: 不做年化转换, 直接用 per-trade sharpe 作为参考
    mapped["sharpe_daily"] = sharpe_r
    return mapped


def _load_pcm_enabled_strategies_from_constitution(
    constitution_path: Path = PROJECT_ROOT / "config/constitution/constitution.yaml",
) -> List[str]:
    """Load PCM-enabled strategy allowlist from constitution.

    Source of truth:
      resource_allocation.enabled_archetypes
    """
    try:
        if not constitution_path.exists():
            return []
        obj = yaml.safe_load(constitution_path.read_text(encoding="utf-8")) or {}
        ra = obj.get("resource_allocation") or {}
        enabled = ra.get("enabled_archetypes") or []
        if not isinstance(enabled, list):
            return []
        return [str(x).strip() for x in enabled if str(x).strip()]
    except Exception:
        return []


def _load_pcm_metrics_from_json(json_path: str) -> Dict[str, Any]:
    """从 PCM 事件回测 JSON 中提取补充指标."""
    try:
        p = Path(json_path)
        if not p.exists():
            return {}
        obj = json.loads(p.read_text(encoding="utf-8"))
        funnel = obj.get("funnel") or {}
        total_signals_checked = int(funnel.get("total_signals_checked", 0) or 0)
        reject_pcm_slot_full = int(funnel.get("reject_pcm_slot_full", 0) or 0)
        slot_full_rate = (
            float(reject_pcm_slot_full) / float(total_signals_checked)
            if total_signals_checked > 0
            else 0.0
        )
        return {
            "sharpe_r": obj.get("sharpe_r"),
            "n_trades": obj.get("n_trades"),
            "win_rate": obj.get("win_rate"),
            "mean_r": obj.get("mean_r"),
            "total_r": obj.get("total_r"),
            "max_drawdown_r": obj.get("max_drawdown_r"),
            "add_position_stats": obj.get("add_position_stats", {}),
            "open_positions_end_count": int(
                len(obj.get("open_positions_end", []) or [])
            ),
            "total_signals_checked": total_signals_checked,
            "reject_pcm_slot_full": reject_pcm_slot_full,
            "slot_full_rate": slot_full_rate,
            "per_archetype": obj.get("per_archetype", {}),
        }
    except Exception:
        return {}


def _run_pcm_joint_backtest(
    results_summary: List[Dict[str, Any]],
    history_dir: Path,
    timestamp: str,
    *,
    dry_run: bool = False,
    use_1min: bool = False,
    live_root: str = "live/highcap",
    data_path: str = "data/parquet_data",
    holdout_start: str = "",
    end_date: str = "",
    output_stem: str = "pcm_event_backtest",
    step_name: str = "PCM Joint Event Backtest (Step 9.5)",
) -> Optional[Dict[str, Any]]:
    """Step 9.5: 全策略完成后, 执行 PCM 联合事件回测.

    Returns dict with pcm_decision, sharpe_daily, conflict_rate, etc.
    Returns None if <2 strategies completed.
    """
    # 收集已完成的策略
    strategy_names = [r["strategy"] for r in results_summary if r.get("evidence_dir")]

    # Constitution-driven allowlist filter (single source of truth).
    enabled_by_constitution = _load_pcm_enabled_strategies_from_constitution()
    if enabled_by_constitution:
        _enabled_set = set(enabled_by_constitution)
        _before = list(strategy_names)
        strategy_names = [s for s in strategy_names if s in _enabled_set]
        _removed = [s for s in _before if s not in _enabled_set]
        if _removed:
            print(f"\n{'='*70}")
            print("[Step 9.5] PCM 策略过滤 (Constitution enabled_archetypes)")
            print(
                f"   ✅ 保留: {', '.join(strategy_names) if strategy_names else '(none)'}"
            )
            print(f"   ⛔ 跳过: {', '.join(_removed)}")
            print(f"{'='*70}")

    if len(strategy_names) < 2:
        if len(results_summary) >= 2:
            print(f"\n{'='*70}")
            print("[Step 9.5] PCM 联合事件回测: ⏭️  SKIP")
            print(f"   找到 {len(strategy_names)} 个策略 (需 ≥2)")
            print(f"{'='*70}")
        return None

    # PCM 输出路径 (统一写入独立目录，避免挂在某个策略目录下造成歧义)
    first_strat = strategy_names[0]
    pcm_out_dir = history_dir / "_pcm_joint" / timestamp
    pcm_json_path = str(pcm_out_dir / f"{output_stem}.json")
    pcm_export_path = str(pcm_out_dir / f"{output_stem}_trades.csv")
    pcm_map_path = str(pcm_out_dir / f"trading_map_{output_stem}.html")

    # 日志文件
    pcm_log = pcm_out_dir / f"{output_stem}.log"
    Path(pcm_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pcm_export_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pcm_map_path).parent.mkdir(parents=True, exist_ok=True)
    pcm_log.parent.mkdir(parents=True, exist_ok=True)

    # 事件回测: --strategy bpc,fer,me-long 多策略 PCM 仲裁
    strategies_str = ",".join(strategy_names)
    # 使用第一个策略的实验 strategies_root
    strategies_root = next(
        (
            r.get("exp_config_dir", "")
            for r in results_summary
            if r["strategy"] == first_strat
        ),
        "",
    )
    # exp_config_dir 是单策略的, 对于多策略联合回测需用父目录
    if strategies_root and Path(strategies_root).name in strategy_names:
        strategies_root = str(Path(strategies_root).parent)

    cmd = [
        "python",
        "scripts/event_backtest.py",
        "--strategy",
        strategies_str,
        "--start-date",
        holdout_start,
        "--end-date",
        end_date,
        "--data-path",
        data_path,
        "--output",
        pcm_json_path,
        "--export",
        pcm_export_path,
        "--trading-map",
        pcm_map_path,
    ]
    if strategies_root:
        cmd.extend(["--strategies-root", strategies_root])

    rc, pcm_out = run_step(step_name, cmd, pcm_log, dry_run=dry_run)

    if dry_run:
        return {
            "pcm_decision": "DRY_RUN",
            "strategies": strategy_names,
            "strategies_count": len(strategy_names),
        }

    # 解析输出
    metrics = _parse_pcm_stdout(pcm_out)
    json_metrics = _load_pcm_metrics_from_json(pcm_json_path)
    sharpe_daily = metrics.get("sharpe_daily", 0)
    conflict_rate = metrics.get("conflict_rate", 0)
    total_trades = metrics.get("total_trades", 0)
    if json_metrics:
        sharpe_daily = float(json_metrics.get("sharpe_r", sharpe_daily) or sharpe_daily)
        total_trades = int(json_metrics.get("n_trades", total_trades) or total_trades)

    # PCM 决策逻辑
    pcm_decision = "PASS"
    pcm_reasons = []

    if rc != 0:
        pcm_decision = "ERROR"
        pcm_reasons.append(f"backtest exit code={rc}")
    elif total_trades < 10:
        pcm_decision = "ERROR"
        pcm_reasons.append(f"trades={total_trades} < 10")
    else:
        if conflict_rate > 0.15:
            pcm_decision = "ALERT"
            pcm_reasons.append(f"conflict_rate={conflict_rate:.2%} > 15%")
        if sharpe_daily < 1.0:
            if pcm_decision != "ALERT":
                pcm_decision = "ALERT"
            pcm_reasons.append(f"sharpe_daily={sharpe_daily:.2f} < 1.0")

    # 打印决策
    pcm_emoji = {"PASS": "\u2705", "ALERT": "\u26a0\ufe0f", "ERROR": "\u274c"}.get(
        pcm_decision, "\u2753"
    )
    print(f"\n   {pcm_emoji} PCM 决策: {pcm_decision}")
    for reason in pcm_reasons:
        print(f"      → {reason}")
    if pcm_decision == "PASS":
        print(
            f"      → sharpe_daily={sharpe_daily:.2f}, conflict_rate={conflict_rate:.2%}"
        )
    print(f"   📄 回测结果: {pcm_json_path}")
    print(f"   🗺️  交易地图: {pcm_map_path}")
    capital_report = write_capital_report_from_trades(
        trades_path=pcm_export_path,
        out_dir=pcm_out_dir,
        unit="r_multiple",
        title=f"{output_stem} Capital Report",
        initial_capital=10000.0,
        risk_per_r=0.01,
        start_date=holdout_start,
        end_date=end_date,
        total_r=json_metrics.get("total_r"),
    )

    return {
        "pcm_decision": pcm_decision,
        "pcm_reasons": pcm_reasons,
        "sharpe_daily": sharpe_daily,
        "conflict_rate": conflict_rate,
        "total_trades": total_trades,
        "mean_r": json_metrics.get("mean_r", metrics.get("mean_r")),
        "win_rate": json_metrics.get("win_rate", metrics.get("win_rate")),
        "total_r": json_metrics.get("total_r"),
        "max_drawdown_r": json_metrics.get("max_drawdown_r"),
        "total_signals_checked": json_metrics.get("total_signals_checked", 0),
        "reject_pcm_slot_full": json_metrics.get("reject_pcm_slot_full", 0),
        "slot_full_rate": json_metrics.get("slot_full_rate", 0.0),
        "per_archetype": json_metrics.get("per_archetype", {}),
        "strategies": strategy_names,
        "strategies_count": len(strategy_names),
        "trading_map": pcm_map_path,
        "capital_report": str(pcm_out_dir / "capital_report.json"),
        "capital_report_metrics": capital_report,
        "trades_csv_path": pcm_export_path,
        "log_path": str(pcm_log),
        "json_path": pcm_json_path,
    }


# ====================================================================
# Multi-seed search helpers
# ====================================================================


def _extract_gate_rules(run_dir: Path, strategy: str) -> List[str]:
    """从 seed trial 的 gate.yaml 提取 gate 特征名（含 system_safety + hard_gates）."""
    arch_dir = run_dir / "strategies" / strategy / "archetypes"
    gate_path = arch_dir / "gate.yaml"
    if not gate_path.exists():
        return []
    gt = yaml.safe_load(gate_path.read_text(encoding="utf-8")) or {}
    rules = []
    _all = list(gt.get("system_safety", []) or []) + list(
        gt.get("hard_gates", []) or []
    )
    for r in _all:
        if r.get("frozen"):
            continue  # 跳过 frozen (prefilter 注入的)
        feat = r.get("feature", r.get("id", "?"))
        rules.append(feat)
    return rules


def _select_best_seed(
    seed_trials: List[dict],
    min_trades: int = 0,
    selection: str = "best_sharpe",
) -> dict:
    """从多 seed 结果中选最佳.

    seed_trials: [{seed, run_dir, result, metrics, gate_rules}, ...]
    Returns the best trial dict.
    """
    # 筛选: 有 backtest_metrics 且无 error
    valid = [
        t
        for t in seed_trials
        if "error" not in t["result"]
        and t["metrics"].get("total_trades", 0) >= max(min_trades, 1)
    ]
    if not valid:
        # 退而求其次: 任何有 metrics 的都行
        valid = [t for t in seed_trials if "error" not in t["result"]]
    if not valid:
        # 全部失败, 返回第一个
        return seed_trials[0]

    # 排序: sharpe_per_trade 降序
    valid.sort(
        key=lambda t: t["metrics"].get("sharpe_per_trade", -999),
        reverse=True,
    )
    return valid[0]


def _print_seed_diagnostics(
    strategy: str,
    seed_trials: List[dict],
    best_trial: dict,
) -> None:
    """打印多 seed 搜索诊断表."""
    print(f"\n{'─'*60}")
    print(f"🔍 {strategy.upper()} Seed 搜索结果 ({len(seed_trials)} seeds):")
    print(f"{'─'*60}")
    print(f"  {'seed':>6s}  {'Sharpe':>8s}  {'trades':>7s}  {'win%':>6s}  gate_rules")
    print(f"  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*20}")
    for t in seed_trials:
        m = t["metrics"]
        sharpe = m.get("sharpe_per_trade", 0)
        trades = m.get("total_trades", 0)
        win = m.get("win_rate", 0)
        rules = t.get("gate_rules", [])
        marker = " 🏆" if t["seed"] == best_trial["seed"] else ""
        err = " ❌" if "error" in t["result"] else ""
        print(
            f"  {t['seed']:>6d}  {sharpe:>8.4f}  {trades:>7.0f}  "
            f"{win*100:>5.1f}%  {', '.join(rules) if rules else '(error)'}{marker}{err}"
        )

    # 稳定性诊断: 统计每个特征被选中的次数
    from collections import Counter

    feat_counts = Counter()
    for t in seed_trials:
        if "error" not in t["result"]:
            for f in t.get("gate_rules", []):
                feat_counts[f] += 1
    n_valid = sum(1 for t in seed_trials if "error" not in t["result"])
    if feat_counts and n_valid > 1:
        print(f"\n  📊 Gate 特征稳定性:")
        for feat, cnt in feat_counts.most_common():
            pct = cnt / n_valid * 100
            bar = "█" * int(pct / 10)
            print(f"     {feat:<30s} {cnt}/{n_valid} ({pct:.0f}%) {bar}")
    print()


def _strategy_type(cfg: Dict[str, Any], strategy: str) -> str:
    scfg = (cfg.get("strategies", {}) or {}).get(strategy, {}) or {}
    return str(scfg.get("strategy_type", "") or "").lower()


def _is_multi_leg_strategy(cfg: Dict[str, Any], strategy: str) -> bool:
    return _strategy_type(cfg, strategy) in {"grid", "dual_add_trend"}


def _resolve_strategy_config_dir(
    cfg: Dict[str, Any],
    strategy: str,
    strategies_root: Path,
) -> Path:
    scfg = (cfg.get("strategies", {}) or {}).get(strategy, {}) or {}
    cfg_dir = str(scfg.get("config", "") or "").strip()
    default_dir = strategies_root / strategy
    if cfg_dir:
        explicit = Path(cfg_dir)
        if not explicit.is_absolute():
            explicit = PROJECT_ROOT / explicit
        if explicit.exists():
            return explicit.resolve()
    return default_dir.resolve()


def _multileg_calibration_candidates(strategy_type: str) -> List[Dict[str, Any]]:
    if strategy_type == "grid":
        return [
            {
                "box_window": 60,
                "entry_chop_min": 0.35,
                "exit_chop_below": 0.22,
                "atr_mult": 0.50,
                "min_pct": 0.004,
                "exclude_box_prefilter": True,
            },
            {
                "box_window": 120,
                "entry_chop_min": 0.40,
                "exit_chop_below": 0.25,
                "atr_mult": 0.50,
                "min_pct": 0.004,
                "exclude_box_prefilter": True,
            },
            {
                "box_window": 240,
                "entry_chop_min": 0.45,
                "exit_chop_below": 0.30,
                "atr_mult": 0.65,
                "min_pct": 0.006,
                "exclude_box_prefilter": False,
            },
        ]
    if strategy_type == "dual_add_trend":
        return [
            {
                "box_window": 60,
                "entry_min": 0.75,
                "exit_below": 0.45,
                "max_semantic_chop_entry": 0.20,
                "max_semantic_chop_hold": 0.35,
                "step_atr_mult": 0.50,
            },
            {
                "box_window": 120,
                "entry_min": 0.80,
                "exit_below": 0.50,
                "max_semantic_chop_entry": 0.25,
                "max_semantic_chop_hold": 0.40,
                "step_atr_mult": 0.50,
            },
            {
                "box_window": 240,
                "entry_min": 0.85,
                "exit_below": 0.55,
                "max_semantic_chop_entry": 0.30,
                "max_semantic_chop_hold": 0.45,
                "step_atr_mult": 0.65,
            },
        ]
    return [{}]


def _apply_multileg_candidate(
    strategy_type: str,
    config_dir: Path,
    candidate: Dict[str, Any],
) -> None:
    if strategy_type == "grid":
        path = config_dir / "grid.yaml"
        obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        regime = obj.setdefault("regime", {})
        grid = obj.setdefault("grid", {})
        spacing = grid.setdefault("spacing", {})
        if "box_window" in candidate:
            regime["box_window"] = int(candidate["box_window"])
        if "entry_chop_min" in candidate:
            regime["entry_chop_min"] = float(candidate["entry_chop_min"])
        if "exit_chop_below" in candidate:
            regime["exit_chop_below"] = float(candidate["exit_chop_below"])
        if "exclude_box_prefilter" in candidate:
            regime["exclude_box_prefilter"] = bool(candidate["exclude_box_prefilter"])
        if "atr_mult" in candidate:
            spacing["atr_mult"] = float(candidate["atr_mult"])
        if "min_pct" in candidate:
            spacing["min_pct"] = float(candidate["min_pct"])
        path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")
        return

    if strategy_type == "dual_add_trend":
        path = config_dir / "dual_add.yaml"
        obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        regime = obj.setdefault("regime", {})
        inv = obj.setdefault("inventory", {})
        spacing = obj.setdefault("add_spacing", {})
        tp = obj.setdefault("take_profit", {})
        if "box_window" in candidate:
            regime["box_window"] = int(candidate["box_window"])
        if "entry_min" in candidate:
            regime["entry_min"] = float(candidate["entry_min"])
        if "exit_below" in candidate:
            regime["exit_below"] = float(candidate["exit_below"])
        if "max_semantic_chop_entry" in candidate:
            regime["max_semantic_chop_entry"] = float(
                candidate["max_semantic_chop_entry"]
            )
        if "max_semantic_chop_hold" in candidate:
            regime["max_semantic_chop_hold"] = float(
                candidate["max_semantic_chop_hold"]
            )
        if "step_atr_mult" in candidate:
            spacing["atr_mult"] = float(candidate["step_atr_mult"])
        if "tp_atr_mult" in candidate:
            tp["atr_mult"] = float(candidate["tp_atr_mult"])
        if "tp_pct" in candidate:
            tp["min_pct"] = float(candidate["tp_pct"])
        if "flip_action" in candidate:
            inv["flip_action"] = str(candidate["flip_action"])
        path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def _run_multileg_backtest_command(
    *,
    cfg: Dict[str, Any],
    strategy: str,
    config_dir: Path,
    data_path: str,
    symbols: str,
    start_date: str,
    end_date: str,
    out_dir: Path,
    with_maps: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    scfg = (cfg.get("strategies", {}) or {}).get(strategy, {}) or {}
    strategy_type = _strategy_type(cfg, strategy)
    out_dir.mkdir(parents=True, exist_ok=True)
    timeframe = str(scfg.get("timeframe", "2h"))
    if timeframe == "120T":
        timeframe = "2h"

    if strategy_type == "grid":
        grid_cfg = cfg.get("grid_backtest", {}) or {}
        cmd = [
            sys.executable,
            "scripts/chop_grid_backtest.py",
            "--config",
            str(config_dir / "grid.yaml"),
            "--data-dir",
            data_path,
            "--symbols",
            symbols,
            "--start",
            start_date,
            "--end",
            end_date,
            "--timeframe",
            timeframe,
            "--out-dir",
            str(out_dir),
        ]
        if with_maps:
            cmd.extend(
                [
                    "--map-symbols",
                    str(grid_cfg.get("map_symbols", symbols) or symbols),
                    "--map-months",
                    str(int(grid_cfg.get("map_months", 12) or 12)),
                    "--continuous-map-symbols",
                    str(grid_cfg.get("continuous_map_symbols", symbols) or symbols),
                    "--continuous-map-months",
                    str(int(grid_cfg.get("continuous_map_months", 0) or 0)),
                ]
            )
        else:
            cmd.append("--no-maps")
        return cmd, {"metrics_path": str(out_dir / "metrics.json")}

    if strategy_type == "dual_add_trend":
        dual_cfg = cfg.get("dual_add_backtest", {}) or {}
        cmd = [
            sys.executable,
            "scripts/diagnose_dual_add_trend.py",
            "--config",
            str(config_dir / "dual_add.yaml"),
            "--data-dir",
            data_path,
            "--symbols",
            str(dual_cfg.get("symbols", symbols) or symbols),
            "--start",
            start_date,
            "--end",
            end_date,
            "--timeframe",
            timeframe,
            "--out-dir",
            str(out_dir),
        ]
        if bool(dual_cfg.get("exclude_box", True)):
            cmd.append("--exclude-box")
        if with_maps:
            cmd.extend(
                [
                    "--map-symbols",
                    str(dual_cfg.get("map_symbols", "BTCUSDT")),
                    "--map-months",
                    str(int(dual_cfg.get("map_months", 12))),
                    "--continuous-map-symbols",
                    str(dual_cfg.get("continuous_map_symbols", symbols) or symbols),
                    "--continuous-map-months",
                    str(int(dual_cfg.get("continuous_map_months", 0))),
                ]
            )
        else:
            cmd.append("--no-maps")
        return cmd, {"summary_path": str(out_dir / "summary.csv")}

    raise ValueError(f"unsupported multi-leg strategy_type={strategy_type}")


def _parse_multileg_metrics(strategy_type: str, out_dir: Path) -> Dict[str, Any]:
    if strategy_type == "grid":
        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists():
            return {"n_trades": 0, "sharpe_r": 0.0}
        obj = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = obj.get("metrics", {}) or {}
        trade = metrics.get("trade_summary", {}) or {}
        segment = metrics.get("segment_summary", {}) or {}
        n_trades = int(trade.get("trades", 0) or 0)
        pnl = float(trade.get("sum_pnl_per_capital", 0.0) or 0.0)
        return {
            "n_trades": n_trades,
            "sharpe_r": float(trade.get("trade_sharpe", 0.0) or 0.0),
            "mean_r": float(pnl / max(n_trades, 1)),
            "total_r": pnl,
            "win_rate": float(trade.get("win_rate", 0.0) or 0.0),
            "max_drawdown_r": float(trade.get("max_drawdown", 0.0) or 0.0),
            "near_stop_rate": float(trade.get("forced_rate", 0.0) or 0.0),
            "worst_segment": float(segment.get("worst_segment", 0.0) or 0.0),
            "segment_win_rate": float(segment.get("segment_win_rate", 0.0) or 0.0),
            "forced_rate": float(trade.get("forced_rate", 0.0) or 0.0),
        }

    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        return {"n_trades": 0, "sharpe_r": 0.0}
    import pandas as pd

    df = pd.read_csv(summary_path)
    if df.empty:
        return {"n_trades": 0, "sharpe_r": 0.0}
    row = df.iloc[0].to_dict()
    n_trades = int(row.get("trades", 0) or 0)
    pnl = float(row.get("sum_pnl_per_capital", 0.0) or 0.0)
    risk_stop = float(row.get("risk_stop_rate", 0.0) or 0.0)
    forced = float(row.get("forced_rate", 0.0) or 0.0)
    return {
        "n_trades": n_trades,
        "sharpe_r": float(row.get("segment_win_rate", 0.0) or 0.0),
        "mean_r": float(pnl / max(n_trades, 1)),
        "total_r": pnl,
        "win_rate": float(row.get("trade_win_rate", 0.0) or 0.0),
        "segment_win_rate": float(row.get("segment_win_rate", 0.0) or 0.0),
        "max_drawdown_r": float(row.get("median_drawdown", 0.0) or 0.0),
        "near_stop_rate": risk_stop,
        "worst_segment": float(row.get("worst_segment", 0.0) or 0.0),
        "risk_stop_rate": risk_stop,
        "forced_rate": forced,
        "max_gross_units": int(row.get("max_gross_units", 0) or 0),
        "max_abs_net_units": int(row.get("max_abs_net_units", 0) or 0),
    }


def _score_multileg_candidate(metrics: Dict[str, Any]) -> float:
    total = float(metrics.get("total_r", 0.0) or 0.0)
    worst = float(metrics.get("worst_segment", 0.0) or 0.0)
    forced = float(metrics.get("forced_rate", 0.0) or 0.0)
    risk_stop = float(
        metrics.get("risk_stop_rate", metrics.get("near_stop_rate", 0.0)) or 0.0
    )
    return total + 5.0 * worst - 0.25 * forced - 0.50 * risk_stop


def _run_multileg_month_strategy(
    *,
    cfg: Dict[str, Any],
    strategy: str,
    run_root: Path,
    month_strategies_root: Path,
    base_strategies_root: Path,
    data_path: str,
    symbols: str,
    calib_start: str,
    calib_end: str,
    test_start: str,
    test_end: str,
    dry_run: bool,
    calibrate: bool,
) -> Dict[str, Any]:
    strategy_type = _strategy_type(cfg, strategy)
    source_dir = _resolve_strategy_config_dir(cfg, strategy, base_strategies_root)
    calibrated_dir = month_strategies_root / strategy
    shutil.rmtree(calibrated_dir, ignore_errors=True)
    shutil.copytree(source_dir, calibrated_dir, dirs_exist_ok=True)

    best: Dict[str, Any] = {"candidate": {}, "metrics": {}, "score": 0.0}
    calib_dir = run_root / strategy / "multileg_calibration"
    if calibrate and not dry_run:
        calib_dir.mkdir(parents=True, exist_ok=True)
        candidates = _multileg_calibration_candidates(strategy_type)
        rows = []
        for idx, candidate in enumerate(candidates, start=1):
            cand_cfg_dir = calib_dir / f"candidate_{idx:02d}" / "config"
            cand_out_dir = calib_dir / f"candidate_{idx:02d}" / "results"
            shutil.copytree(source_dir, cand_cfg_dir, dirs_exist_ok=True)
            _apply_multileg_candidate(strategy_type, cand_cfg_dir, candidate)
            cmd, _ = _run_multileg_backtest_command(
                cfg=cfg,
                strategy=strategy,
                config_dir=cand_cfg_dir,
                data_path=data_path,
                symbols=symbols,
                start_date=calib_start,
                end_date=calib_end,
                out_dir=cand_out_dir,
                with_maps=False,
            )
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            if proc.returncode != 0:
                rows.append(
                    {
                        "candidate": candidate,
                        "error": (proc.stderr or proc.stdout or "")[-1000:],
                    }
                )
                continue
            metrics = _parse_multileg_metrics(strategy_type, cand_out_dir)
            score = _score_multileg_candidate(metrics)
            rows.append({"candidate": candidate, "metrics": metrics, "score": score})
            if not best["metrics"] or score > float(best["score"]):
                best = {"candidate": candidate, "metrics": metrics, "score": score}
        (calib_dir / "calibration_results.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        if best["metrics"]:
            shutil.rmtree(calibrated_dir, ignore_errors=True)
            shutil.copytree(source_dir, calibrated_dir, dirs_exist_ok=True)
            _apply_multileg_candidate(strategy_type, calibrated_dir, best["candidate"])

    strat_dir = run_root / strategy
    strat_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        metrics = {"n_trades": 0, "sharpe_r": 0.0, "mean_r": 0.0, "total_r": 0.0}
        result = {"rc": 0, "metrics": metrics, "map_path": "", "json_path": ""}
    else:
        cmd, _ = _run_multileg_backtest_command(
            cfg=cfg,
            strategy=strategy,
            config_dir=calibrated_dir,
            data_path=data_path,
            symbols=symbols,
            start_date=test_start,
            end_date=test_end,
            out_dir=strat_dir,
            with_maps=True,
        )
        print(f"\n   ▶️  {strategy}: multi-leg {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(f"multi-leg backtest failed for {strategy}\n{tail}")
        if proc.stdout:
            print(proc.stdout[-1200:])
        metrics = _parse_multileg_metrics(strategy_type, strat_dir)
        result = {
            "rc": 0,
            "metrics": metrics,
            "map_path": str(strat_dir / "trading_map_continuous.html"),
            "json_path": str(strat_dir / "multileg_summary.json"),
            "capital_report": str(strat_dir / "capital_report.json"),
        }
    summary = {
        "strategy": strategy,
        "strategy_type": strategy_type,
        "calibration_window": {"start": calib_start, "end": calib_end},
        "test_window": {"start": test_start, "end": test_end},
        "calibrated_config_dir": str(calibrated_dir),
        "best_calibration": best,
        "metrics": result["metrics"],
        "map_path": result["map_path"],
        "capital_report": result.get("capital_report", ""),
    }
    (strat_dir / "multileg_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return result


def _collect_multileg_stitched_metrics(
    *,
    cfg: Dict[str, Any],
    ledger: List[Dict[str, Any]],
    strategies: List[str],
) -> Dict[str, Any]:
    month_rows: List[Dict[str, Any]] = []
    total_r = 0.0
    total_trades = 0
    worst_segment: Optional[float] = None
    worst_single_month_dd: Optional[float] = None
    trade_points: List[Tuple[Any, float]] = []

    for row in ledger:
        run_root = Path(str(row.get("run_root", "") or ""))
        month = str(row.get("month", "") or "")
        month_r = 0.0
        month_trades = 0
        month_dd = 0.0
        active = False
        for strategy in strategies:
            if not _is_multi_leg_strategy(cfg, strategy):
                continue
            summary_path = run_root / strategy / "multileg_summary.json"
            if _strategy_type(cfg, strategy) == "grid":
                trade_path = run_root / strategy / "grid_trades.csv"
            else:
                trade_path = run_root / strategy / "dual_add_trades.csv"
            if trade_path.exists():
                try:
                    import pandas as pd

                    tdf = pd.read_csv(trade_path)
                    if (
                        not tdf.empty
                        and "exit_time" in tdf.columns
                        and "pnl_per_capital" in tdf.columns
                    ):
                        for _, tr in tdf.iterrows():
                            trade_points.append(
                                (
                                    pd.Timestamp(tr["exit_time"]),
                                    float(tr.get("pnl_per_capital", 0.0) or 0.0),
                                )
                            )
                except Exception:
                    pass
            if not summary_path.exists():
                continue
            try:
                obj = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metrics = obj.get("metrics", {}) or {}
            active = True
            _r = float(metrics.get("total_r", 0.0) or 0.0)
            _n = int(metrics.get("n_trades", 0) or 0)
            _dd = float(metrics.get("max_drawdown_r", 0.0) or 0.0)
            _worst = float(metrics.get("worst_segment", 0.0) or 0.0)
            month_r += _r
            month_trades += _n
            month_dd = min(month_dd, _dd)
            worst_segment = (
                _worst if worst_segment is None else min(worst_segment, _worst)
            )
            worst_single_month_dd = (
                _dd
                if worst_single_month_dd is None
                else min(worst_single_month_dd, _dd)
            )
        if active:
            month_rows.append(
                {
                    "month": month,
                    "total_r": month_r,
                    "trades": month_trades,
                    "max_drawdown_r": month_dd,
                }
            )
            total_r += month_r
            total_trades += month_trades

    equity = 0.0
    peak = 0.0
    stitched_maxdd = 0.0
    if trade_points:
        for _, pnl in sorted(trade_points, key=lambda x: x[0]):
            equity += float(pnl)
            peak = max(peak, equity)
            stitched_maxdd = min(stitched_maxdd, equity - peak)
    else:
        for row in month_rows:
            equity += float(row.get("total_r", 0.0) or 0.0)
            peak = max(peak, equity)
            stitched_maxdd = min(stitched_maxdd, equity - peak)

    return {
        "total_r": total_r,
        "total_trades": total_trades,
        "max_drawdown_r": stitched_maxdd,
        "max_drawdown_source": "trade_equity" if trade_points else "monthly_equity",
        "worst_single_month_drawdown_r": worst_single_month_dd or 0.0,
        "worst_segment": worst_segment or 0.0,
        "monthly": month_rows,
    }


def _build_multileg_rolling_continuous_map(
    *,
    cfg: Dict[str, Any],
    ledger: List[Dict[str, Any]],
    strategies: List[str],
    roll_root: Path,
    data_path: str,
    output_path: Path,
) -> str:
    import pandas as pd

    trade_frames = []
    segment_frames = []
    for row in ledger:
        run_root = Path(str(row.get("run_root", "") or ""))
        if not run_root.is_dir():
            continue
        for strategy in strategies:
            if not _is_multi_leg_strategy(cfg, strategy):
                continue
            strat_dir = run_root / strategy
            if _strategy_type(cfg, strategy) == "grid":
                trade_path = strat_dir / "grid_trades.csv"
                segment_path = strat_dir / "grid_segments.csv"
            else:
                trade_path = strat_dir / "dual_add_trades.csv"
                segment_path = strat_dir / "dual_add_segments.csv"
            if trade_path.exists():
                df = pd.read_csv(trade_path)
                if not df.empty:
                    df["strategy"] = strategy
                    trade_frames.append(df)
            if segment_path.exists():
                df = pd.read_csv(segment_path)
                if not df.empty:
                    df["strategy"] = strategy
                    segment_frames.append(df)
    if not trade_frames and not segment_frames:
        return ""
    trades = (
        pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    )
    segments = (
        pd.concat(segment_frames, ignore_index=True)
        if segment_frames
        else pd.DataFrame()
    )
    dates = cfg.get("dates", {}) or {}
    ledger_months = [str(r.get("month", "") or "") for r in ledger if r.get("month")]
    if ledger_months:
        report_start, _ = _month_token_to_range(ledger_months[0])
        _, report_end = _month_token_to_range(ledger_months[-1])
    else:
        report_start = str(dates.get("start_date") or "")
        report_end = str(dates.get("end_date") or "")
    combined_trades_path = output_path.parent / "multi_leg_all_trades.csv"
    trades.to_csv(combined_trades_path, index=False)
    write_capital_report_from_trades(
        trades_path=combined_trades_path,
        out_dir=output_path.parent,
        unit="capital_normalized",
        title="Multi-Leg Rolling Capital Report",
        initial_capital=_capital_report_initial_capital(cfg),
        risk_per_r=_capital_report_risk_per_r(cfg),
        start_date=report_start,
        end_date=report_end,
        total_r=(
            float(trades["pnl_per_capital"].sum())
            if "pnl_per_capital" in trades.columns
            else None
        ),
    )
    write_continuous_trading_map(
        out_path=output_path,
        data_dir=PROJECT_ROOT / str(data_path),
        symbols=resolve_symbols_from_config(cfg),
        map_symbols=resolve_symbols_from_config(cfg),
        timeframe="2h",
        start=str(dates.get("start_date") or "2022-01-01"),
        end=str(dates.get("end_date") or ""),
        warmup_days=120,
        map_months=0,
        trades=trades,
        segments=segments,
        title="Multi-Leg Rolling Continuous Trading Map",
    )
    return str(output_path) if output_path.exists() else ""


def _run_fast_month_stage(
    *,
    cfg: Dict[str, Any],
    strategies: List[str],
    history_dir: Path,
    timestamp: str,
    month_token: str,
    dry_run: bool,
    use_1min: bool,
    live_root: str,
    data_path: str,
    event_sym_r: str,
    strategies_root: str = "",
    calibration_months: int = 3,
    calibrate_all_layers: bool = True,
    feature_search_enabled: bool = True,
    rolling_mode: str = "legacy",
    config_path: str = "",
    prev_side_state: Optional[Dict[str, Any]] = None,
    prev_resume_state_paths: Optional[Dict[str, str]] = None,
    keep_open_positions: bool = False,
    month_index: int = 0,
) -> Dict[str, Any]:
    """Run one-month fast loop with optional calib/test window split."""
    month_start, month_end = _month_token_to_range(month_token)
    windows = _calib_and_test_windows(
        month_token=month_token, calibration_months=int(calibration_months)
    )
    calib_start = windows["calib_start"]
    calib_end = windows["calib_end"]
    test_start = windows["test_start"]
    test_end = windows["test_end"]
    if not calibrate_all_layers:
        # Legacy-compatible behavior: optimize and backtest in the same month window.
        calib_start, calib_end = test_start, test_end
    fast_loop_cfg = cfg.get("fast_loop", {}) or {}
    event_cfg = cfg.get("event_backtest", {}) or {}
    if isinstance(event_cfg, dict) and "enabled" not in event_cfg:
        print(
            "⚠️  event_backtest.enabled 未显式设置；当前按默认 true 处理。"
            "建议在配置中明确声明。"
        )
    # Keep legacy path backward compatible: ignore fast_loop toggles there.
    _fast_loop_active = rolling_mode != "legacy"

    def _section_enabled(name: str, default: bool = True) -> bool:
        sec = fast_loop_cfg.get(name, {}) or {}
        if isinstance(sec, dict):
            return bool(sec.get("enabled", default))
        return bool(default)

    threshold_calibration_enabled = _section_enabled("threshold_calibration", True)
    _threshold_cfg = fast_loop_cfg.get("threshold_calibration", {}) or {}
    _fast_disable_train_raw = fast_loop_cfg.get("disable_model_training", None)
    if _fast_disable_train_raw is None:
        threshold_calibration_disable_model_training = (
            bool(_threshold_cfg.get("disable_model_training", False))
            if isinstance(_threshold_cfg, dict)
            else False
        )
    else:
        threshold_calibration_disable_model_training = bool(_fast_disable_train_raw)
    prefilter_cfg = fast_loop_cfg.get("prefilter", {}) or {}
    prefilter_optimize_enabled = (
        bool(prefilter_cfg.get("optimize", True))
        if isinstance(prefilter_cfg, dict)
        else True
    )
    execution_opt_enabled = _section_enabled("execution_opt", True)
    pcm_eval_enabled = _section_enabled("pcm_eval", True)
    direction_tuning_enabled = _section_enabled("direction_tuning", True)
    # rolling.windows.calibration_months = 标定窗口长度 (见 _calib_and_test_windows)，
    # 与「多久跑一次方向调优」无关。方向节奏单独用 direction_tuning.cadence_months，默认 1=每月。
    _dir_tune_sec = fast_loop_cfg.get("direction_tuning") or {}
    _direction_cadence_raw = (
        _dir_tune_sec.get("cadence_months", 1) if isinstance(_dir_tune_sec, dict) else 1
    )
    try:
        _direction_cadence = max(int(_direction_cadence_raw), 1)
    except (TypeError, ValueError):
        _direction_cadence = 1
    skip_direction_for_month = not direction_tuning_enabled or (
        month_index % _direction_cadence != 0
    )
    event_backtest_enabled = bool(event_cfg.get("enabled", True))
    if not _fast_loop_active:
        threshold_calibration_enabled = True
        threshold_calibration_disable_model_training = False
        prefilter_optimize_enabled = True
        execution_opt_enabled = True
        pcm_eval_enabled = True
        event_backtest_enabled = True
        skip_direction_for_month = False
    _calibrate_threshold_layers = bool(
        calibrate_all_layers and threshold_calibration_enabled
    )
    if rolling_mode == "legacy":
        # Legacy path keeps original lightweight behavior: no prior-window layer calibration.
        _calibrate_threshold_layers = False
    run_root = history_dir / "_rolling_sim" / timestamp / f"fast_month_{month_token}"
    run_root.mkdir(parents=True, exist_ok=True)
    _base_strategies_root = (
        Path(strategies_root)
        if str(strategies_root or "").strip()
        else (PROJECT_ROOT / "config" / "strategies")
    )
    if not _base_strategies_root.is_absolute():
        _base_strategies_root = PROJECT_ROOT / _base_strategies_root
    _base_strategies_root = _base_strategies_root.resolve()

    results_summary: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    side_state: Dict[str, Any] = {}
    end_state_paths: Dict[str, str] = {}
    prev_side_state = prev_side_state or {}
    prev_resume_state_paths = prev_resume_state_paths or {}
    symbol_policy = cfg.get("symbol_policy", {}) or {}
    enable_threshold = float(symbol_policy.get("enable_threshold", 0.0) or 0.0)
    min_symbol_trades_soft = int(symbol_policy.get("min_symbol_trades_soft", 10) or 10)
    hard_fail_min_sharpe = float(
        (symbol_policy.get("carry_forward_hard_fail_rules", {}) or {}).get(
            "min_sharpe_r", -0.25
        )
        or -0.25
    )
    slot_alloc = cfg.get("slot_allocation", {}) or {}
    quality_w = slot_alloc.get("quality_score_weights", {}) or {}
    hist_w = float(quality_w.get("history_edge", 0.55) or 0.55)
    now_w = float(quality_w.get("now_strength", 0.45) or 0.45)
    max_symbols_per_side = int(slot_alloc.get("max_symbols_per_side", 2) or 2)

    print(f"\n{'='*70}")
    print(f"🗓️  Fast Month Replay: {month_token} ({test_start} ~ {test_end})")
    print(
        f"   mode={rolling_mode} | calib={calib_start}~{calib_end} | "
        f"test={test_start}~{test_end}"
    )
    print(f"   base_strategies_root={_base_strategies_root}")
    print(
        "   fast_loop: "
        f"threshold_calibration={threshold_calibration_enabled}, "
        f"disable_model_training={threshold_calibration_disable_model_training}, "
        f"prefilter_optimize={prefilter_optimize_enabled}, "
        f"execution_opt={execution_opt_enabled}, "
        f"pcm_eval={pcm_eval_enabled}, "
        f"direction_tuning={direction_tuning_enabled} "
        f"(cadence_months={_direction_cadence}, month_idx={month_index}, "
        f"run_this_month={not skip_direction_for_month}), "
        f"event_backtest={event_backtest_enabled}"
    )
    if threshold_calibration_enabled and threshold_calibration_disable_model_training:
        print("   NO_MODEL_TUNING: prefilter/gate/entry_filter")
    print(f"{'='*70}")

    # A) Calibrate prefilter/gate/entry on prior window into a month-scoped strategy root.
    month_strategies_root = run_root / "strategies_calibrated"
    month_strategies_root.mkdir(parents=True, exist_ok=True)
    if _calibrate_threshold_layers:
        for strat in strategies:
            if strat not in cfg.get("strategies", {}):
                continue
            if _is_multi_leg_strategy(cfg, strat):
                continue
            strat_calib_dir = run_root / strat / "threshold_calibration"
            strat_calib_dir.mkdir(parents=True, exist_ok=True)
            scfg = cfg["strategies"][strat]
            strat_dates = (
                scfg.get("dates", {}) if isinstance(scfg.get("dates"), dict) else {}
            )
            _start_date = str(
                strat_dates.get(
                    "start_date", (cfg.get("dates", {}) or {}).get("start_date")
                )
            )
            _res = pipeline_strategy.run_strategy_pipeline(
                strat,
                cfg,
                end_date=calib_end,
                holdout_start=calib_start,
                holdout_months=int(calibration_months),
                validation_months=0,
                start_date=_start_date,
                symbols=resolve_symbols_from_config(cfg),
                data_path=data_path,
                run_dir=strat_calib_dir,
                seed=42,
                dry_run=dry_run,
                use_1min=use_1min,
                live_root=live_root,
                skip_shap=(not feature_search_enabled),
                feature_search_enabled=feature_search_enabled,
                threshold_calibration_enabled=True,
                prefilter_optimization_enabled=prefilter_optimize_enabled,
                source_strategies_root=str(_base_strategies_root),
                config_path=(config_path or str(DEFAULT_CONFIG)),
                stage_stop="entry_filter",
                disable_model_training=threshold_calibration_disable_model_training,
                skip_direction_tuning=skip_direction_for_month,
            )
            exp_cfg_dir = Path(str(_res.get("exp_config_dir", "") or ""))
            if exp_cfg_dir.exists():
                dst = month_strategies_root / strat
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(exp_cfg_dir, dst, dirs_exist_ok=True)
    else:
        for strat in strategies:
            # 同上：优先尊重 strategies.<strat>.config 的显式 per-strategy 覆盖。
            _strat_scfg = cfg.get("strategies", {}).get(strat, {}) or {}
            _strat_prod_cfg = str(_strat_scfg.get("config", "") or "").strip()
            _strat_default = (PROJECT_ROOT / "config" / "strategies" / strat).resolve()
            src: Optional[Path] = None
            if _strat_prod_cfg:
                _prod_abs = (PROJECT_ROOT / _strat_prod_cfg).resolve()
                if _prod_abs != _strat_default and _prod_abs.exists():
                    src = _prod_abs
            if src is None:
                _cand = _base_strategies_root / strat
                if _cand.exists():
                    src = _cand
                elif _strat_prod_cfg:
                    src = (PROJECT_ROOT / _strat_prod_cfg).resolve()
            dst = month_strategies_root / strat
            if src is not None and src.exists():
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)

    # B) Calibrate execution on prior window and promote into month_strategies_root.
    if _calibrate_threshold_layers and execution_opt_enabled:
        for strat in strategies:
            if strat not in cfg.get("strategies", {}):
                continue
            if _is_multi_leg_strategy(cfg, strat):
                continue
            obj_cfg = _resolve_event_exec_objective_for_strategy(cfg, strat)
            sym_r = _resolve_event_sym_r_for_strategy(cfg, strat, event_sym_r)
            _exec_grid = _resolve_event_exec_grid_for_strategy(cfg, strat)
            _opt_dir = run_root / strat / "execution_calibration"
            _opt_dir.mkdir(parents=True, exist_ok=True)
            pipeline_events.run_event_execution_opt_only(
                strat,
                _opt_dir,
                holdout_start=calib_start,
                end_date=calib_end,
                strategies_root=str(month_strategies_root),
                data_path=data_path,
                dry_run=dry_run,
                sym_r=sym_r,
                exec_grid=_exec_grid,
                promote=True,
                objective=str(obj_cfg["objective"]),
                near_stop_threshold_r=float(obj_cfg["near_stop_threshold_r"]),
                near_stop_penalty=float(obj_cfg["near_stop_penalty"]),
                max_dd_penalty=float(obj_cfg["max_dd_penalty"]),
                min_trades_soft=int(obj_cfg["min_trades_soft"]),
                undertrade_penalty=float(obj_cfg["undertrade_penalty"]),
            )

    for strat in strategies:
        if strat not in cfg.get("strategies", {}):
            continue
        strat_dir = run_root / strat
        strat_dir.mkdir(parents=True, exist_ok=True)
        end_state_path = str(strat_dir / "end_state.json")
        if _is_multi_leg_strategy(cfg, strat):
            ev = _run_multileg_month_strategy(
                cfg=cfg,
                strategy=strat,
                run_root=run_root,
                month_strategies_root=month_strategies_root,
                base_strategies_root=_base_strategies_root,
                data_path=data_path,
                symbols=resolve_symbols_from_config(cfg),
                calib_start=calib_start,
                calib_end=calib_end,
                test_start=test_start,
                test_end=test_end,
                dry_run=dry_run,
                calibrate=_calibrate_threshold_layers,
            )
        elif event_backtest_enabled:
            obj_cfg = _resolve_event_exec_objective_for_strategy(cfg, strat)
            sym_r = _resolve_event_sym_r_for_strategy(cfg, strat, event_sym_r)
            _exec_grid = _resolve_event_exec_grid_for_strategy(cfg, strat)
            _grid_label = f"grid={_exec_grid}" if _exec_grid else f"sym-r={sym_r}"
            print(f"\n   ▶️  {strat}: {_grid_label}, objective={obj_cfg['objective']}")
            resume_state_path = str(prev_resume_state_paths.get(strat, "") or "")
            ev = pipeline_events.run_event_backtest_step(
                strat,
                str(strat_dir),
                strat_dir,
                holdout_start=test_start,
                end_date=test_end,
                strategies_root=str(month_strategies_root),
                data_path=data_path,
                dry_run=dry_run,
                sym_r=sym_r,
                exec_grid=_exec_grid,
                promote=False,
                objective=str(obj_cfg["objective"]),
                near_stop_threshold_r=float(obj_cfg["near_stop_threshold_r"]),
                near_stop_penalty=float(obj_cfg["near_stop_penalty"]),
                max_dd_penalty=float(obj_cfg["max_dd_penalty"]),
                min_trades_soft=int(obj_cfg["min_trades_soft"]),
                undertrade_penalty=float(obj_cfg["undertrade_penalty"]),
                run_execution_opt=(not _calibrate_threshold_layers)
                and execution_opt_enabled,
                opt_start_date=calib_start,
                opt_end_date=calib_end,
                event_start_date=test_start,
                event_end_date=test_end,
                resume_state_path=resume_state_path,
                dump_end_state_path=end_state_path,
                keep_open_positions=keep_open_positions,
                map_extra_months=_event_trading_map_extra_months(cfg),
                no_kill_switch=bool(
                    (cfg.get("event_backtest") or {}).get("no_kill_switch", False)
                ),
            )
        else:
            print(f"   ⏭️  跳过事件回测: event_backtest.enabled=false ({strat})")
            ev = {
                "rc": 0,
                "metrics": {"n_trades": 0, "sharpe_r": 0.0, "mean_r": 0.0},
                "map_path": "",
                "json_path": "",
                "end_state_path": end_state_path,
            }
        end_state_paths[strat] = end_state_path
        m = ev.get("metrics", {}) or {}
        q, q_components = _quality_score_from_event_metrics(
            m, history_w=hist_w, now_w=now_w
        )
        ranking_rows.append(
            {
                "strategy": strat,
                "month": month_token,
                "quality_score": float(q),
                "quality_components": q_components,
                "metrics": m,
                "map_path": ev.get("map_path"),
            }
        )
        side = _resolve_strategy_side(
            strat, cfg.get("strategies", {}).get(strat, {}) or {}
        )
        _sh = float(m.get("sharpe_r", 0.0) or 0.0)
        _nt = int(m.get("n_trades", 0) or 0)
        active = bool(_sh > enable_threshold and _nt >= min_symbol_trades_soft)
        prev_state = (prev_side_state.get(strat, {}) or {}).get("state", "disabled")
        hard_fail = _sh <= hard_fail_min_sharpe
        can_carry = prev_state in {"active", "carry_forward"} and not hard_fail
        state = "active" if active else ("carry_forward" if can_carry else "disabled")
        side_state[strat] = {
            "side": side,
            "state": state,
            "month": month_token,
            "sharpe_r": _sh,
            "n_trades": _nt,
            "prev_state": prev_state,
            "hard_fail": hard_fail,
            "quality_score": float(q),
        }

        results_summary.append(
            {
                "strategy": strat,
                "decision": "FAST_MONTH",
                "evidence_dir": str(strat_dir),
                "run_dir_name": str(run_root.name),
                "exp_config_dir": str(month_strategies_root / strat),
            }
        )

    ranking_rows = sorted(
        ranking_rows,
        key=lambda r: (
            -float(r.get("quality_score", 0.0) or 0.0),
            float((r.get("metrics", {}) or {}).get("near_stop_rate", 1.0) or 1.0),
            float((r.get("metrics", {}) or {}).get("max_drawdown_r", 1e9) or 1e9),
            -int((r.get("metrics", {}) or {}).get("n_trades", 0) or 0),
            str(r.get("strategy", "")),
        ),
    )

    # Slot allocation: cap selected strategies per side by quality ranking.
    side_selected: Dict[str, int] = {"long": 0, "short": 0}
    selected_strategies: List[str] = []
    for row in ranking_rows:
        st = str(row.get("strategy", ""))
        side = _resolve_strategy_side(st, cfg.get("strategies", {}).get(st, {}) or {})
        if side not in {"long", "short"}:
            row["selected_for_slot"] = True
            selected_strategies.append(st)
            continue
        if side_selected[side] >= max_symbols_per_side:
            row["selected_for_slot"] = False
            continue
        row["selected_for_slot"] = True
        side_selected[side] += 1
        selected_strategies.append(st)

    quality_path = run_root / f"quality_ranking_{month_token}.json"
    side_path = run_root / "symbol_side_state.json"
    quality_path.write_text(
        json.dumps(
            {"month": month_token, "rankings": ranking_rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    side_path.write_text(
        json.dumps(
            {"month": month_token, "states": side_state},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pcm_result = None
    selected_results_summary = [
        r
        for r in results_summary
        if str(r.get("strategy", "")) in set(selected_strategies)
        and not _is_multi_leg_strategy(cfg, str(r.get("strategy", "")))
    ]
    if pcm_eval_enabled and len(selected_results_summary) >= 2:
        pcm_result = pipeline_events.run_pcm_joint_backtest(
            selected_results_summary,
            history_dir,
            timestamp,
            dry_run=dry_run,
            use_1min=use_1min,
            live_root=live_root,
            data_path=data_path,
            holdout_start=test_start,
            end_date=test_end,
            output_stem=f"pcm_fast_month_{month_token}",
            step_name=f"PCM Joint Event Backtest (fast_month {month_token})",
        )
    elif not pcm_eval_enabled:
        print("   ⏭️  跳过 PCM 联合回测: fast_loop.pcm_eval.enabled=false")

    summary = {
        "month": month_token,
        "month_start": month_start,
        "month_end": month_end,
        "rolling_mode": rolling_mode,
        "calibration_window": {"start": calib_start, "end": calib_end},
        "test_window": {"start": test_start, "end": test_end},
        "strategies_root": str(month_strategies_root),
        "run_root": str(run_root),
        "quality_ranking_path": str(quality_path),
        "side_state_path": str(side_path),
        "selected_strategies": selected_strategies,
        "end_state_paths": end_state_paths,
        "pcm": pcm_result or {},
    }
    (run_root / "fast_month_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _run_slow_structure_snapshot_for_month(
    *,
    cfg: Dict[str, Any],
    strategies: List[str],
    history_dir: Path,
    timestamp: str,
    month_token: str,
    data_path: str,
    dry_run: bool,
    use_1min: bool,
    live_root: str,
    lookback_months: int,
    source_strategies_root: str,
    config_path: str,
) -> Dict[str, Any]:
    """Build slow structure snapshot as-of previous month end."""
    prev_end = _month_prev_end(month_token)
    struct_end = prev_end
    struct_start = _add_months(f"{month_token}-01", -int(lookback_months))
    snap_root = (
        history_dir / "_rolling_sim" / timestamp / f"slow_snapshot_{month_token}"
    )
    snap_strategies_root = snap_root / "strategies"
    snap_root.mkdir(parents=True, exist_ok=True)
    snap_strategies_root.mkdir(parents=True, exist_ok=True)

    for strategy in strategies:
        if strategy not in cfg.get("strategies", {}):
            continue
        strat_dir = snap_root / strategy
        strat_dir.mkdir(parents=True, exist_ok=True)
        if _is_multi_leg_strategy(cfg, strategy):
            base_root = Path(source_strategies_root)
            if not base_root.is_absolute():
                base_root = PROJECT_ROOT / base_root
            _run_multileg_month_strategy(
                cfg=cfg,
                strategy=strategy,
                run_root=snap_root,
                month_strategies_root=snap_strategies_root,
                base_strategies_root=base_root,
                data_path=data_path,
                symbols=resolve_symbols_from_config(cfg),
                calib_start=struct_start,
                calib_end=struct_end,
                test_start=struct_start,
                test_end=struct_end,
                dry_run=dry_run,
                calibrate=True,
            )
            continue
        res = pipeline_strategy.run_strategy_pipeline(
            strategy,
            cfg,
            end_date=struct_end,
            holdout_start=compute_holdout_start(struct_end, 3),
            holdout_months=3,
            validation_months=0,
            start_date=struct_start,
            symbols=resolve_symbols_from_config(cfg),
            data_path=data_path,
            run_dir=strat_dir,
            seed=42,
            dry_run=dry_run,
            use_1min=use_1min,
            live_root=live_root,
            skip_shap=False,
            feature_search_enabled=True,
            threshold_calibration_enabled=False,
            source_strategies_root=source_strategies_root,
            config_path=config_path,
            stage_stop="entry_filter",
        )
        exp_cfg_dir = Path(str(res.get("exp_config_dir", "") or ""))
        if exp_cfg_dir.exists():
            dst = snap_strategies_root / strategy
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(exp_cfg_dir, dst, dirs_exist_ok=True)

    manifest = {
        "run_id": timestamp,
        "stage": "slow_snapshot",
        "mode": "slow_realistic",
        "target_month": month_token,
        "structure_start": struct_start,
        "structure_end": struct_end,
        "lookback_months": int(lookback_months),
        "strategies": strategies,
        "strategies_root": str(snap_strategies_root),
        "created_at": datetime.now().isoformat(),
    }
    manifest_path = snap_root / "slow_snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "snapshot_root": str(snap_root),
        "strategies_root": str(snap_strategies_root),
        "manifest_path": str(manifest_path),
        "structure_start": struct_start,
        "structure_end": struct_end,
    }


# ====================================================================
# Main
# ====================================================================


def main():
    p = argparse.ArgumentParser(description="自动研究流水线 (实验隔离版)")
    p.add_argument("--strategy", help="策略名 (bpc/fer/me-long)")
    p.add_argument("--all", action="store_true", help="执行所有策略")
    p.add_argument("--end-date", help="数据截止日期 (默认自动检测)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline 配置文件")
    p.add_argument("--compare-only", action="store_true", help="只对比, 不重训")
    p.add_argument("--dry-run", action="store_true", help="打印命令但不执行")
    p.add_argument(
        "--no-adopt", action="store_true", help="禁止自动采纳, 仅保存实验结果"
    )
    p.add_argument(
        "--list",
        dest="list_experiments",
        action="store_true",
        help="列出历史实验及其 metrics",
    )
    p.add_argument(
        "--adopt",
        metavar="TIMESTAMP",
        help="手动采纳指定时间戳的实验 (如 20260222_120000)",
    )
    p.add_argument(
        "--diff",
        nargs=2,
        metavar="TS",
        help="对比两次实验的 archetypes 差异 (如 --diff TS1 TS2)",
    )
    p.add_argument(
        "--skip-shap",
        action="store_true",
        help="跳过 SHAP 特征筛选 (快速迭代用)",
    )
    p.add_argument(
        "--use-1min",
        action="store_true",
        help="使用 1min bar 精细模拟止损/移动止损 (匹配实盘精度)",
    )
    p.add_argument(
        "--live-root",
        default="live/highcap",
        help="1min bar 数据根目录 (default: live/highcap)",
    )
    p.add_argument(
        "--event-backtest",
        action="store_true",
        help="训练后蒹加事件回测 execution 优化步骤 (sym-r 1.0:0.5:4.0 + 交易地图)",
    )
    p.add_argument(
        "--event-sym-r",
        default="1.0:0.5:4.0",
        help="事件回测 execution 优化的 sym-r 范围 (default: 1.0:0.5:4.0)",
    )
    p.add_argument(
        "--locked-prefilter-override",
        default="",
        help="可选: 指定 prefilter.yaml 作为 locked 规则来源 (调参工具专用)",
    )
    p.add_argument(
        "--disable-auto-locked-tuning",
        action="store_true",
        help="禁用自动 locked 阈值调优",
    )
    p.add_argument(
        "--stage",
        choices=[
            "full",
            "prefilter",
            "gate",
            "entry_filter",
            "slow_snapshot",
            "execution_opt",
            "event_backtest",
            "fast_month",
            "rolling_sim",
            "grid_backtest",
            "dual_add_backtest",
            "pcm_joint",
            "pcm_slot_grid",
        ],
        default="full",
        help=(
            "运行阶段: full/prefilter/gate/entry_filter/slow_snapshot/"
            "execution_opt/event_backtest/fast_month/rolling_sim/grid_backtest/"
            "dual_add_backtest/pcm_joint/pcm_slot_grid"
        ),
    )
    p.add_argument(
        "--month",
        default="",
        help=(
            "月份窗口 (YYYY-MM), 供 --stage fast_month 使用；"
            "多个月可用逗号或空格分隔, 例如 2024-07,2024-08,2024-09"
        ),
    )
    args = p.parse_args()

    cfg = load_pipeline_config(Path(args.config))
    history_dir = PROJECT_ROOT / cfg["output"]["history_dir"]

    # ── 子命令: 列出历史实验 ──
    if args.list_experiments:
        if not args.strategy and not args.all:
            p.error("--list 需要指定 --strategy 或 --all")
        strats = list(cfg["strategies"].keys()) if args.all else [args.strategy]
        for s in strats:
            _cmd_list_experiments(history_dir, s)
        return

    # ── 子命令: 手动采纳实验 ──
    if args.adopt:
        if not args.strategy:
            p.error("--adopt 需要指定 --strategy")
        _cmd_adopt_experiment(history_dir, cfg, args.strategy, args.adopt)
        return

    # ── 子命令: 对比两次实验 ──
    if args.diff:
        if not args.strategy:
            p.error("--diff 需要指定 --strategy")
        _cmd_diff_experiments(history_dir, args.strategy, args.diff[0], args.diff[1])
        return

    if not args.strategy and not args.all:
        p.error("必须指定 --strategy 或 --all")

    dates = cfg["dates"]
    symbols = resolve_symbols_from_config(cfg)
    data_path = cfg["data_path"]

    # ── 自动检测日期 ──
    if args.end_date:
        default_end_date = args.end_date
    elif str(dates.get("end_date", "") or "").strip():
        default_end_date = str(dates["end_date"]).strip()
    else:
        default_end_date = detect_latest_data_date(data_path, symbols)

    # Display baseline uses global dates (actual run may be overridden per strategy)
    _display_start = dates["start_date"]
    _display_end = default_end_date
    _display_holdout_months = int(dates["holdout_months"])
    _display_validation_months = int(dates.get("validation_months", 0))
    _display_holdout_start = compute_holdout_start(
        _display_end, _display_holdout_months
    )
    if 0 < _display_validation_months < _display_holdout_months:
        _test_start_display = compute_holdout_start(
            _display_end, _display_holdout_months - _display_validation_months
        )
    else:
        _test_start_display = _display_holdout_start

    print("=" * 70)
    print("🚀 自动研究流水线")
    print("=" * 70)
    print(f"   数据范围:    {_display_start} ~ {_display_end}")
    print(f"   Train:       {_display_start} ~ {_display_holdout_start}")
    if _test_start_display != _display_holdout_start:
        print(
            f"   Val:         {_display_holdout_start} ~ {_test_start_display} ({_display_validation_months} 个月, Gate 调阈值)"
        )
        print(
            f"   Test:        {_test_start_display} ~ {_display_end} ({_display_holdout_months - _display_validation_months} 个月, 纯 OOS)"
        )
    else:
        print(
            f"   Holdout:     {_display_holdout_start} ~ {_display_end} ({dates['holdout_months']} 个月)"
        )
    print(f"   Symbols:     {symbols}")
    print(f"   History:     {history_dir}")
    # ── 多 Seed 配置 ──
    training_cfg = cfg.get("training", {})
    seeds = training_cfg.get("seeds", [42])
    seed_selection = training_cfg.get("seed_selection", "best_sharpe")
    print(f"   Seeds:       {seeds}")
    if args.dry_run:
        print("   Mode:        DRY RUN")
    print("=" * 70)

    # ── 策略方向范围开关 ──
    strategy_direction_scope = _resolve_strategy_direction_scope(cfg)

    # ── 确定策略列表 ──
    if args.all:
        strategies = _filter_strategies_by_direction_scope(
            list(cfg["strategies"].keys()), strategy_direction_scope, cfg
        )
    else:
        strategies = [args.strategy]
    if args.all:
        print(f"   StrategyScope: direction={strategy_direction_scope}")
    if not strategies:
        p.error(
            f"当前 --all 在 strategy_scope.direction={strategy_direction_scope} 下无可运行策略"
        )
    _run_config_consistency_checks(
        cfg=cfg,
        strategies=strategies,
        stage=str(args.stage or "full"),
        dry_run=bool(args.dry_run),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_summary = []

    if args.stage == "grid_backtest":
        grid_start = str(
            (cfg.get("grid_backtest", {}) or {}).get("start_date")
            or dates["start_date"]
        )
        grid_end = str(
            (cfg.get("grid_backtest", {}) or {}).get("end_date") or default_end_date
        )
        summaries = _run_grid_backtest_stage(
            cfg=cfg,
            strategies=strategies,
            history_dir=history_dir,
            timestamp=timestamp,
            dry_run=args.dry_run,
            data_path=cfg["data_path"],
            symbols=symbols,
            start_date=grid_start,
            end_date=grid_end,
        )
        print(f"\n{'='*70}")
        print("📋 Grid Backtest 汇总")
        print(f"{'='*70}")
        for row in summaries:
            print(
                f"   • {row.get('strategy')}: {row.get('report') or row.get('out_dir')}"
            )
        return

    if args.stage == "dual_add_backtest":
        dual_start = str(
            (cfg.get("dual_add_backtest", {}) or {}).get("start_date")
            or dates["start_date"]
        )
        dual_end = str(
            (cfg.get("dual_add_backtest", {}) or {}).get("end_date") or default_end_date
        )
        summaries = _run_dual_add_backtest_stage(
            cfg=cfg,
            strategies=strategies,
            history_dir=history_dir,
            timestamp=timestamp,
            dry_run=args.dry_run,
            data_path=cfg["data_path"],
            symbols=symbols,
            start_date=dual_start,
            end_date=dual_end,
        )
        print(f"\n{'='*70}")
        print("📋 Dual Add Backtest 汇总")
        print(f"{'='*70}")
        for row in summaries:
            print(f"   • {row.get('strategy')}: {row.get('out_dir')}")
            summary = row.get("summary") or {}
            if summary:
                print(
                    "     "
                    f"net={summary.get('sum_pnl_per_capital')} "
                    f"win={summary.get('segment_win_rate')} "
                    f"worst={summary.get('worst_segment')}"
                )
        return

    if args.stage == "fast_month":
        if not args.month:
            p.error(
                "--stage fast_month 需要 --month YYYY-MM "
                "(多个月: 逗号或空格分隔, 如 2024-07,2024-08)"
            )
        month_tokens = _split_month_list(str(args.month))
        if not month_tokens:
            p.error("--month 解析后为空")
        rolling_cfg = cfg.get("rolling", {}) or {}
        rolling_mode = str(rolling_cfg.get("mode", "legacy"))
        calibration_months = int(
            ((rolling_cfg.get("windows", {}) or {}).get("calibration_months", 3) or 3)
        )
        if rolling_mode == "turbo_fixed_features":
            stage_root = str(
                (rolling_cfg.get("turbo_fixed_features", {}) or {}).get(
                    "fixed_strategies_root", "config/strategies"
                )
            )
            stage_feature_search = False
        else:
            stage_root = "config/strategies"
            stage_feature_search = True
        side_state_cursor: Dict[str, Any] = {}
        resume_state_cursor: Dict[str, str] = {}
        summaries: List[Dict[str, Any]] = []
        for month_idx, mt in enumerate(month_tokens):
            _is_last_month = month_idx == len(month_tokens) - 1
            try:
                _summary = _run_fast_month_stage(
                    cfg=cfg,
                    strategies=strategies,
                    history_dir=history_dir,
                    timestamp=timestamp,
                    month_token=mt,
                    dry_run=args.dry_run,
                    use_1min=args.use_1min,
                    live_root=args.live_root,
                    data_path=cfg["data_path"],
                    event_sym_r=args.event_sym_r,
                    strategies_root=stage_root,
                    calibration_months=calibration_months,
                    calibrate_all_layers=(rolling_mode != "legacy"),
                    feature_search_enabled=stage_feature_search,
                    rolling_mode=rolling_mode,
                    config_path=args.config,
                    prev_side_state=side_state_cursor,
                    prev_resume_state_paths=resume_state_cursor,
                    keep_open_positions=not _is_last_month,
                    month_index=month_idx,
                )
            except ValueError as exc:
                p.error(str(exc))
            summaries.append(_summary)
            try:
                _ssp = str(_summary.get("side_state_path", "") or "")
                if _ssp:
                    side_obj = json.loads(Path(_ssp).read_text(encoding="utf-8"))
                    side_state_cursor = dict(side_obj.get("states", {}) or {})
            except Exception:
                pass
            resume_state_cursor = dict(_summary.get("end_state_paths", {}) or {})
        print(f"\n{'='*70}")
        print("📋 Stage 汇总")
        print(f"{'='*70}")
        print(f"   Stage: fast_month | months={month_tokens}")
        for _s in summaries:
            print(
                f"   • {_s.get('month')}: {_s.get('month_start')}~{_s.get('month_end')} "
                f"→ {_s.get('run_root')}"
            )
        if summaries:
            _last = summaries[-1]
            print(f"   (最后一月) SideState: {_last.get('side_state_path')}")
            print(f"   (最后一月) Quality: {_last.get('quality_ranking_path')}")
        return

    if args.stage == "rolling_sim":
        rolling_cfg = cfg.get("rolling", {}) or {}
        rolling_mode = str(rolling_cfg.get("mode", "legacy"))
        windows_cfg = rolling_cfg.get("windows", {}) or {}
        slow_cfg = rolling_cfg.get("slow_realistic", {}) or {}
        turbo_cfg = rolling_cfg.get("turbo_fixed_features", {}) or {}
        calibration_months = int(windows_cfg.get("calibration_months", 3) or 3)
        structure_lookback_months = int(
            windows_cfg.get("structure_lookback_months", 12) or 12
        )
        cadence_months = int(slow_cfg.get("cadence_months", 3) or 3)
        triggered_retrain_enabled = bool(
            slow_cfg.get("triggered_retrain_enabled", True)
        )
        turbo_root = str(turbo_cfg.get("fixed_strategies_root", "config/strategies"))
        turbo_feature_search = not bool(turbo_cfg.get("disable_feature_search", True))

        month_tokens = _iter_month_tokens(_display_holdout_start, _display_end)
        if not month_tokens:
            p.error("rolling_sim 无可用月份窗口")
        roll_root = history_dir / "_rolling_sim" / timestamp
        roll_root.mkdir(parents=True, exist_ok=True)
        ledger: List[Dict[str, Any]] = []
        side_state_cursor: Dict[str, Any] = {}
        resume_state_cursor: Dict[str, str] = {}
        active_strategies_root = str(PROJECT_ROOT / "config" / "strategies")
        active_slow_manifest = ""
        for month_idx, mt in enumerate(month_tokens):
            if rolling_mode == "slow_realistic":
                if (
                    triggered_retrain_enabled
                    and month_idx % max(cadence_months, 1) == 0
                ):
                    slow_snapshot = _run_slow_structure_snapshot_for_month(
                        cfg=cfg,
                        strategies=strategies,
                        history_dir=history_dir,
                        timestamp=timestamp,
                        month_token=mt,
                        data_path=cfg["data_path"],
                        dry_run=args.dry_run,
                        use_1min=args.use_1min,
                        live_root=args.live_root,
                        lookback_months=structure_lookback_months,
                        source_strategies_root=str(
                            PROJECT_ROOT / "config" / "strategies"
                        ),
                        config_path=args.config,
                    )
                    active_strategies_root = str(
                        slow_snapshot.get("strategies_root", "")
                    )
                    active_slow_manifest = str(slow_snapshot.get("manifest_path", ""))
            elif rolling_mode == "turbo_fixed_features":
                active_strategies_root = turbo_root
            else:
                active_strategies_root = str(PROJECT_ROOT / "config" / "strategies")

            _is_last_month = mt == month_tokens[-1]
            _summary = _run_fast_month_stage(
                cfg=cfg,
                strategies=strategies,
                history_dir=history_dir,
                timestamp=timestamp,
                month_token=mt,
                dry_run=args.dry_run,
                use_1min=args.use_1min,
                live_root=args.live_root,
                data_path=cfg["data_path"],
                event_sym_r=args.event_sym_r,
                strategies_root=active_strategies_root,
                calibration_months=calibration_months,
                calibrate_all_layers=(rolling_mode != "legacy"),
                feature_search_enabled=(rolling_mode == "slow_realistic")
                or turbo_feature_search,
                rolling_mode=rolling_mode,
                config_path=args.config,
                prev_side_state=side_state_cursor,
                prev_resume_state_paths=resume_state_cursor,
                keep_open_positions=not _is_last_month,
                month_index=month_idx,
            )
            _summary["slow_manifest"] = active_slow_manifest
            _summary["active_strategies_root"] = active_strategies_root
            ledger.append(_summary)
            try:
                side_obj = json.loads(
                    Path(_summary.get("side_state_path", "")).read_text(
                        encoding="utf-8"
                    )
                )
                side_state_cursor = dict(side_obj.get("states", {}) or {})
            except Exception:
                pass
            resume_state_cursor = dict(_summary.get("end_state_paths", {}) or {})
        ledger_path = roll_root / "monthly_ledger.jsonl"
        with ledger_path.open("w", encoding="utf-8") as f:
            for row in ledger:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pcm_total_r = float(
            sum(
                float(((r.get("pcm") or {}).get("total_r", 0.0) or 0.0)) for r in ledger
            )
        )
        pcm_trades = int(
            sum(int(((r.get("pcm") or {}).get("total_trades", 0) or 0)) for r in ledger)
        )
        if pcm_trades == 0:
            _fb_r, _fb_n = _ledger_stitched_pcm_fallback_totals(ledger)
            if _fb_n > 0:
                pcm_total_r = _fb_r
                pcm_trades = _fb_n
        multi_leg_metrics = _collect_multileg_stitched_metrics(
            cfg=cfg,
            ledger=ledger,
            strategies=strategies,
        )
        if pcm_trades == 0 and int(multi_leg_metrics.get("total_trades", 0) or 0) > 0:
            pcm_total_r = float(multi_leg_metrics.get("total_r", 0.0) or 0.0)
            pcm_trades = int(multi_leg_metrics.get("total_trades", 0) or 0)
        month_maps = [
            str((r.get("pcm") or {}).get("trading_map", "") or "") for r in ledger
        ]
        _map_sroot = PROJECT_ROOT / "config" / "strategies"
        _roll_cfg = cfg.get("rolling") or {}
        if str(_roll_cfg.get("mode", "") or "") == "turbo_fixed_features":
            _fs = (_roll_cfg.get("turbo_fixed_features") or {}).get(
                "fixed_strategies_root"
            )
            if _fs:
                _pfs = Path(str(_fs))
                _map_sroot = _pfs if _pfs.is_absolute() else PROJECT_ROOT / _pfs
        _band_pair = _read_bpc_vwap_band_abs(_map_sroot)
        _bi, _bo = _band_pair if _band_pair else (0.005, 0.05)
        _dates_for_map = cfg.get("dates") or {}
        continuous_map = _build_continuous_pcm_trading_map(
            ledger,
            roll_root / "trading_map_continuous.html",
            data_path=cfg["data_path"],
            band_inner_abs=_bi,
            band_outer_abs=_bo,
            chart_x_start=str(_dates_for_map.get("start_date") or "").strip() or None,
            chart_x_end=str(_dates_for_map.get("end_date") or "").strip() or None,
        )
        multi_leg_continuous_map = ""
        if any(_is_multi_leg_strategy(cfg, _st) for _st in strategies):
            _multi_leg_map_path = (
                roll_root / "trading_map_continuous.html"
                if all(_is_multi_leg_strategy(cfg, _st) for _st in strategies)
                else roll_root / "trading_map_continuous_multi_leg.html"
            )
            multi_leg_continuous_map = _build_multileg_rolling_continuous_map(
                cfg=cfg,
                ledger=ledger,
                strategies=strategies,
                roll_root=roll_root,
                data_path=cfg["data_path"],
                output_path=_multi_leg_map_path,
            )
            if multi_leg_continuous_map and not continuous_map:
                continuous_map = multi_leg_continuous_map
        stitch_map = roll_root / "trading_map_stitched.html"
        stitch_lines = [
            "<html><head><meta charset='utf-8'><title>Stitched Trading Maps</title></head><body>",
            "<h2>Rolling Sim Trading Maps</h2>",
            (
                f"<p><a href='{continuous_map}'>continuous_map</a></p>"
                if continuous_map
                else ""
            ),
            "<ul>",
        ]
        for r in ledger:
            m = str(r.get("month", ""))
            mp = str((r.get("pcm") or {}).get("trading_map", "") or "")
            if mp:
                stitch_lines.append(f"<li>{m}: <a href='{mp}'>{mp}</a></li>")
        stitch_lines.extend(["</ul>", "</body></html>"])
        stitch_map.write_text("\n".join(stitch_lines), encoding="utf-8")
        stitched = {
            "run_id": timestamp,
            "mode": rolling_mode,
            "months": [r.get("month") for r in ledger],
            "count_months": len(ledger),
            "ledger_path": str(ledger_path),
            "stitched_total_r": pcm_total_r,
            "stitched_total_trades": pcm_trades,
            "stitched_max_drawdown_r": float(
                multi_leg_metrics.get("max_drawdown_r", 0.0) or 0.0
            ),
            "multi_leg_metrics": multi_leg_metrics,
            "stitched_map_index": str(stitch_map),
            "month_pcm_maps": month_maps,
            "continuous_map": continuous_map,
            "multi_leg_continuous_map": multi_leg_continuous_map,
            "capital_report": (
                str(roll_root / "capital_report.json")
                if (roll_root / "capital_report.json").exists()
                else ""
            ),
        }
        stitched_path = roll_root / "stitched_summary.json"
        stitched_path.write_text(
            json.dumps(stitched, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # ── Auto threshold-drift report (per strategy) ──
        if args.dry_run:
            print("   ⏭️  ThresholdDrift: skipped (dry-run)")
        else:
            print("\n📈 ThresholdDrift 报告生成...")
            for _st in strategies:
                _cmd = [
                    sys.executable,
                    "scripts/plot_monthly_threshold_drift.py",
                    "--run-root",
                    str(roll_root),
                    "--strategy",
                    str(_st),
                ]
                try:
                    _p = subprocess.run(
                        _cmd,
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True,
                    )
                    if _p.returncode == 0:
                        _lines = [
                            ln.strip()
                            for ln in str(_p.stdout or "").splitlines()
                            if ln.strip()
                        ]
                        _csv = next(
                            (
                                ln.replace("csv=", "")
                                for ln in _lines
                                if ln.startswith("csv=")
                            ),
                            "",
                        )
                        _html = next(
                            (
                                ln.replace("html=", "")
                                for ln in _lines
                                if ln.startswith("html=")
                            ),
                            "",
                        )
                        print(f"   ✅ {_st}: csv={_csv}")
                        print(f"      html={_html}")
                    else:
                        _err = (str(_p.stderr or "") or str(_p.stdout or "")).strip()
                        _tail = _err[-400:] if _err else "unknown error"
                        print(f"   ⚠️  {_st}: ThresholdDrift 失败: {_tail}")
                except Exception as _exc:
                    print(f"   ⚠️  {_st}: ThresholdDrift 异常: {_exc}")
        print(f"\n{'='*70}")
        print("📋 Stage 汇总")
        print(f"{'='*70}")
        print(
            f"   Stage: rolling_sim | months={len(ledger)} "
            f"({_display_holdout_start}~{_display_end})"
        )
        print(f"   Ledger: {ledger_path}")
        print(f"   Summary: {stitched_path}")
        print(f"   StitchedMap: {stitch_map}")
        if continuous_map:
            print(f"   ContinuousMap: {continuous_map}")
        return

    # ── Stage: 仅跑 PCM slot 网格 ──
    if args.stage == "pcm_slot_grid":
        if not args.all:
            p.error("--stage pcm_slot_grid 需要 --all (使用多策略 PCM 组合)")
        slot_cfg = cfg.get("pcm_slot_grid", {}) or {}
        if not bool(slot_cfg.get("enabled", False)):
            p.error("配置未启用 pcm_slot_grid.enabled=true")

        scoped_cfg_strategies = _filter_strategies_by_direction_scope(
            list(cfg["strategies"].keys()), strategy_direction_scope
        )
        enabled_by_constitution = _load_pcm_enabled_strategies_from_constitution()
        enabled_set = (
            set(enabled_by_constitution)
            if enabled_by_constitution
            else set(scoped_cfg_strategies)
        )
        pcm_strategies = [s for s in scoped_cfg_strategies if s in enabled_set]
        if len(pcm_strategies) < 2:
            p.error("可用于 PCM 的策略数 < 2，无法执行 slot 网格")

        for s in pcm_strategies:
            results_summary.append(
                {
                    "strategy": s,
                    "decision": "STAGE_ONLY",
                    "evidence_dir": str(history_dir / s / timestamp / "results"),
                    "run_dir_name": timestamp,
                    "exp_config_dir": str(PROJECT_ROOT / "config" / "strategies" / s),
                }
            )

        slot_grid_result = _run_pcm_slot_grid_backtest(
            cfg_slot=slot_cfg,
            results_summary=results_summary,
            history_dir=history_dir,
            timestamp=timestamp,
            dry_run=args.dry_run,
            use_1min=args.use_1min,
            live_root=args.live_root,
            data_path=cfg["data_path"],
            holdout_start=_display_holdout_start,
            end_date=_display_end,
        )
        print(f"\n{'='*70}")
        print("📋 Stage 汇总")
        print(f"{'='*70}")
        print(f"   Stage: pcm_slot_grid")
        print(
            f"   推荐 Case: {slot_grid_result.get('recommended_case') if slot_grid_result else 'N/A'}"
        )
        if slot_grid_result and slot_grid_result.get("report_json"):
            print(f"   报告: {slot_grid_result.get('report_json')}")
        return

    if args.stage == "pcm_joint":
        if not args.all:
            p.error("--stage pcm_joint 需要 --all (使用多策略 PCM 组合)")
        scoped_cfg_strategies = _filter_strategies_by_direction_scope(
            list(cfg["strategies"].keys()), strategy_direction_scope
        )
        enabled_by_constitution = _load_pcm_enabled_strategies_from_constitution()
        enabled_set = (
            set(enabled_by_constitution)
            if enabled_by_constitution
            else set(scoped_cfg_strategies)
        )
        pcm_strategies = [s for s in scoped_cfg_strategies if s in enabled_set]
        if len(pcm_strategies) < 2:
            p.error("可用于 PCM 的策略数 < 2，无法执行联合回测")

        for s in pcm_strategies:
            results_summary.append(
                {
                    "strategy": s,
                    "decision": "STAGE_ONLY",
                    "evidence_dir": str(history_dir / s / timestamp / "results"),
                    "run_dir_name": timestamp,
                    "exp_config_dir": str(PROJECT_ROOT / "config" / "strategies" / s),
                }
            )
        pcm_result = pipeline_events.run_pcm_joint_backtest(
            results_summary,
            history_dir,
            timestamp,
            dry_run=args.dry_run,
            use_1min=args.use_1min,
            live_root=args.live_root,
            data_path=cfg["data_path"],
            holdout_start=_display_holdout_start,
            end_date=_display_end,
            output_stem="pcm_event_backtest_stage",
            step_name="PCM Joint Event Backtest (stage)",
        )
        print(f"\n{'='*70}")
        print("📋 Stage 汇总")
        print(f"{'='*70}")
        if pcm_result:
            print(
                f"   Stage: pcm_joint | sharpe={pcm_result.get('sharpe_daily')} "
                f"trades={pcm_result.get('total_trades')} total_r={pcm_result.get('total_r')}"
            )
            if pcm_result.get("json_path"):
                print(f"   JSON: {pcm_result.get('json_path')}")
            if pcm_result.get("trades_csv_path"):
                print(f"   Trades CSV: {pcm_result.get('trades_csv_path')}")
            if pcm_result.get("log_path"):
                print(f"   Log: {pcm_result.get('log_path')}")
            if pcm_result.get("trading_map"):
                print(f"   Trading Map: {pcm_result.get('trading_map')}")
        return

    if args.stage in ("execution_opt", "event_backtest"):
        event_cfg = cfg.get("event_backtest", {}) or {}
        event_promote = bool(event_cfg.get("promote", True))
        print(f"\n{'='*70}")
        print(f"🎓 Stage: {args.stage}")
        print(f"{'='*70}")
        for strat in strategies:
            if strat not in cfg["strategies"]:
                print(f"   ❌ 未知策略: {strat}, 跳过")
                continue
            strat_dates = resolve_strategy_dates(
                cfg,
                strat,
                default_end_date=default_end_date,
                forced_end_date=args.end_date or "",
            )
            event_sym_r = _resolve_event_sym_r_for_strategy(
                cfg, strat, args.event_sym_r
            )
            _exec_grid = _resolve_event_exec_grid_for_strategy(cfg, strat)
            obj_cfg = _resolve_event_exec_objective_for_strategy(cfg, strat)
            stage_run_dir = history_dir / strat / timestamp
            stage_run_dir.mkdir(parents=True, exist_ok=True)
            stage_strategies_root = str(PROJECT_ROOT / "config" / "strategies")
            _grid_label = f"grid={_exec_grid}" if _exec_grid else f"sym-r={event_sym_r}"
            print(f"\n   ▶️  {strat}: {_grid_label}, objective={obj_cfg['objective']}")
            if args.stage == "execution_opt":
                _res = pipeline_events.run_event_execution_opt_only(
                    strat,
                    stage_run_dir,
                    holdout_start=strat_dates["holdout_start"],
                    end_date=strat_dates["end_date"],
                    strategies_root=stage_strategies_root,
                    data_path=data_path,
                    dry_run=args.dry_run,
                    sym_r=event_sym_r,
                    exec_grid=_exec_grid,
                    promote=event_promote,
                    objective=str(obj_cfg["objective"]),
                    near_stop_threshold_r=float(obj_cfg["near_stop_threshold_r"]),
                    near_stop_penalty=float(obj_cfg["near_stop_penalty"]),
                    max_dd_penalty=float(obj_cfg["max_dd_penalty"]),
                    min_trades_soft=int(obj_cfg["min_trades_soft"]),
                    undertrade_penalty=float(obj_cfg["undertrade_penalty"]),
                )
                print(f"      rc={_res.get('rc')} output={_res.get('output')}")
            else:
                _ev = pipeline_events.run_event_backtest_step(
                    strat,
                    str(stage_run_dir),
                    stage_run_dir,
                    holdout_start=strat_dates["holdout_start"],
                    end_date=strat_dates["end_date"],
                    strategies_root=stage_strategies_root,
                    data_path=data_path,
                    dry_run=args.dry_run,
                    sym_r=event_sym_r,
                    exec_grid=_exec_grid,
                    promote=event_promote,
                    objective=str(obj_cfg["objective"]),
                    near_stop_threshold_r=float(obj_cfg["near_stop_threshold_r"]),
                    near_stop_penalty=float(obj_cfg["near_stop_penalty"]),
                    max_dd_penalty=float(obj_cfg["max_dd_penalty"]),
                    min_trades_soft=int(obj_cfg["min_trades_soft"]),
                    undertrade_penalty=float(obj_cfg["undertrade_penalty"]),
                    map_extra_months=_event_trading_map_extra_months(cfg),
                    no_kill_switch=bool(
                        (cfg.get("event_backtest") or {}).get("no_kill_switch", False)
                    ),
                )
                _m = _ev.get("metrics", {})
                print(
                    f"      sharpe={_m.get('sharpe_r', 'N/A')} "
                    f"trades={_m.get('n_trades', 'N/A')} "
                    f"map={_ev.get('map_path', 'N/A')}"
                )
        return

    if args.stage in ("prefilter", "gate", "entry_filter", "slow_snapshot"):
        _stage_label = "entry_filter" if args.stage == "slow_snapshot" else args.stage
        print(f"\n{'='*70}")
        print(f"🎯 Stage: {args.stage} (逐策略)")
        print(f"{'='*70}")
        for strategy in strategies:
            if strategy not in cfg["strategies"]:
                print(f"\n❌ 未知策略: {strategy}, 跳过")
                continue
            if args.stage == "slow_snapshot" and _is_multi_leg_strategy(cfg, strategy):
                if not args.month:
                    p.error(
                        "--stage slow_snapshot 对 multi-leg 策略需要 --month YYYY-MM"
                    )
                rolling_cfg = cfg.get("rolling", {}) or {}
                windows_cfg = rolling_cfg.get("windows", {}) or {}
                lookback_months = int(
                    windows_cfg.get("structure_lookback_months", 12) or 12
                )
                snap = _run_slow_structure_snapshot_for_month(
                    cfg=cfg,
                    strategies=[strategy],
                    history_dir=history_dir,
                    timestamp=timestamp,
                    month_token=str(args.month).split(",")[0].strip(),
                    data_path=cfg["data_path"],
                    dry_run=args.dry_run,
                    use_1min=args.use_1min,
                    live_root=args.live_root,
                    lookback_months=lookback_months,
                    source_strategies_root=str(PROJECT_ROOT / "config" / "strategies"),
                    config_path=args.config,
                )
                print(
                    f"   ✅ {strategy}: multi-leg slow_snapshot={snap.get('manifest_path')}"
                )
                continue
            strategy_dates = resolve_strategy_dates(
                cfg,
                strategy=strategy,
                default_end_date=default_end_date,
                forced_end_date=args.end_date or "",
            )
            run_dir = history_dir / strategy / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            stage_seed = int((training_cfg.get("seeds") or [42])[0])
            result = pipeline_strategy.run_strategy_pipeline(
                strategy,
                cfg,
                end_date=strategy_dates["end_date"],
                holdout_start=strategy_dates["holdout_start"],
                holdout_months=strategy_dates["holdout_months"],
                validation_months=strategy_dates["validation_months"],
                start_date=strategy_dates["start_date"],
                symbols=symbols,
                data_path=data_path,
                run_dir=run_dir,
                seed=stage_seed,
                dry_run=args.dry_run,
                use_1min=args.use_1min,
                live_root=args.live_root,
                skip_shap=args.skip_shap,
                config_path=args.config,
                locked_prefilter_override=args.locked_prefilter_override,
                disable_auto_locked_tuning=args.disable_auto_locked_tuning,
                stage_stop=_stage_label,
            )
            if "error" in result:
                print(f"   ❌ {strategy}: {result['error']}")
            else:
                print(
                    f"   ✅ {strategy}: stage={result.get('stage', _stage_label)} "
                    f"run_dir={run_dir}"
                )
        if args.stage == "slow_snapshot":
            snap_root = history_dir / "_rolling_sim" / timestamp
            snap_root.mkdir(parents=True, exist_ok=True)
            snap_path = snap_root / "slow_snapshot_manifest.json"
            rolling_cfg = cfg.get("rolling", {}) or {}
            snap_path.write_text(
                json.dumps(
                    {
                        "run_id": timestamp,
                        "stage": "slow_snapshot",
                        "mode": str(rolling_cfg.get("mode", "legacy")),
                        "strategies": strategies,
                        "strategies_root": str(PROJECT_ROOT / "config" / "strategies"),
                        "config_path": str(args.config),
                        "created_at": datetime.now().isoformat(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"   Snapshot Manifest: {snap_path}")
        return

    for strategy in strategies:
        if strategy not in cfg["strategies"]:
            print(f"\n❌ 未知策略: {strategy}, 跳过")
            continue

        print(f"\n{'#'*70}")
        print(f"# 策略: {strategy.upper()}")
        print(f"{'#'*70}")
        strategy_dates = resolve_strategy_dates(
            cfg,
            strategy=strategy,
            default_end_date=default_end_date,
            forced_end_date=args.end_date or "",
        )
        print(
            f"   日期覆盖: start={strategy_dates['start_date']}, "
            f"holdout={strategy_dates['holdout_months']}m, "
            f"val={strategy_dates['validation_months']}m"
        )

        if args.compare_only:
            # 只对比
            prev = find_previous_report(history_dir, strategy)
            if prev:
                print(f"\n📊 上次运行: {prev.get('timestamp')}")
                print(
                    f"   Sharpe: {prev.get('backtest_metrics', {}).get('sharpe_per_trade', 'N/A')}"
                )
            else:
                print("\n📊 无历史记录")
            continue

        # ── 多 Seed 搜索 ──
        scfg = cfg["strategies"][strategy]
        exec_min_trades = (
            scfg.get("kpi_gates", {}).get("backtest", {}).get("min_trades", 0)
        )
        multi_seed = len(seeds) > 1
        seed_trials = []

        for seed_idx, seed in enumerate(seeds):
            if multi_seed:
                seed_run_dir = history_dir / strategy / f"{timestamp}_s{seed}"
                print(f"\n  🌱 Seed {seed} ({seed_idx+1}/{len(seeds)})")
            else:
                seed_run_dir = history_dir / strategy / timestamp
            seed_run_dir.mkdir(parents=True, exist_ok=True)

            result = pipeline_strategy.run_strategy_pipeline(
                strategy,
                cfg,
                end_date=strategy_dates["end_date"],
                holdout_start=strategy_dates["holdout_start"],
                holdout_months=strategy_dates["holdout_months"],
                validation_months=strategy_dates["validation_months"],
                start_date=strategy_dates["start_date"],
                symbols=symbols,
                data_path=data_path,
                run_dir=seed_run_dir,
                seed=seed,
                dry_run=args.dry_run,
                use_1min=args.use_1min,
                live_root=args.live_root,
                skip_shap=args.skip_shap,
                config_path=args.config,
                locked_prefilter_override=args.locked_prefilter_override,
                disable_auto_locked_tuning=args.disable_auto_locked_tuning,
            )

            metrics = result.get("backtest_metrics", {})
            gate_rules = (
                _extract_gate_rules(seed_run_dir, strategy) if not args.dry_run else []
            )
            seed_trials.append(
                {
                    "seed": seed,
                    "run_dir": seed_run_dir,
                    "result": result,
                    "metrics": metrics,
                    "gate_rules": gate_rules,
                }
            )

        # ── 选优 ──
        if multi_seed:
            best = _select_best_seed(
                seed_trials, min_trades=exec_min_trades, selection=seed_selection
            )
            _print_seed_diagnostics(strategy, seed_trials, best)
        else:
            best = seed_trials[0]

        run_dir = best["run_dir"]
        pipeline_result = best["result"]

        if "error" in pipeline_result:
            print(f"\n❌ Pipeline failed: {pipeline_result['error']}")
            results_summary.append(
                {
                    "strategy": strategy,
                    "decision": "ERROR",
                    "reason": pipeline_result["error"],
                }
            )
            continue

        # ── 对比决策 ──
        prev = find_previous_report(history_dir, strategy)
        comparison = compare_runs(
            {"backtest_metrics": pipeline_result["backtest_metrics"]},
            prev,
            cfg.get("comparison", {}),
        )

        # ── 保存 ──
        report_path = save_report(
            strategy,
            cfg,
            run_dir,
            pipeline_result,
            comparison,
            start_date=strategy_dates["start_date"],
            end_date=strategy_dates["end_date"],
            holdout_start=strategy_dates["holdout_start"],
            holdout_months=strategy_dates["holdout_months"],
            validation_months=strategy_dates["validation_months"],
            validation_end=pipeline_result.get(
                "validation_end", strategy_dates["holdout_start"]
            ),
        )

        # ── 打印决策 ──
        decision = comparison["decision"]
        emoji = {"ADOPT": "✅", "ALERT": "⚠️", "KEEP": "🔄", "ERROR": "❌"}.get(
            decision, "❓"
        )

        bt = pipeline_result["backtest_metrics"]

        # ── 分层汇总表 (每层规则/阈值 + 与上次对比) ──
        cur_arch = _find_arch_dir(run_dir, strategy)
        prev_arch = None
        prev_bt = {}
        if prev and not args.dry_run:
            prev_ts = prev.get("timestamp", "")
            prev_dir = history_dir / strategy / prev_ts
            prev_arch = _find_arch_dir(prev_dir, strategy)
            prev_bt = prev.get("backtest_metrics", {})
        if cur_arch:
            print_layer_summary(
                strategy,
                run_dir.name,
                cur_arch,
                bt,
                prev_arch_dir=prev_arch,
                prev_metrics=prev_bt if prev_bt else None,
            )

        print(f"\n   {emoji} 决策: {decision}")
        for r in comparison.get("reasons", []):
            print(f"      → {r}")
        print(f"   📁 Report: {report_path}")
        print(f"   📦 实验配置: {run_dir}/strategies/{strategy}/archetypes/")

        # ── 漂移报告 (当存在上次实验时自动输出) ──
        drift_levels = None
        if prev and not args.dry_run and prev_arch and cur_arch:
            drift_levels = _print_drift_report(
                strategy,
                prev.get("timestamp", ""),
                run_dir.name,
                prev_arch,
                cur_arch,
                prev_bt,
                bt,
            )

        # ── Deploy 门禁检查 ──
        deploy_cfg = cfg.get("deploy_gate", {})
        # per-strategy kpi_gates.deploy 覆盖全局默认
        deploy_kpi = cfg["strategies"][strategy].get("kpi_gates", {}).get("deploy", {})
        if deploy_kpi.get("min_trades") is not None:
            deploy_cfg = {**deploy_cfg, "min_trades": deploy_kpi["min_trades"]}
        deploy_result = check_deploy_gate(
            decision, comparison, drift_levels, deploy_cfg
        )
        deploy_ready = deploy_result["deploy_ready"]

        # 打印 deploy 状态
        if deploy_ready:
            print(f"\n   🚀 Deploy: ✅ 值得且允许 deploy")
            if deploy_result.get("require_human_confirm"):
                print(
                    f"      运行: python scripts/deploy_config_to_live.py --diff --strategy {strategy}"
                )
                print(
                    f"      确认后: python scripts/deploy_config_to_live.py --deploy --strategy {strategy}"
                )
        elif not deploy_result.get("triggered"):
            print(
                f"\n   ⏭️  Deploy: SKIP — {deploy_result.get('skip_reason', '无触发条件')}"
            )
        else:
            print(f"\n   🚫 Deploy: ❌ 有触发但安全门禁未通过")
            for b in deploy_result["blocked_by"]:
                print(f"      ❌ {b}")

        # 写入 report.json
        _patch_report_deploy(report_path, deploy_result)

        # ── 自动采纳 ──
        prod_config_dir = pipeline_result.get("prod_config_dir")
        exp_cfg_dir = pipeline_result.get("exp_config_dir")
        if (
            decision == "ADOPT"
            and not args.no_adopt
            and prod_config_dir
            and exp_cfg_dir
        ):
            _adopt_experiment_config(Path(exp_cfg_dir), prod_config_dir)
        elif decision == "ADOPT" and args.no_adopt:
            print(f"\n   ⏭️  --no-adopt: 跳过自动采纳, 可后续手动:")
            print(
                f"      python scripts/auto_research_pipeline.py --strategy {strategy} --adopt {run_dir.name}"
            )

        results_summary.append(
            {
                "strategy": strategy,
                "decision": decision,
                "sharpe": bt.get("sharpe_per_trade"),
                "trades": bt.get("total_trades"),
                "evidence_dir": pipeline_result.get("evidence_dir"),
                "run_dir_name": run_dir.name,
                "seed": best["seed"] if multi_seed else seeds[0],
                "prefilter_comparison": pipeline_result.get("prefilter_comparison"),
            }
        )

    # ── Event Backtest Execution 优化 (可选, --event-backtest 或 config 开启) ──
    event_cfg = cfg.get("event_backtest", {}) or {}
    event_enabled = bool(getattr(args, "event_backtest", False)) or bool(
        event_cfg.get("enabled", False)
    )
    event_promote = bool(event_cfg.get("promote", True))
    if event_enabled and not args.compare_only:
        print(f"\n\n{'='*70}")
        print("🎓 Event Backtest Execution 优化 (CLI/Config)")
        print(f"{'='*70}")
        for r in results_summary:
            if r.get("decision") == "ERROR" or not r.get("evidence_dir"):
                continue
            strat = r["strategy"]
            # 每个策略使用自己的日期窗配置，避免引用外层循环变量
            strat_dates = resolve_strategy_dates(
                cfg,
                strat,
                default_end_date=default_end_date,
                forced_end_date=args.end_date or "",
            )
            event_sym_r = _resolve_event_sym_r_for_strategy(
                cfg, strat, args.event_sym_r
            )
            obj_cfg = _resolve_event_exec_objective_for_strategy(cfg, strat)
            print(
                f"   🔧 {strat}: event sym-r = {event_sym_r} | objective={obj_cfg['objective']}"
            )
            rdn = r.get("run_dir_name", timestamp)
            strat_run_dir = history_dir / strat / rdn
            # 事件回测的 strategies_root = 实验子目录中的 strategies/
            ev_strategies_root = str(strat_run_dir / "strategies")
            ev_result = pipeline_events.run_event_backtest_step(
                strat,
                r["evidence_dir"],
                strat_run_dir,
                holdout_start=strat_dates["holdout_start"],
                end_date=strat_dates["end_date"],
                strategies_root=ev_strategies_root,
                data_path=data_path,
                dry_run=args.dry_run,
                sym_r=event_sym_r,
                promote=event_promote,
                objective=str(obj_cfg["objective"]),
                near_stop_threshold_r=float(obj_cfg["near_stop_threshold_r"]),
                near_stop_penalty=float(obj_cfg["near_stop_penalty"]),
                max_dd_penalty=float(obj_cfg["max_dd_penalty"]),
                min_trades_soft=int(obj_cfg["min_trades_soft"]),
                undertrade_penalty=float(obj_cfg["undertrade_penalty"]),
                map_extra_months=_event_trading_map_extra_months(cfg),
                no_kill_switch=bool(
                    (cfg.get("event_backtest") or {}).get("no_kill_switch", False)
                ),
            )
            ev_m = ev_result.get("metrics", {})
            print(
                f"\n   ✅ {strat}: 事件回测完成"
                f" sharpe_r={ev_m.get('sharpe_r', 'N/A')}"
                f" trades={ev_m.get('n_trades', 'N/A')}"
                f" win_rate={ev_m.get('win_rate', 'N/A')}"
            )
            print(f"   🗺️  交易地图: {ev_result.get('map_path', 'N/A')}")
            # 将事件回测结果写入 report.json
            rp = strat_run_dir / "report.json"
            if rp.exists():
                _patch_report_event(rp, ev_m)

    # ── Step 9.5: PCM 联合回测 (在 execution 优化之后执行) ──
    pcm_result = None
    pcm_slot_grid_result = None
    if args.all and not args.compare_only:
        pcm_result = pipeline_events.run_pcm_joint_backtest(
            results_summary,
            history_dir,
            timestamp,
            dry_run=args.dry_run,
            use_1min=args.use_1min,
            live_root=args.live_root,
            data_path=cfg["data_path"],
            holdout_start=_display_holdout_start,
            end_date=_display_end,
        )
        slot_cfg = cfg.get("pcm_slot_grid", {}) or {}
        if bool(slot_cfg.get("enabled", False)):
            pcm_slot_grid_result = _run_pcm_slot_grid_backtest(
                cfg_slot=slot_cfg,
                results_summary=results_summary,
                history_dir=history_dir,
                timestamp=timestamp,
                dry_run=args.dry_run,
                use_1min=args.use_1min,
                live_root=args.live_root,
                data_path=cfg["data_path"],
                holdout_start=_display_holdout_start,
                end_date=_display_end,
            )

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print("📋 汇总")
    print(f"{'='*70}")
    for r in results_summary:
        emoji = {"ADOPT": "✅", "ALERT": "⚠️", "KEEP": "🔄", "ERROR": "❌"}.get(
            r["decision"], "❓"
        )
        seed_str = f" seed={r['seed']}" if r.get("seed") is not None else ""
        print(
            f"   {emoji} {r['strategy']:>6s}: {r['decision']:<8s} sharpe={r.get('sharpe', 'N/A')} trades={r.get('trades', 'N/A')}{seed_str}"
        )
        # Prefilter Score 对比表
        _pfc = r.get("prefilter_comparison")
        if _pfc and _pfc.get("candidates"):
            _best_pf = _pfc["best"]
            _cands = _pfc["candidates"]
            print(f"      🔬 Prefilter 对比 (Score=Sharpe-TradePenalty):")
            _zr_set = set(_pfc.get("zero_rule_methods", []))
            for _pm in sorted(
                _cands,
                key=lambda m: (
                    -_cands[m].get("score", float("-inf"))
                    if _cands[m].get("score", float("-inf")) != float("-inf")
                    else float("inf")
                ),
            ):
                _pc = _cands[_pm]
                _score = _pc.get("score", float("-inf"))
                _score_str = f"{_score:+.4f}" if _score != float("-inf") else "  FAIL"
                _ps = _pc["sharpe"]
                _ps_str = f"{_ps:+.4f}" if _ps != float("-inf") else "  FAIL"
                _flag = " ←" if _pm == _best_pf else ""
                _note = "  (0规则=empty)" if _pm in _zr_set else ""
                print(
                    f"         {_pm:<20s} Score={_score_str}  Sharpe={_ps_str}  "
                    f"Trades={_pc['trades']:>5}  Rules={_pc['rules']}{_flag}{_note}"
                )
    if pcm_result:
        pcm_emoji = {"PASS": "✅", "ALERT": "⚠️", "ERROR": "❌"}.get(
            pcm_result.get("pcm_decision", "?"), "❓"
        )
        print(
            f"\n   {pcm_emoji}    PCM: {pcm_result.get('pcm_decision', '?'):<8s} "
            f"sharpe_daily={pcm_result.get('sharpe_daily', 'N/A')} "
            f"conflict_rate={pcm_result.get('conflict_rate', 'N/A')} "
            f"strategies={pcm_result.get('strategies_count', 0)}"
        )
        # 保存 pcm_stats.json 到每个策略的实验目录
        for r in results_summary:
            rdn = r.get("run_dir_name", timestamp)
            strat_run = history_dir / r["strategy"] / rdn
            if strat_run.exists():
                _patch_report_pcm(strat_run / "report.json", pcm_result)
                if pcm_slot_grid_result:
                    _patch_report_pcm_slot_grid(
                        strat_run / "report.json", pcm_slot_grid_result
                    )
    if pcm_slot_grid_result:
        print(
            f"\n   🧪 SlotGrid 推荐: {pcm_slot_grid_result.get('recommended_case', 'N/A')}"
        )
        print(
            f"      score={pcm_slot_grid_result.get('recommended_metrics', {}).get('score', 'N/A')} "
            f"sharpe={pcm_slot_grid_result.get('recommended_metrics', {}).get('sharpe_daily', 'N/A')} "
            f"dd={pcm_slot_grid_result.get('recommended_metrics', {}).get('max_drawdown_r', 'N/A')}"
        )
        if pcm_slot_grid_result.get("report_json"):
            print(f"      📄 {pcm_slot_grid_result.get('report_json')}")


# ====================================================================
# 实验管理子命令
# ====================================================================


def _adopt_experiment_config(exp_config_dir: Path, prod_config_dir: str) -> bool:
    """将实验 archetypes 复制回生产 config.

    语义锁定: 采纳前校验生产 prefilter.yaml 和 gate.yaml 中的 locked 规则
    是否在实验版本中存在. locked 信息直接存储在规则上 (locked: true).
    """
    import yaml as _yaml_adopt

    exp_arch = exp_config_dir / "archetypes"
    prod_arch = PROJECT_ROOT / prod_config_dir / "archetypes"

    if not exp_arch.exists():
        print(f"   ❌ 实验 archetypes 不存在: {exp_arch}")
        return False

    # ── 语义锁定: 读取生产 prefilter.yaml 中的 locked 规则, 检查实验版本是否保留 ──
    prod_prefilter = prod_arch / "prefilter.yaml"
    if prod_prefilter.exists():
        prod_pf_raw = (
            _yaml_adopt.safe_load(prod_prefilter.read_text(encoding="utf-8")) or {}
        )
        locked_rules = [r for r in prod_pf_raw.get("rules", []) if r.get("locked")]

        if locked_rules:
            exp_prefilter = exp_arch / "prefilter.yaml"
            exp_gate = exp_arch / "gate.yaml"
            exp_features = set()

            if exp_prefilter.exists():
                exp_pf_raw = (
                    _yaml_adopt.safe_load(exp_prefilter.read_text(encoding="utf-8"))
                    or {}
                )
                for r in exp_pf_raw.get("rules", []):
                    if "feature" in r:
                        exp_features.add(r["feature"])
                    for s in r.get("any_of", []):
                        if "feature" in s:
                            exp_features.add(s["feature"])

            if exp_gate.exists():
                exp_gate_raw = (
                    _yaml_adopt.safe_load(exp_gate.read_text(encoding="utf-8")) or {}
                )
                _exp_gate_rules = list(
                    exp_gate_raw.get("system_safety", []) or []
                ) + list(exp_gate_raw.get("hard_gates", []) or [])
                for r in _exp_gate_rules:
                    for feat in r.get("when", {}).keys():
                        exp_features.add(feat)

            missing_locked = []
            for lr in locked_rules:
                if "feature" in lr:
                    if lr["feature"] not in exp_features:
                        missing_locked.append(lr)
                elif "any_of" in lr:
                    sub_feats = {s["feature"] for s in lr["any_of"] if "feature" in s}
                    if not sub_feats & exp_features:
                        missing_locked.append(lr)

            if missing_locked:
                print(
                    f"   ⛔ 语义锁定拒绝采纳: {len(missing_locked)} 条 locked prefilter 规则在实验中缺失:"
                )
                for lr in missing_locked:
                    desc = lr.get("feature") or [
                        s.get("feature") for s in lr.get("any_of", [])
                    ]
                    print(f"      🔒 {desc}  ({lr.get('lock_reason', '?')})")
                print(
                    f"   → 使用 --force-unlock 强制采纳, 或检查 pipeline 是否保留了语义核心特征"
                )
                return False
            else:
                print(
                    f"   🔒 Prefilter 语义锁定校验通过: {len(locked_rules)} 条 locked 规则均存在"
                )

    # ── 语义锁定: 读取生产 gate.yaml 中的 locked 规则, 确保采纳后不丢失 ──
    prod_gate = prod_arch / "gate.yaml"
    exp_gate_path = exp_arch / "gate.yaml"
    if prod_gate.exists():
        prod_locked_gates = load_locked_gate_rules(prod_gate)
        if prod_locked_gates:
            if exp_gate_path.exists():
                exp_locked_gates = load_locked_gate_rules(exp_gate_path)
                exp_gate_ids = {r.get("id", "") for r in exp_locked_gates}
                missing_gate_locked = [
                    r for r in prod_locked_gates if r.get("id", "") not in exp_gate_ids
                ]
                if missing_gate_locked:
                    print(
                        f"   🔒 Gate: {len(missing_gate_locked)} 条 locked 规则在实验中缺失, 将自动回补"
                    )
            else:
                missing_gate_locked = prod_locked_gates

            # 采纳后自动回补 locked gate 规则（可能 disabled, 但不丢失）
            if missing_gate_locked:
                _merge_target = exp_gate_path if exp_gate_path.exists() else prod_gate
                merge_locked_gate_rules(_merge_target, missing_gate_locked)

    # ── 语义锁定: 读取生产 entry_filters.yaml 中的 locked filters, 确保采纳后不丢失 ──
    prod_entry_filters = prod_arch / "entry_filters.yaml"
    exp_entry_filters = exp_arch / "entry_filters.yaml"
    if prod_entry_filters.exists():
        prod_locked_entry = load_locked_entry_filters(prod_entry_filters)
        if prod_locked_entry:
            if exp_entry_filters.exists():
                exp_locked_entry = load_locked_entry_filters(exp_entry_filters)
                exp_ids = {f.get("id", "") for f in exp_locked_entry}
                missing_locked_entry = [
                    f for f in prod_locked_entry if f.get("id", "") not in exp_ids
                ]
                if missing_locked_entry:
                    print(
                        f"   🔒 EntryFilter: {len(missing_locked_entry)} 条 locked 规则在实验中缺失, 将自动回补"
                    )
            else:
                missing_locked_entry = prod_locked_entry

            if missing_locked_entry:
                _ef_merge_target = (
                    exp_entry_filters
                    if exp_entry_filters.exists()
                    else prod_entry_filters
                )
                merge_locked_entry_filters(_ef_merge_target, missing_locked_entry)

    prod_arch.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in exp_arch.iterdir():
        if f.is_file():
            shutil.copy2(f, prod_arch / f.name)
            copied += 1

    # gate_draft.yaml 仅保留在实验/rolling 输出目录，不 adopt 到 config/strategies（与模板分离）

    print(f"   ✅ Adopted: {copied} files → {prod_arch}")
    return True


def _cmd_list_experiments(history_dir: Path, strategy: str):
    """列出指定策略的历史实验。

    Multi-seed 实验组（同一 timestamp, 不同 _s{N} 后缀）只显示胜出的 seed，
    其余 seed 折叠为一行摘要，避免列表污染。
    """
    import re
    import time

    strat_dir = history_dir / strategy
    if not strat_dir.exists():
        print(f"\n📋 {strategy.upper()}: 无历史实验")
        return

    runs = sorted(d for d in strat_dir.iterdir() if d.is_dir())

    # 分组: base_timestamp -> [(run_dir, seed_num)]
    # e.g. 20260226_211920_s1, _s2, _s42 => base=20260226_211920
    seed_re = re.compile(r"^(.+?)_s(\d+)$")
    groups = {}  # base_ts -> [(run_dir, seed_num)]
    standalone = []  # 无 seed 后缀的单独实验
    for run_dir in runs:
        m = seed_re.match(run_dir.name)
        if m:
            base_ts, seed_num = m.group(1), int(m.group(2))
            groups.setdefault(base_ts, []).append((run_dir, seed_num))
        else:
            standalone.append(run_dir)

    # 构建显示列表: (sort_key, lines)
    display_items = []

    for run_dir in standalone:
        lines = _format_experiment_line(run_dir)
        display_items.append((run_dir.name, lines))

    for base_ts, members in groups.items():
        members.sort(key=lambda x: x[1])
        if len(members) == 1:
            # 只有一个 seed，当单独实验显示
            lines = _format_experiment_line(members[0][0])
            display_items.append((base_ts, lines))
            continue

        # 多 seed: 找胜出的 (有 report + 最高 sharpe)
        best_dir, best_seed, best_sharpe = None, None, -999
        n_total = len(members)
        n_no_report = 0
        n_error = 0
        for run_dir, seed_num in members:
            report_file = run_dir / "report.json"
            if not report_file.exists():
                n_no_report += 1
                continue
            try:
                report = json.loads(report_file.read_text(encoding="utf-8"))
                bt = report.get("backtest_metrics", {})
                comp = report.get("comparison", {})
                decision = comp.get("decision", "?")
                sharpe = bt.get("sharpe_per_trade", 0) or 0
                if decision == "ERROR":
                    n_error += 1
                if isinstance(sharpe, (int, float)) and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_dir = run_dir
                    best_seed = seed_num
            except Exception:
                n_no_report += 1

        if best_dir is not None:
            lines = _format_experiment_line(best_dir)
            seed_note = f"  └─ multi-seed: winner=s{best_seed}/{n_total} seeds"
            if n_no_report > 0:
                seed_note += f", {n_no_report} incomplete"
            if n_error > 0:
                seed_note += f", {n_error} error"
            lines.append(seed_note)
            display_items.append((base_ts, lines))
        else:
            # 所有 seed 都没有有效 report
            first_dir = members[0][0]
            created_time = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(first_dir.stat().st_ctime)
            )
            lines = [
                f"  {base_ts:<22s} {created_time:<15s}  (无有效 report, {n_total} seeds 全部失败)"
            ]
            display_items.append((base_ts, lines))

    display_items.sort(key=lambda x: x[0])
    n_logical = len(display_items)
    print(f"\n📋 {strategy.upper()} 历史实验 ({n_logical} 次):")
    print(f"{'─'*100}")
    print(
        f"  {'时间戳':<22s} {'创建时间':<15s} {'Sharpe':>10s} {'Trades':>8s} {'决策':>8s}  备注"
    )
    print(f"{'─'*100}")
    for _, lines in display_items:
        for line in lines:
            print(line)


def _format_experiment_line(run_dir: Path) -> list:
    """格式化单个实验目录为显示行."""
    import time

    created_time = time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(run_dir.stat().st_ctime)
    )
    report_file = run_dir / "report.json"
    if not report_file.exists():
        return [f"  {run_dir.name:<22s} {created_time:<15s}  (无 report.json)"]

    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except Exception:
        return [f"  {run_dir.name:<22s} {created_time:<15s}  (report.json 损坏)"]

    bt = report.get("backtest_metrics", {})
    comp = report.get("comparison", {})
    decision = comp.get("decision", "?")
    sharpe = bt.get("sharpe_per_trade", "N/A")
    trades = bt.get("total_trades", "N/A")
    dr = report.get("data_range", {})
    note = f"{dr.get('start_date', '?')}~{dr.get('end_date', '?')}"

    emoji = {"ADOPT": "✅", "ALERT": "⚠️", "KEEP": "🔄", "ERROR": "❌"}.get(
        decision, "❓"
    )
    sharpe_str = f"{sharpe:.4f}" if isinstance(sharpe, (int, float)) else str(sharpe)
    lines = [
        f"  {run_dir.name:<22s} {created_time:<15s} {sharpe_str:>10s} {str(trades):>8s} {emoji}{decision:>6s}  {note}"
    ]
    # 尝试找到交易地图 HTML 路径 (gate_dir / trading_map_{strategy}.html)
    _artifacts = report.get("artifacts", {})
    gate_dir_str = _artifacts.get("gate_dir") or _artifacts.get("evidence_dir")
    if gate_dir_str:
        # strategy 名从 gate_dir 路径的最后一段推断
        _gpath = Path(gate_dir_str)
        _strategy_name = _gpath.name
        _map_candidates = list(_gpath.glob(f"trading_map_{_strategy_name}*.html"))
        if _map_candidates:
            _map_path = _map_candidates[0]
            lines.append(f"    └─ 🗺️  {_map_path}")
    return lines


def _cmd_adopt_experiment(history_dir: Path, cfg: dict, strategy: str, timestamp: str):
    """手动采纳指定实验."""
    run_dir = history_dir / strategy / timestamp
    if not run_dir.exists():
        print(f"❌ 实验不存在: {run_dir}")
        # 列出可用的
        strat_dir = history_dir / strategy
        if strat_dir.exists():
            available = [d.name for d in sorted(strat_dir.iterdir()) if d.is_dir()]
            if available:
                print(f"   可用: {', '.join(available[-5:])}")
        return

    scfg = cfg["strategies"][strategy]
    exp_config_dir = run_dir / "strategies" / strategy
    if not exp_config_dir.exists():
        # 旧版实验 (无隔离), 尝试从 archetypes 快照恢复
        arch_snapshot = run_dir / "archetypes"
        if arch_snapshot.exists():
            prod_arch = PROJECT_ROOT / scfg["config"] / "archetypes"
            for f in arch_snapshot.iterdir():
                if f.is_file():
                    shutil.copy2(f, prod_arch / f.name)
            print(f"✅ Adopted (from snapshot): {prod_arch}")
        else:
            print(f"❌ 实验目录中找不到 strategies/ 或 archetypes/ 快照")
        return

    _adopt_experiment_config(exp_config_dir, scfg["config"])


def _cmd_diff_experiments(history_dir: Path, strategy: str, ts1: str, ts2: str):
    """对比两次实验 — 输出结构化漂移报告."""
    dir1 = history_dir / strategy / ts1
    dir2 = history_dir / strategy / ts2

    for d, ts in [(dir1, ts1), (dir2, ts2)]:
        if not d.exists():
            print(f"❌ 实验不存在: {d}")
            return

    arch1 = _find_arch_dir(dir1, strategy)
    arch2 = _find_arch_dir(dir2, strategy)
    if not arch1 or not arch2:
        print("❌ 至少一个实验缺少 archetypes 数据")
        return

    rpt1 = _load_report_metrics(dir1)
    rpt2 = _load_report_metrics(dir2)

    _print_drift_report(strategy, ts1, ts2, arch1, arch2, rpt1, rpt2)


def _find_arch_dir(run_dir: Path, strategy: str) -> Optional[Path]:
    """查找 archetypes 目录 (优先实验隔离版, fallback 快照)."""
    exp_arch = run_dir / "strategies" / strategy / "archetypes"
    if exp_arch.exists():
        return exp_arch
    snap_arch = run_dir / "archetypes"
    if snap_arch.exists():
        return snap_arch
    return None


def _load_report_metrics(run_dir: Path) -> Dict[str, Any]:
    rpt = run_dir / "report.json"
    if rpt.exists():
        r = json.loads(rpt.read_text(encoding="utf-8"))
        return r.get("backtest_metrics", {})
    return {}


# ── 分层汇总表 ─────────────────────────────────────────────────────


def _read_yaml_safe(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _extract_layer_info(arch_dir: Path) -> Dict[str, Any]:
    """Extract per-layer summary from archetypes directory."""
    info: Dict[str, Any] = {}

    # Prefilter
    pf = _read_yaml_safe(arch_dir / "prefilter.yaml")
    pf_rules = pf.get("rules", [])
    pf_descs = []
    for r in pf_rules:
        if "any_of" in r:
            subs = [
                f"{s.get('feature', '?')}{s.get('operator', '')}{s.get('value', '')}"
                for s in r["any_of"]
            ]
            pf_descs.append("(" + " OR ".join(subs) + ")")
        else:
            pf_descs.append(
                f"{r.get('feature', '?')} {r.get('operator', '')} {r.get('value', '')}"
            )
    info["prefilter"] = {"count": len(pf_rules), "rules": pf_descs}

    # Direction
    dr = _read_yaml_safe(arch_dir / "direction.yaml")
    dr_rules = dr.get("direction_rules", [])
    dr_descs = [f"{r.get('feature', '?')}(w={r.get('weight', 1)})" for r in dr_rules]
    info["direction"] = {"count": len(dr_rules), "rules": dr_descs}

    # Gate
    gt = _read_yaml_safe(arch_dir / "gate.yaml")
    sg = gt.get("system_safety", [])
    hg = gt.get("hard_gates", [])
    gr = gt.get("guardrails", [])
    hg_descs = []
    for r in list(sg) + list(hg):
        rid = r.get("id", "")
        w = r.get("when", {})
        if "all_of" in w:
            # compound gate (e.g. OR prefilter negated)
            parts = []
            for sub in w["all_of"]:
                if isinstance(sub, dict):
                    for sf, sc in sub.items():
                        if isinstance(sc, dict):
                            for sop, sv in sc.items():
                                parts.append(f"{sf} {sop.replace('value_','')} {sv}")
            hg_descs.append(" AND ".join(parts) if parts else rid)
        else:
            for feat, cond in w.items():
                if isinstance(cond, dict):
                    for op, val in cond.items():
                        op_s = op.replace("value_", "")
                        hg_descs.append(f"{feat} {op_s} {val}")
                else:
                    hg_descs.append(f"{feat}: {cond}")
    info["gate"] = {
        "system_safety": len(sg),
        "hard_gates": len(hg),
        "guardrails": len(gr),
        "rules": hg_descs,
    }

    # Evidence
    ev = _read_yaml_safe(arch_dir / "evidence.yaml")
    ev_axes = ev.get("evidence_axes", ev.get("evidence", []))
    ev_descs = []
    for r in ev_axes:
        feat = r.get("feature", "?")
        direction = r.get("direction", "")
        qm = r.get("quantile_mapping", {})
        bins = qm.get("bins", [])
        if bins:
            ev_descs.append(f"{feat}({direction}, bins={len(bins)})")
        else:
            ev_descs.append(f"{feat}({direction})")
    info["evidence"] = {"count": len(ev_axes), "rules": ev_descs}

    # Entry Filters
    ef = _read_yaml_safe(arch_dir / "entry_filters.yaml")
    ef_filters = ef.get("filters", [])
    ef_descs = [f.get("id", f.get("name", "?")) for f in ef_filters]
    info["entry_filters"] = {"count": len(ef_filters), "rules": ef_descs}

    # Execution
    ex = _read_yaml_safe(arch_dir / "execution.yaml")
    ex_summary = {}
    if ex.get("stop_loss"):
        sl = ex["stop_loss"]
        init_r = sl.get("initial_r", sl.get("r_multiple", sl.get("atr_multiple", "?")))
        trail = sl.get("trailing", {})
        act_r = trail.get("activation_r", "")
        trail_r = trail.get("trail_r", "")
        sl_s = f"{init_r}R"
        if act_r:
            sl_s += f"(act={act_r},trail={trail_r})"
        ex_summary["SL"] = sl_s
    if ex.get("take_profit") and ex["take_profit"].get("enabled", True):
        tp = ex["take_profit"]
        tp_val = tp.get("r_multiple", tp.get("target_r", tp.get("atr_multiple", "?")))
        ex_summary["TP"] = f"{tp_val}R"
    if ex.get("holding"):
        h = ex["holding"]
        mb = h.get("max_holding_bars", h.get("max_bars", None))
        ts = h.get("time_stop_bars", None)
        if ts:
            ex_summary["time_stop"] = f"{ts}bars"
        elif mb:
            ex_summary["max_bars"] = mb
    if ex.get("tiers", {}).get("enabled"):
        ex_summary["tiers"] = len(ex["tiers"].get("levels", []))
    # fallback: generic params
    ex_params = ex.get("params", ex.get("execution_params", {}))
    if ex_params:
        ex_summary.update({k: v for k, v in list(ex_params.items())[:3]})
    info["execution"] = ex_summary

    return info


def _fmt_rules(rules: list, max_show: int = 3) -> str:
    if not rules:
        return ""
    shown = rules[:max_show]
    rest = len(rules) - max_show
    s = ", ".join(str(r) for r in shown)
    if rest > 0:
        s += f" (+{rest}更多)"
    return s


def _delta_str(cur: int, prev: int) -> str:
    if prev == cur:
        return "—"
    diff = cur - prev
    return f"{diff:+d}" if diff != 0 else "—"


def print_layer_summary(
    strategy: str,
    timestamp: str,
    arch_dir: Path,
    backtest_metrics: Dict[str, Any],
    prev_arch_dir: Optional[Path] = None,
    prev_metrics: Optional[Dict[str, Any]] = None,
):
    """Pipeline 结束时打印分层汇总表."""
    cur = _extract_layer_info(arch_dir)
    prev = _extract_layer_info(prev_arch_dir) if prev_arch_dir else None
    w = 74
    sep = "─" * w

    print(f"\n{'═' * w}")
    print(f"  {strategy.upper()} 分层配置汇总  ({timestamp})")
    print(f"{'═' * w}")

    # ── Prefilter ──
    pf = cur["prefilter"]
    line = f"  L2 Prefilter     {pf['count']} rule(s)"
    if prev:
        pp = prev["prefilter"]
        line += f"  ← prev {pp['count']}  {_delta_str(pf['count'], pp['count'])}"
    print(line)
    if pf["rules"]:
        print(f"                   {_fmt_rules(pf['rules'])}")

    # ── Direction ──
    dr = cur["direction"]
    line = f"  L3 Direction     {dr['count']} feature(s)"
    if prev:
        pd_ = prev["direction"]
        line += f"  ← prev {pd_['count']}  {_delta_str(dr['count'], pd_['count'])}"
    print(line)
    if dr["rules"]:
        print(f"                   {_fmt_rules(dr['rules'])}")

    # ── Gate ──
    gt = cur["gate"]
    line = (
        f"  L4 Gate          {gt['system_safety']} safety_gate(s), "
        f"{gt['hard_gates']} hard_gate(s), {gt['guardrails']} guardrail(s)"
    )
    if prev:
        pg = prev["gate"]
        line += (
            f"  ← prev {pg['system_safety']}+{pg['hard_gates']}+{pg['guardrails']}  "
            f"{_delta_str(gt['system_safety'] + gt['hard_gates'], pg['system_safety'] + pg['hard_gates'])}"
        )
    print(line)
    if gt["rules"]:
        print(f"                   {_fmt_rules(gt['rules'])}")

    # ── Evidence ──
    ev = cur["evidence"]
    line = f"  L5 Evidence      {ev['count']} axis/axes"
    if prev:
        pe = prev["evidence"]
        line += f"  ← prev {pe['count']}  {_delta_str(ev['count'], pe['count'])}"
    print(line)
    if ev["rules"]:
        print(f"                   {_fmt_rules(ev['rules'], max_show=5)}")

    # ── Entry Filters ──
    ef = cur["entry_filters"]
    line = f"  L6 Entry Filter  {ef['count']} filter(s)"
    if prev:
        pef = prev["entry_filters"]
        line += f"  ← prev {pef['count']}  {_delta_str(ef['count'], pef['count'])}"
    print(line)
    if ef["rules"]:
        print(f"                   {_fmt_rules(ef['rules'])}")

    # ── Execution ──
    ex = cur.get("execution", {})
    ex_parts = [f"{k}={v}" for k, v in list(ex.items())[:5]] if ex else ["(默认)"]
    print(f"  L7 Execution     {', '.join(ex_parts)}")

    # ── Backtest ──
    print(f"  {sep}")
    bt = backtest_metrics
    sharpe_pt = bt.get("sharpe_per_trade", "N/A")
    sharpe_d = bt.get("sharpe_daily", "")
    trades = bt.get("total_trades", "N/A")
    winr = bt.get("win_rate", "N/A")
    mean_r = bt.get("mean_r", "N/A")
    sharpe_s = (
        f"{sharpe_pt:.4f}" if isinstance(sharpe_pt, (int, float)) else str(sharpe_pt)
    )
    daily_s = f" (daily {sharpe_d:.2f})" if isinstance(sharpe_d, (int, float)) else ""
    winr_s = f"{winr:.1%}" if isinstance(winr, (int, float)) else str(winr)
    mean_r_s = f"{mean_r:.4f}" if isinstance(mean_r, (int, float)) else str(mean_r)
    line = f"  Backtest         Sharpe={sharpe_s}{daily_s}  Trades={trades}  Win={winr_s}  MeanR={mean_r_s}"
    print(line)
    if prev_metrics:
        p_sharpe = prev_metrics.get("sharpe_per_trade")
        p_trades = prev_metrics.get("total_trades")
        p_winr = prev_metrics.get("win_rate")
        parts = []
        if isinstance(p_sharpe, (int, float)):
            parts.append(f"Sharpe={p_sharpe:.4f}")
            if isinstance(sharpe_pt, (int, float)) and p_sharpe != 0:
                pct = (sharpe_pt - p_sharpe) / abs(p_sharpe) * 100
                parts.append(f"Δ={pct:+.1f}%")
        if p_trades is not None:
            parts.append(f"Trades={p_trades}")
        if isinstance(p_winr, (int, float)):
            parts.append(f"Win={p_winr:.1%}")
        if parts:
            print(f"     prev:         {', '.join(parts)}")

    print(f"{'═' * w}")


# ── 漂移报告核心 ─────────────────────────────────────────────────


def _pct_change(old: float, new: float) -> str:
    if old == 0:
        return "N/A"
    pct = (new - old) / abs(old) * 100
    return f"{pct:+.1f}%"


def _drift_level(changes: List[str]) -> str:
    """从子项漂移标记列表中取最高."""
    order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
    level = max(changes, key=lambda x: order.get(x, 0), default="NONE")
    return level


def _drift_emoji(level: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "NONE": "⚪"}.get(level, "❓")


def _analyze_prefilter(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 prefilter.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    r1 = y1.get("rules", [])
    r2 = y2.get("rules", [])

    # 提取所有 feature->value 对
    def _extract_features(rules: list) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for rule in rules:
            if isinstance(rule, dict):
                if "any_of" in rule:
                    for sub in rule["any_of"]:
                        out[sub.get("feature", "?")] = sub.get("value", 0)
                elif "feature" in rule:
                    out[rule["feature"]] = rule.get("value", 0)
        return out

    f1 = _extract_features(r1)
    f2 = _extract_features(r2)
    feats1 = set(f1.keys())
    feats2 = set(f2.keys())

    if feats1 == feats2:
        lines.append(f"   Rules 特征: 不变 ({', '.join(sorted(feats1))})")
    else:
        added = feats2 - feats1
        removed = feats1 - feats2
        if added:
            lines.append(f"   Rules 新增特征: {', '.join(sorted(added))}")
        if removed:
            lines.append(f"   Rules 移除特征: {', '.join(sorted(removed))}")
        drifts.append("HIGH")

    # 共有特征的阈值漂移
    for feat in sorted(feats1 & feats2):
        v1, v2 = f1[feat], f2[feat]
        if v1 != v2:
            lines.append(f"   阈值漂移: {feat} {v1} → {v2} ({_pct_change(v1, v2)})")
            pct = abs(v2 - v1) / max(abs(v1), 1e-9) * 100
            drifts.append("MEDIUM" if pct < 20 else "HIGH")
        else:
            drifts.append("NONE")

    if not drifts:
        drifts.append("NONE")
    level = _drift_level(drifts)
    return lines, level


def _analyze_gate(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 gate.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    hg1 = list(y1.get("system_safety", []) or []) + list(y1.get("hard_gates", []) or [])
    hg2 = list(y2.get("system_safety", []) or []) + list(y2.get("hard_gates", []) or [])
    ids1 = {r.get("id", f"rule_{i}"): r for i, r in enumerate(hg1)}
    ids2 = {r.get("id", f"rule_{i}"): r for i, r in enumerate(hg2)}
    set1, set2 = set(ids1.keys()), set(ids2.keys())

    lines.append(f"   规则数: {len(hg1)} → {len(hg2)}")
    added = set2 - set1
    removed = set1 - set2
    common = set1 & set2

    if added:
        lines.append(f"   新增规则: {', '.join(sorted(added))}")
        drifts.append("MEDIUM")
    if removed:
        lines.append(f"   移除规则: {', '.join(sorted(removed))}")
        drifts.append("HIGH" if len(removed) > 2 else "MEDIUM")

    # 共有规则阈值对比
    changed_count = 0
    for rid in sorted(common):
        r1, r2 = ids1[rid], ids2[rid]
        w1, w2 = r1.get("when", {}), r2.get("when", {})

        # 提取阈值
        def _get_threshold(when: dict) -> Optional[float]:
            for feat, conds in when.items():
                if isinstance(conds, dict):
                    for k, v in conds.items():
                        if k.startswith("value_") and isinstance(v, (int, float)):
                            return float(v)
            return None

        t1, t2 = _get_threshold(w1), _get_threshold(w2)
        if t1 is not None and t2 is not None and t1 != t2:
            lines.append(f"   {rid}: 阈值 {t1:.4f} → {t2:.4f} ({_pct_change(t1, t2)})")
            changed_count += 1
            pct = abs(t2 - t1) / max(abs(t1), 1e-9) * 100
            drifts.append("MEDIUM" if pct < 30 else "HIGH")

    if changed_count == 0 and not added and not removed:
        lines.append("   阈值: 全部不变")
        drifts.append("NONE")

    level = _drift_level(drifts) if drifts else "NONE"
    return lines, level


def _analyze_evidence(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 evidence.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    def _get_features(y: dict) -> Dict[str, dict]:
        feats = y.get("features", y.get("evidence_features", []))
        if isinstance(feats, list):
            return {
                f.get("name", f.get("feature", f"feat_{i}")): f
                for i, f in enumerate(feats)
            }
        return {}

    f1, f2 = _get_features(y1), _get_features(y2)
    set1, set2 = set(f1.keys()), set(f2.keys())

    if set1 == set2:
        lines.append(f"   特征集合: 不变 ({len(set1)} 个)")
    else:
        added = set2 - set1
        removed = set1 - set2
        if added:
            lines.append(f"   新增特征: {', '.join(sorted(added))}")
            drifts.append("MEDIUM")
        if removed:
            lines.append(f"   移除特征: {', '.join(sorted(removed))}")
            drifts.append("MEDIUM")

    # 共有特征阈值对比
    for fname in sorted(set1 & set2):
        e1, e2 = f1[fname], f2[fname]
        for key in ["threshold", "weight", "min_score", "value"]:
            v1 = e1.get(key)
            v2 = e2.get(key)
            if v1 is not None and v2 is not None and v1 != v2:
                lines.append(
                    f"   {fname}.{key}: {v1} → {v2} ({_pct_change(float(v1), float(v2))})"
                )
                drifts.append("LOW")

    if not drifts:
        drifts.append("NONE")
    return lines, _drift_level(drifts)


def _analyze_execution(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 execution.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    for section in ["stop_loss", "take_profit", "trailing_stop"]:
        s1 = y1.get(section, {})
        s2 = y2.get(section, {})
        if not isinstance(s1, dict) or not isinstance(s2, dict):
            continue
        all_keys = sorted(set(list(s1.keys()) + list(s2.keys())))
        for k in all_keys:
            v1, v2 = s1.get(k), s2.get(k)
            if v1 == v2:
                continue
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                lines.append(
                    f"   {section}.{k}: {v1} → {v2} ({_pct_change(float(v1), float(v2))})"
                )
                pct = abs(v2 - v1) / max(abs(v1), 1e-9) * 100
                drifts.append("LOW" if pct < 15 else "MEDIUM")
            elif v1 != v2:
                lines.append(f"   {section}.{k}: {v1} → {v2}")
                drifts.append("LOW")

    if not drifts:
        lines.append("   参数: 全部不变")
        drifts.append("NONE")
    return lines, _drift_level(drifts)


def _analyze_direction(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 direction.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    # primary feature
    p1 = y1.get("primary", y1.get("direction_feature", ""))
    p2 = y2.get("primary", y2.get("direction_feature", ""))
    if p1 == p2:
        lines.append(f"   主特征: {p1} (不变)")
    else:
        lines.append(f"   主特征: {p1} → {p2}")
        drifts.append("HIGH")

    # fallback features
    eval1 = y1.get("last_evaluation", {})
    eval2 = y2.get("last_evaluation", {})
    fb1 = [f.get("feature", "") for f in eval1.get("fallback", [])]
    fb2 = [f.get("feature", "") for f in eval2.get("fallback", [])]
    fb_common = len(set(fb1) & set(fb2))
    fb_total = max(len(set(fb1) | set(fb2)), 1)
    if fb1 == fb2:
        lines.append(f"   Fallback 候选: 不变 ({len(fb1)} 个)")
    else:
        lines.append(f"   Fallback 候选: {fb_common}/{fb_total} 个重合")
        overlap = fb_common / fb_total
        drifts.append("LOW" if overlap > 0.6 else "MEDIUM")

    # n_rows change
    nr1 = eval1.get("n_rows", 0)
    nr2 = eval2.get("n_rows", 0)
    if nr1 and nr2 and nr1 != nr2:
        lines.append(f"   数据量: {nr1:,} → {nr2:,} ({_pct_change(nr1, nr2)})")

    if not drifts:
        drifts.append("NONE")
    return lines, _drift_level(drifts)


def _analyze_entry_filters(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 entry_filters.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    filters1 = y1.get("filters", [])
    filters2 = y2.get("filters", [])
    ids1 = {f.get("id", f"f{i}"): f for i, f in enumerate(filters1)}
    ids2 = {f.get("id", f"f{i}"): f for i, f in enumerate(filters2)}
    set1, set2 = set(ids1.keys()), set(ids2.keys())

    lines.append(f"   Filter 数: {len(filters1)} → {len(filters2)}")
    added = set2 - set1
    removed = set1 - set2
    if added:
        lines.append(f"   新增: {', '.join(sorted(added))}")
        drifts.append("MEDIUM")
    if removed:
        lines.append(f"   移除: {', '.join(sorted(removed))}")
        drifts.append("MEDIUM")

    # 共有 filter 的 enabled/threshold 对比
    for fid in sorted(set1 & set2):
        ef1, ef2 = ids1[fid], ids2[fid]
        en1, en2 = ef1.get("enabled", True), ef2.get("enabled", True)
        if en1 != en2:
            lines.append(f"   {fid}: enabled {en1} → {en2}")
            drifts.append("MEDIUM")
        # threshold
        for key in ["threshold", "value", "min_value", "max_value"]:
            v1, v2 = ef1.get(key), ef2.get(key)
            if v1 is not None and v2 is not None and v1 != v2:
                lines.append(f"   {fid}.{key}: {v1} → {v2}")
                drifts.append("LOW")

    if not drifts:
        drifts.append("NONE")
    return lines, _drift_level(drifts)


def _analyze_generic(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """通用 YAML 对比 (holding.yaml 等)."""
    if y1 == y2:
        return ["   无变化"], "NONE"
    lines: List[str] = []
    _flat_diff(y1, y2, lines, prefix="   ")
    level = "LOW" if len(lines) <= 3 else "MEDIUM"
    return lines, level


def _flat_diff(
    d1: dict, d2: dict, lines: List[str], prefix: str = "", max_lines: int = 10
):
    """递归扁平化 diff, 最多 max_lines 行."""
    all_keys = sorted(set(list(d1.keys()) + list(d2.keys())))
    for k in all_keys:
        if len(lines) >= max_lines:
            lines.append(f"{prefix}... (更多差异省略)")
            return
        v1, v2 = d1.get(k), d2.get(k)
        if v1 == v2:
            continue
        if k not in d1:
            lines.append(f"{prefix}+ {k}")
        elif k not in d2:
            lines.append(f"{prefix}- {k}")
        elif isinstance(v1, dict) and isinstance(v2, dict):
            _flat_diff(v1, v2, lines, prefix, max_lines)
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            lines.append(
                f"{prefix}{k}: {v1} → {v2} ({_pct_change(float(v1), float(v2))})"
            )
        else:
            # 对于 list 等复杂类型, 只显示有/无变化
            lines.append(f"{prefix}{k}: 已变更")


_FILE_ANALYZERS = {
    "prefilter.yaml": _analyze_prefilter,
    "gate.yaml": _analyze_gate,
    "evidence.yaml": _analyze_evidence,
    "execution.yaml": _analyze_execution,
    "direction.yaml": _analyze_direction,
    "entry_filters.yaml": _analyze_entry_filters,
}


def _print_drift_report(
    strategy: str,
    ts1: str,
    ts2: str,
    arch1: Path,
    arch2: Path,
    metrics1: Dict[str, Any],
    metrics2: Dict[str, Any],
) -> Dict[str, str]:
    """输出结构化漂移报告, 返回 {filename: drift_level}."""
    w = 72
    print(f"\n{'╔' + '═' * w + '╗'}")
    print(f"║  {strategy.upper()} Archetype 漂移报告{' ' * (w - len(strategy) - 22)}║")
    print(f"║  旧: {ts1}   新: {ts2}{' ' * (w - len(ts1) - len(ts2) - 12)}║")
    print(f"{'╚' + '═' * w + '╝'}")

    # ── Metrics 对比 ──
    print(f"\n📊 Metrics 对比")
    print(f"   {'─' * 56}")
    print(f"   {'指标':<16s} {'旧':>12s} {'新':>12s} {'变化':>10s}")
    print(f"   {'─' * 56}")
    for key, label, fmt in [
        ("sharpe_per_trade", "Sharpe", ".4f"),
        ("total_trades", "Trades", ".0f"),
        ("win_rate", "Win Rate", ".2%"),
        ("mean_r", "Mean R", ".4f"),
    ]:
        v1 = metrics1.get(key)
        v2 = metrics2.get(key)
        if v1 is not None and v2 is not None:
            s1 = f"{v1:{fmt}}" if isinstance(v1, (int, float)) else str(v1)
            s2 = f"{v2:{fmt}}" if isinstance(v2, (int, float)) else str(v2)
            chg = (
                _pct_change(float(v1), float(v2))
                if isinstance(v1, (int, float)) and v1 != 0
                else ""
            )
            print(f"   {label:<16s} {s1:>12s} {s2:>12s} {chg:>10s}")
    print(f"   {'─' * 56}")

    # ── 逐文件分析 ──
    file_drifts: Dict[str, str] = {}
    all_files = sorted(
        set(
            [f.name for f in arch1.iterdir() if f.is_file()]
            + [f.name for f in arch2.iterdir() if f.is_file()]
        )
    )

    for fname in all_files:
        f1_path, f2_path = arch1 / fname, arch2 / fname
        if not f1_path.exists():
            print(f"\n📄 {fname}: 仅存在于新版 ⚡")
            file_drifts[fname] = "HIGH"
            continue
        if not f2_path.exists():
            print(f"\n📄 {fname}: 新版中已移除 ⚡")
            file_drifts[fname] = "HIGH"
            continue

        text1, text2 = f1_path.read_text(encoding="utf-8"), f2_path.read_text(
            encoding="utf-8"
        )
        if text1 == text2:
            print(f"\n📄 {fname}: 无变化 ✅")
            file_drifts[fname] = "NONE"
            continue

        try:
            y1 = yaml.safe_load(text1) or {}
            y2 = yaml.safe_load(text2) or {}
        except Exception:
            print(f"\n📄 {fname}: 有差异 (YAML 解析失败)")
            file_drifts[fname] = "MEDIUM"
            continue

        analyzer = _FILE_ANALYZERS.get(fname, _analyze_generic)
        detail_lines, level = analyzer(y1, y2)
        emoji = _drift_emoji(level)
        print(f"\n📄 {fname}: {emoji} {level}")
        for line in detail_lines:
            print(line)
        file_drifts[fname] = level

    # ── 综合判定 ──
    overall = _drift_level(list(file_drifts.values()))
    overall_emoji = _drift_emoji(overall)

    # 决定建议
    sharpe1 = metrics1.get("sharpe_per_trade", 0)
    sharpe2 = metrics2.get("sharpe_per_trade", 0)
    sharpe_stable = abs(sharpe2 - sharpe1) / max(abs(sharpe1), 1e-9) < 0.05  # < 5% 变化

    if overall == "NONE" or (overall == "LOW" and sharpe_stable):
        advice = "STABLE — 参数稳定, 可直接 ADOPT"
    elif overall in ("LOW", "MEDIUM") and sharpe_stable:
        advice = "MONITOR — Sharpe 稳定但参数有漂移, 建议检查变动项后 ADOPT"
    elif overall == "MEDIUM" and not sharpe_stable:
        advice = "REVIEW — 参数与 Sharpe 同时漂移, 需人工审查变动原因"
    else:  # HIGH
        high_files = [f for f, l in file_drifts.items() if l == "HIGH"]
        advice = f"ADJUST — 大幅漂移 ({', '.join(high_files)}), 需人工审查并可能回退"

    print(f"\n{'━' * 74}")
    print(f"🎯 综合判定")
    print(f"   总体漂移: {overall_emoji} {overall}")
    print(f"   建议:     {advice}")
    print(f"{'━' * 74}")

    return file_drifts


if __name__ == "__main__":
    raise SystemExit(pipeline_cli.main())
