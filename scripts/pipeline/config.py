from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.config.strategy_layout import deep_merge_dicts

from .context import PROJECT_ROOT


def _load_yaml_extends_chain(path: Path) -> Dict[str, Any]:
    """Load YAML following ``extends`` (child overlays parent). Paths are relative to each file."""
    chain: List[Dict[str, Any]] = []
    cur = path.resolve()
    visited: set[Path] = set()
    for _ in range(64):
        if cur in visited:
            raise ValueError(f"extends 循环引用: {cur}")
        visited.add(cur)
        raw = yaml.safe_load(cur.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"配置文件格式错误: {cur}")
        ext = raw.pop("extends", None)
        chain.append(raw)
        if not ext:
            break
        nxt = (cur.parent / str(ext).strip()).resolve()
        if not nxt.is_file():
            raise ValueError(f"extends 指向的文件不存在: {ext!r}（自 {cur}）")
        cur = nxt
    merged: Dict[str, Any] = {}
    for layer in reversed(chain):
        merged = deep_merge_dicts(merged, layer)
    return merged


def load_pipeline_config(path: Path) -> dict:
    cfg = _load_yaml_extends_chain(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误: {path}")

    dates = cfg.get("dates", {}) or {}
    if isinstance(dates, dict):
        d_cal = dates.get("calibration_months")
        rolling = cfg.setdefault("rolling", {})
        windows = rolling.setdefault("windows", {})
        if d_cal is not None and windows.get("calibration_months") is None:
            windows["calibration_months"] = d_cal
        rw_cal = windows.get("calibration_months")
        if d_cal is not None and rw_cal is not None and int(d_cal) != int(rw_cal):
            raise ValueError(
                "rolling.windows.calibration_months 与 dates.calibration_months 冲突"
            )

    grid_bt = cfg.get("grid_backtest")
    if isinstance(grid_bt, dict) and bool(grid_bt.get("enabled")):
        ds = (
            (cfg.get("dates") or {}).get("start_date")
            if isinstance(dates, dict)
            else None
        )
        gs = grid_bt.get("start_date")
        if ds and gs and str(ds).strip() != str(gs).strip():
            raise ValueError("grid_backtest.start_date 与 dates.start_date 不一致")

    rolling = cfg.get("rolling", {}) or {}
    if not isinstance(rolling, dict):
        rolling = {}
    mode = (
        str(rolling.get("mode", "slow_realistic") or "slow_realistic").strip().lower()
    )
    if mode not in {
        "slow_realistic",
        "turbo_fixed_features",
        "legacy",
        "non_rolling",
    }:
        raise ValueError(
            f"rolling.mode 非法: {mode} (允许 slow_realistic/turbo_fixed_features/legacy/non_rolling)"
        )

    tsp = rolling.get("time_split_policy")
    if tsp is not None:
        tsp_s = str(tsp).strip().lower()
        if tsp_s not in {"static_holdout"}:
            raise ValueError(f"time_split_policy 非法: {tsp} (仅支持 static_holdout)")
        rolling["time_split_policy"] = tsp_s
    elif mode == "turbo_fixed_features":
        rolling["time_split_policy"] = "static_holdout"

    windows = rolling.get("windows", {}) or {}
    if not isinstance(windows, dict):
        windows = {}
    calibration_months = int(windows.get("calibration_months", 3) or 3)
    structure_lookback_months = int(windows.get("structure_lookback_months", 12) or 12)
    _stw_raw = (
        str(windows.get("structure_train_window", "rolling_window") or "rolling_window")
        .strip()
        .lower()
    )
    if _stw_raw not in {"rolling_window", "full_history"}:
        raise ValueError(
            "rolling.windows.structure_train_window 非法: "
            f"{windows.get('structure_train_window')!r} "
            "(仅允许 rolling_window / full_history)"
        )
    structure_train_window = _stw_raw

    if mode != "non_rolling":
        if calibration_months <= 0:
            raise ValueError("rolling.windows.calibration_months 必须 > 0")
        if structure_lookback_months <= 0:
            raise ValueError("rolling.windows.structure_lookback_months 必须 > 0")
        if (
            mode == "slow_realistic"
            and structure_train_window != "full_history"
            and structure_lookback_months <= calibration_months
        ):
            raise ValueError(
                f"slow_realistic（rolling_window）要求 rolling.windows.structure_lookback_months "
                f"({structure_lookback_months}) > rolling.windows.calibration_months "
                f"({calibration_months})，否则慢结构快照窗内无训练段。"
                "请增大 structure_lookback_months 或减小 calibration_months（或 dates.calibration_months）；"
                "或使用 structure_train_window: full_history（起点为 dates.start_date）。"
            )
        if mode == "slow_realistic" and structure_train_window == "full_history":
            _gdates = cfg.get("dates")
            if not isinstance(_gdates, dict):
                raise ValueError(
                    "structure_train_window: full_history 需要 cfg dates 为字典且含 start_date"
                )
            _ds = _gdates.get("start_date")
            if not _ds or not str(_ds).strip():
                raise ValueError(
                    "structure_train_window: full_history 需要 dates.start_date（慢结构快照起点）"
                )
            try:
                datetime.strptime(str(_ds).strip()[:10], "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(
                    f"dates.start_date 非法（期望 YYYY-MM-DD）: {_ds!r}"
                ) from exc

    slow_realistic = rolling.get("slow_realistic", {}) or {}
    if not isinstance(slow_realistic, dict):
        slow_realistic = {}
    cadence_months = int(slow_realistic.get("cadence_months", 3) or 3)
    if mode == "slow_realistic" and cadence_months <= 0:
        raise ValueError("rolling.slow_realistic.cadence_months 必须 > 0")

    turbo_fixed = rolling.get("turbo_fixed_features", {}) or {}
    if not isinstance(turbo_fixed, dict):
        turbo_fixed = {}
    fixed_root = str(
        turbo_fixed.get("fixed_strategies_root", "config/strategies") or ""
    ).strip()
    if mode in {"slow_realistic", "turbo_fixed_features", "legacy"} and not fixed_root:
        raise ValueError("rolling.turbo_fixed_features.fixed_strategies_root 不能为空")

    contract_cfg = cfg.get("config_contract", {}) or {}
    if not isinstance(contract_cfg, dict):
        contract_cfg = {}
    slow_loop = cfg.get("slow_loop", {}) or {}
    if not isinstance(slow_loop, dict):
        slow_loop = {}
    slow_loop_policy = str(
        contract_cfg.get("slow_loop_policy", "warn") or "warn"
    ).lower()
    if slow_loop_policy not in {"warn", "error", "ignore"}:
        slow_loop_policy = "warn"
    if mode == "slow_realistic" and slow_loop:
        mismatches = []
        if "cadence_months" in slow_loop:
            try:
                sl_cad = int(
                    slow_loop.get("cadence_months", cadence_months) or cadence_months
                )
                if sl_cad != cadence_months:
                    mismatches.append(
                        f"slow_loop.cadence_months={sl_cad} vs rolling.slow_realistic.cadence_months={cadence_months}"
                    )
            except Exception:
                mismatches.append("slow_loop.cadence_months 非法")
        trig = slow_loop.get("triggered_retrain")
        if isinstance(trig, dict) and "enabled" in trig:
            sl_trig = bool(trig.get("enabled"))
            sr_trig = bool(slow_realistic.get("triggered_retrain_enabled", True))
            if sl_trig != sr_trig:
                mismatches.append(
                    f"slow_loop.triggered_retrain.enabled={sl_trig} vs rolling.slow_realistic.triggered_retrain_enabled={sr_trig}"
                )
        _msg = (
            "检测到 slow_loop 配置；rolling_sim 在 slow_realistic 模式只读取 "
            "rolling.slow_realistic.*，slow_loop.* 不会直接生效。"
        )
        if mismatches:
            _msg += " 键值不一致: " + "; ".join(mismatches)
        if slow_loop_policy == "error":
            raise ValueError(_msg)
        if slow_loop_policy == "warn":
            print(f"⚠️  {path}: {_msg}")

    rolling["mode"] = mode
    rolling["windows"] = {
        "calibration_months": calibration_months,
        "structure_lookback_months": structure_lookback_months,
        "structure_train_window": structure_train_window,
    }
    rolling["slow_realistic"] = {
        "cadence_months": cadence_months,
        "triggered_retrain_enabled": bool(
            slow_realistic.get("triggered_retrain_enabled", True)
        ),
    }
    rolling["turbo_fixed_features"] = {
        "fixed_strategies_root": fixed_root,
        "disable_feature_search": bool(turbo_fixed.get("disable_feature_search", True)),
    }
    cfg["rolling"] = rolling

    require_event_enabled = bool(
        contract_cfg.get("require_event_backtest_enabled", False)
    )
    event_cfg = cfg.get("event_backtest", {}) or {}
    if not isinstance(event_cfg, dict):
        event_cfg = {}
    if "enabled" not in event_cfg:
        _msg = (
            "event_backtest.enabled 未显式设置；将使用默认值 true。"
            "建议在配置中明确声明，避免 rolling_sim 误跑/误停。"
        )
        if require_event_enabled:
            raise ValueError(f"{path}: {_msg}")
        print(f"⚠️  {path}: {_msg}")
    event_cfg["enabled"] = bool(event_cfg.get("enabled", True))
    cfg["event_backtest"] = event_cfg

    # rolling_calibration: 月度 replay（fast_month / rolling_sim）归一化开关；业务口径以
    # threshold_calibration 与各根键为准，本函数把默认与显式覆写对齐到统一 dict。
    rolling_calibration = cfg.get("rolling_calibration", {}) or {}
    if not isinstance(rolling_calibration, dict):
        rolling_calibration = {}
    threshold_cfg = cfg.get("threshold_calibration", {}) or {}
    if not isinstance(threshold_cfg, dict):
        threshold_cfg = {}
    step_months = int(rolling_calibration.get("step_months", 1) or 1)
    if step_months <= 0:
        raise ValueError("rolling_calibration.step_months 必须 > 0")

    def _enabled(section_name: str, default: bool = True) -> bool:
        sec = rolling_calibration.get(section_name, {}) or {}
        if isinstance(sec, dict):
            return bool(sec.get("enabled", default))
        root_sec = threshold_cfg.get(section_name, {}) or {}
        if isinstance(root_sec, dict):
            return bool(root_sec.get("enabled", default))
        return bool(default)

    prefilter_cfg = rolling_calibration.get("prefilter", {}) or {}
    if not isinstance(prefilter_cfg, dict):
        prefilter_cfg = {}
    root_prefilter_cfg = threshold_cfg.get("prefilter", {}) or {}
    if not isinstance(root_prefilter_cfg, dict):
        root_prefilter_cfg = {}
    prefilter_optimize = bool(
        prefilter_cfg.get("optimize", root_prefilter_cfg.get("optimize", True))
    )

    threshold_fast_cfg = rolling_calibration.get("threshold_calibration", {}) or {}
    if not isinstance(threshold_fast_cfg, dict):
        threshold_fast_cfg = {}
    enable_mt = rolling_calibration.get("enable_model_training", None)
    if enable_mt is None:
        enable_mt = threshold_fast_cfg.get("enable_model_training", None)
    if enable_mt is None:
        enable_mt = threshold_cfg.get("enable_model_training", None)
    if enable_mt is None:
        enable_mt = True

    cfg["rolling_calibration"] = {
        "step_months": step_months,
        "threshold_calibration": {"enabled": _enabled("threshold_calibration", True)},
        "prefilter": {"optimize": prefilter_optimize},
        "symbol_threshold_calibration": {
            "enabled": _enabled("symbol_threshold_calibration", True)
        },
        "execution_opt": {"enabled": _enabled("execution_opt", True)},
        "pcm_eval": {"enabled": _enabled("pcm_eval", True)},
        "enable_model_training": bool(enable_mt),
    }
    for _fk, _fv in rolling_calibration.items():
        if _fk not in cfg["rolling_calibration"]:
            cfg["rolling_calibration"][_fk] = _fv
    return cfg


def resolve_symbols_from_config(cfg: dict) -> str:
    if "universe_group" in cfg:
        ug = cfg["universe_group"]
        ug_file = PROJECT_ROOT / ug["file"]
        ug_data = yaml.safe_load(ug_file.read_text(encoding="utf-8"))
        universe_set = ug["universe_set"]
        group = ug["group"]
        tokens = ug_data["universe_sets"][universe_set]["groups"][group]
        quote = ug_data.get("quote", "USDT")
        return ",".join(f"{t}{quote}" for t in tokens)
    if "symbols" in cfg:
        return cfg["symbols"]
    raise KeyError("research_pipeline.yaml 必须包含 universe_group 或 symbols 配置")


def compute_holdout_start(end_date: str, holdout_months: int) -> str:
    end = datetime.strptime(end_date, "%Y-%m-%d")
    y = end.year
    m = end.month - int(holdout_months)
    while m <= 0:
        y -= 1
        m += 12
    return f"{y:04d}-{m:02d}-01"


def resolve_strategy_dates(
    cfg: Dict[str, Any],
    *,
    strategy: str,
    default_end_date: str,
    forced_end_date: str = "",
) -> Dict[str, Any]:
    global_dates = cfg.get("dates", {})
    scfg = cfg["strategies"][strategy]
    strat_dates = (
        scfg.get("dates", {}) if isinstance(scfg.get("dates", {}), dict) else {}
    )

    end_date = forced_end_date or str(strat_dates.get("end_date", default_end_date))
    start_date = str(strat_dates.get("start_date", global_dates["start_date"]))
    holdout_months = int(
        strat_dates.get("holdout_months", global_dates["holdout_months"])
    )
    validation_months = int(
        strat_dates.get("validation_months", global_dates.get("validation_months", 0))
    )

    holdout_start = compute_holdout_start(end_date, holdout_months)
    if validation_months > 0 and validation_months < holdout_months:
        test_start = compute_holdout_start(end_date, holdout_months - validation_months)
    else:
        test_start = holdout_start

    return {
        "start_date": start_date,
        "end_date": end_date,
        "holdout_months": holdout_months,
        "validation_months": validation_months,
        "holdout_start": holdout_start,
        "test_start": test_start,
    }


def iter_month_tokens(start_date: str, end_date: str) -> List[str]:
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    cur_y, cur_m = s.year, s.month
    out: List[str] = []
    while (cur_y, cur_m) <= (e.year, e.month):
        out.append(f"{cur_y:04d}-{cur_m:02d}")
        if cur_m == 12:
            cur_y += 1
            cur_m = 1
        else:
            cur_m += 1
    return out
