#!/usr/bin/env python3
"""DD experiment matrix runner with state cleanup and data preloading."""

import copy, subprocess, sys, yaml, glob, os, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "live/highcap/config/constitution/constitution.yaml"
base = yaml.safe_load(BASE.read_text())

SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
CHOP = (
    "config/experiments/20260613_multileg_sizing_validate/variants/chop_prod/meta.yaml"
)
TREND = (
    "config/experiments/20260613_multileg_sizing_validate/variants/trend_prod/meta.yaml"
)
PRELOAD = Path("/tmp/bt_preload.pkl")


def clean_state():
    for f in glob.glob("/tmp/bt_chop_*.json") + glob.glob("/tmp/bt_trend_*.json"):
        try:
            os.remove(f)
        except Exception:
            pass


def run_one(label, max_dd, daily_loss, seg_dd, time_filter, use_preload):
    clean_state()
    cfg = copy.deepcopy(base)
    cfg["kill_switch"]["max_dd"] = max_dd
    cfg["kill_switch"]["daily_loss_limit"] = daily_loss
    cfg["multi_leg"]["sizing"]["chop_grid"]["segment_dd_target"] = seg_dd
    cfg["multi_leg"]["sizing"]["trend_scalp"]["segment_dd_target"] = seg_dd
    tmp = Path(f"/tmp/const_exp_{label}.yaml")
    tmp.write_text(yaml.dump(cfg))

    cmd = [
        sys.executable,
        "scripts/backtest_multileg_timeline.py",
        "--start",
        "2025-12-01",
        "--end",
        "2026-05-31",
        "--symbols",
        SYMBOLS,
        "--chop-config",
        CHOP,
        "--trend-config",
        TREND,
        "--constitution-yaml",
        str(tmp),
        "--equity",
        "50000",
    ]
    if time_filter:
        cmd.append("--trend-time-filter")
    if use_preload and PRELOAD.exists():
        cmd.extend(["--load-preload", str(PRELOAD)])

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    stdout = r.stdout
    if r.returncode != 0:
        print(f"  [{label}] ERROR rc={r.returncode}", flush=True)
        if r.stderr:
            print(f"  STDERR: {r.stderr[-300:]}", flush=True)
        return None

    eq_match = re.search(r"([\d,]+)\s*→\s*([\d,]+)\s*\(([+\-][\d.]+)%\)", stdout)
    dd_match = re.search(r"Max DD:\s*([+\-][\d.]+)%", stdout)
    trades_match = re.search(r"(\d+)\s*ok\s*/\s*(\d+)\s*rej", stdout)
    halted_match = re.search(r"Halted:\s*(\w+)", stdout)
    halt_reason_match = re.search(r"Halted:\s*\w+\s+(.+)", stdout)

    eq_end = float(eq_match.group(2).replace(",", "")) if eq_match else 0
    ret = float(eq_match.group(3)) if eq_match else 0
    dd = float(dd_match.group(1)) if dd_match else 0
    trades_ok = int(trades_match.group(1)) if trades_match else 0
    trades_rej = int(trades_match.group(2)) if trades_match else 0
    halted = halted_match.group(1) if halted_match else "?"
    reason = halt_reason_match.group(1).strip() if halt_reason_match else ""

    return {
        "label": label,
        "eq_end": eq_end,
        "ret": ret,
        "max_dd": dd,
        "trades_ok": trades_ok,
        "trades_rej": trades_rej,
        "halted": halted == "True",
        "halt_reason": reason,
        "max_dd_cfg": max_dd,
        "daily_loss_cfg": daily_loss,
        "seg_dd_cfg": seg_dd,
        "time_filter": time_filter,
    }


# ── Step 1: Generate preload if needed ──
if not PRELOAD.exists():
    print("=== Generating preload data (one-time, ~1-2 min) ===", flush=True)
    clean_state()
    r = subprocess.run(
        [
            sys.executable,
            "scripts/backtest_multileg_timeline.py",
            "--start",
            "2025-12-01",
            "--end",
            "2026-05-31",
            "--symbols",
            SYMBOLS,
            "--chop-config",
            CHOP,
            "--trend-config",
            TREND,
            "--constitution-yaml",
            str(BASE),
            "--equity",
            "50000",
            "--save-preload",
            str(PRELOAD),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        print(f"FATAL: preload generation failed: {r.stderr[-500:]}")
        sys.exit(1)
    print("Preload generated.\n", flush=True)
else:
    print(f"Using cached preload: {PRELOAD}\n", flush=True)

# ── Step 2: Run all variants ──
variants = [
    ("prod", 0.20, 0.06, 0.072, False),
    ("dd_half", 0.10, 0.06, 0.072, False),
    ("dd15", 0.15, 0.06, 0.072, False),
    ("loss8pct", 0.20, 0.08, 0.072, False),
    ("loss10pct", 0.20, 0.10, 0.072, False),
    ("dd_half_loss8", 0.10, 0.08, 0.072, False),
    ("sz_half", 0.20, 0.06, 0.036, False),
    ("sz_half_dd_half", 0.10, 0.06, 0.036, False),
    ("prod_tfilter", 0.20, 0.06, 0.072, True),
    ("dd_half_tfilter", 0.10, 0.06, 0.072, True),
    ("sz_half_tfilter", 0.20, 0.06, 0.036, True),
]

print(
    f"{'Label':<24s} {'Eq End':>10s} {'Ret':>8s} {'MaxDD':>8s} {'Trades':>10s} {'Halted':>6s} {'Note'}"
)
print("-" * 95, flush=True)

rows = []
for label, max_dd, daily_loss, seg_dd, tfilter in variants:
    print(f"  Running {label}...", end=" ", flush=True)
    r = run_one(label, max_dd, daily_loss, seg_dd, tfilter, use_preload=True)
    if r is None:
        print("FAILED", flush=True)
        continue
    rows.append(r)
    note_parts = []
    if r["halted"]:
        note_parts.append(r["halt_reason"][:40])
    if r["time_filter"]:
        note_parts.append("time_filter")
    note = " ".join(note_parts)
    print(f"done (eq={r['eq_end']:,.0f}, dd={r['max_dd']:+.1f}%)", flush=True)

print("\n" + "=" * 95)
print(
    f"{'Label':<24s} {'Eq End':>10s} {'Ret':>8s} {'MaxDD':>8s} {'Trades':>10s} {'Halted':>6s} {'Note'}"
)
print("-" * 95)
for r in rows:
    note = ""
    if r["halted"]:
        note = r["halt_reason"][:30]
    if r["time_filter"]:
        note = (note + " tf").strip()
    print(
        f"{r['label']:<24s} {r['eq_end']:>10,.0f} {r['ret']:>+7.1f}% {r['max_dd']:>+7.1f}% "
        f"{r['trades_ok']:>5d}/{r['trades_rej']:<5d} {'HALT' if r['halted'] else 'OK':>6s} {note}"
    )

print("\n=== DONE ===", flush=True)
