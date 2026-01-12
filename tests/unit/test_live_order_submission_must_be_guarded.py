import os
import re
from pathlib import Path


def test_live_no_direct_submit_order_calls() -> None:
    """
    Regression guard: live code must submit orders via ExecutionManager to ensure
    enforce_before_order is not bypassed.
    """
    repo_root = Path(__file__).resolve().parents[2]
    live_dir = repo_root / "src" / "time_series_model" / "live"
    assert live_dir.exists(), f"missing live dir: {live_dir}"

    allow_files = {
        "execution_manager.py",
    }

    bad = []
    pat = re.compile(r"\.submit_order\(")

    for p in live_dir.glob("*.py"):
        if p.name in allow_files:
            continue
        txt = p.read_text(encoding="utf-8")
        if pat.search(txt):
            bad.append(str(p.relative_to(repo_root)))

    assert not bad, "Direct submit_order calls found in live code:\n" + "\n".join(bad)
