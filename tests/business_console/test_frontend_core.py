"""Frontend smoke tests — React SPA (Vitest) + built dist."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = REPO_ROOT / "frontend"
DIST_ROOT = REPO_ROOT / "src" / "mlbot_console" / "static" / "dist"


def test_spa_dist_built():
    index = DIST_ROOT / "index.html"
    assert index.is_file(), "Run: make frontend-build"
    html = index.read_text(encoding="utf-8")
    assert 'id="root"' in html
    assert "/static/assets/" in html


@pytest.mark.skipif(
    not (FRONTEND_ROOT / "package.json").is_file(),
    reason="frontend/ not present",
)
def test_frontend_vitest():
    proc = subprocess.run(
        ["npm", "test"],
        cwd=str(FRONTEND_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.skipif(
    subprocess.run(["which", "node"], capture_output=True).returncode != 0,
    reason="node not installed",
)
def test_frontend_typecheck():
    proc = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(FRONTEND_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
