"""Test d'intégration minimal : le script d'évaluation doit s'exécuter sans erreur."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_evaluate_dry_run_exits_zero():
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "evaluate.py"),
            "--dry-run",
            "--limit",
            "2",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_root_evaluate_delegates():
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "evaluate.py"),
            "--dry-run",
            "--limit",
            "1",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
