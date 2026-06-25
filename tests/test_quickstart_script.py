from __future__ import annotations

import subprocess
from pathlib import Path


def test_demo_quickstart_script_prints_usage_and_observability_sections() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "demo_quickstart.py"

    completed = subprocess.run(
        [str(repo_root / ".venv" / "Scripts" / "python.exe"), str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "=== SETUP ===" in completed.stdout
    assert "=== SUCCESS RESULT ===" in completed.stdout
    assert "=== FAILURE RESULT ===" in completed.stdout
    assert "=== QUERY RESULT ===" in completed.stdout
    assert "=== HEALTH ===" in completed.stdout
    assert "=== METRICS ===" in completed.stdout
    assert "=== PROMETHEUS ===" in completed.stdout
    assert "=== TRACES ===" in completed.stdout
    assert "demo-async" in completed.stdout
    assert "WORKER_UNAVAILABLE" in completed.stdout
