from __future__ import annotations

import subprocess
from pathlib import Path


def test_demo_observability_script_prints_all_sections() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "demo_observability.py"

    completed = subprocess.run(
        [str(repo_root / ".venv" / "Scripts" / "python.exe"), str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "=== TASK RESULT ===" in completed.stdout
    assert "=== TASK FAILURE ===" in completed.stdout
    assert "=== HEALTH ===" in completed.stdout
    assert "=== METRICS ===" in completed.stdout
    assert "=== PROMETHEUS ===" in completed.stdout
    assert "=== TRACES ===" in completed.stdout
    assert "submitted_total=2" in completed.stdout
    assert "completed_total=1" in completed.stdout
    assert "failed_total=1" in completed.stdout
    assert "task.submitted" in completed.stdout
    assert "task.completed" in completed.stdout
    assert "task.failed" in completed.stdout
