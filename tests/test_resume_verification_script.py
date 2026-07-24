import os
import subprocess
import sys
from pathlib import Path


def test_verification_script_does_not_require_xdist():
    source = Path("scripts/verify_resume_reproducibility.py").read_text()
    assert '"-n"' not in source
    assert "'-n'" not in source


def test_resume_verification_script_runs_without_xdist():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "scripts/verify_resume_reproducibility.py"],
        cwd=".", env=environment, text=True, capture_output=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "RESUME_REPRODUCIBILITY_VERIFICATION: PASS" in result.stdout
