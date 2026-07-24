#!/usr/bin/env python3
"""Run the complete synthetic resume/reproducibility integration verification."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


TESTS = [
    "tests/test_schema3_checkpoint_loading.py",
    "tests/test_exact_resume_reproducibility.py",
    "tests/test_teacher_exact_resume.py",
    "tests/test_distillation_exact_resume.py",
    "tests/test_best_checkpoint_resume.py",
    "tests/test_source_checkpoint_resume.py",
    "tests/test_legacy_output_removed.py",
    "tests/test_distillation_epoch_statistics.py",
    "tests/test_loss_config_metadata.py",
    "tests/test_invocation_history.py",
]


def main() -> int:
    missing = [path for path in TESTS if not Path(path).exists()]
    if missing:
        for path in missing:
            print(f"Missing verification test: {path}", file=sys.stderr)
        print("RESUME_REPRODUCIBILITY_VERIFICATION: FAIL", file=sys.stderr)
        return 1
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src" + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    groups = [TESTS[::3], TESTS[1::3], TESTS[2::3]]
    processes = [subprocess.Popen([sys.executable, "-m", "pytest", *group, "-q"], env=environment) for group in groups]
    codes = [process.wait() for process in processes]
    if any(codes):
        print("RESUME_REPRODUCIBILITY_VERIFICATION: FAIL", file=sys.stderr)
        return next(code for code in codes if code) or 1
    print("RESUME_REPRODUCIBILITY_VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
