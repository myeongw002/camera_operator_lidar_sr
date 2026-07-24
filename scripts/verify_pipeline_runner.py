#!/usr/bin/env python3
"""Run the focused pipeline-runner regression suite without optional plugins."""
import os
import subprocess
import sys
from pathlib import Path

TESTS = ["tests/test_pipeline_config.py", "tests/test_pipeline_stage_graph.py", "tests/test_pipeline_command_building.py", "tests/test_pipeline_state.py", "tests/test_pipeline_resume.py", "tests/test_pipeline_failure_handling.py", "tests/test_pipeline_dry_run.py", "tests/test_pipeline_artifact_wiring.py", "tests/test_pipeline_summary.py", "tests/test_pipeline_end_to_end_synthetic.py", "tests/test_pipeline_force_training_stage.py", "tests/test_pipeline_failed_stage_recovery.py", "tests/test_pipeline_frame_limit.py", "tests/test_pipeline_preprocessing_validation.py", "tests/test_depth_precompute_resume.py", "tests/test_pipeline_dry_run_details.py", "tests/test_pipeline_preflight_kitti.py", "tests/test_pipeline_config_usage.py", "tests/test_pipeline_scientific_summary.py", "tests/test_pipeline_evaluation_output_validation.py", "tests/test_pipeline_inference_output_validation.py", "tests/test_pipeline_state_transition.py", "tests/test_pipeline_live_subprocess_logging.py", "tests/test_pipeline_elevation_estimation.py"]

def main() -> int:
    missing = [path for path in TESTS if not Path(path).exists()]
    if missing:
        print("PIPELINE_RUNNER_VERIFICATION: FAIL", file=sys.stderr); return 1
    environment = dict(os.environ); environment["PYTHONPATH"] = "src" + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment.setdefault("OMP_NUM_THREADS", "1"); environment.setdefault("MKL_NUM_THREADS", "1")
    # Keep this dependency-free while allowing the expensive independent
    # synthetic subprocess integrations to run concurrently.  Each test has a
    # pytest tmp_path, so it has no shared output directory with another group.
    expensive = {
        "tests/test_pipeline_end_to_end_synthetic.py",
        "tests/test_pipeline_force_training_stage.py",
        "tests/test_pipeline_failed_stage_recovery.py",
    }
    groups = [[path for path in TESTS if path not in expensive], *([path] for path in sorted(expensive))]
    processes = [subprocess.Popen([sys.executable, "-m", "pytest", *group, "-q"], env=environment) for group in groups]
    codes = [process.wait() for process in processes]
    if any(codes):
        print("PIPELINE_RUNNER_VERIFICATION: FAIL", file=sys.stderr); return next(code for code in codes if code) or 1
    print("PIPELINE_RUNNER_VERIFICATION: PASS"); return 0

if __name__ == "__main__": raise SystemExit(main())
