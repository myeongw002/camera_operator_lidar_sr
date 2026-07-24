import subprocess
import sys

from camera_operator_sr.pipeline.runner import PipelineRunner


def test_process_group_termination_stops_active_stage_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    PipelineRunner._terminate_process_group(process)
    assert process.wait(timeout=5) != 0
