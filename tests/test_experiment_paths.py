from camera_operator_sr.training.experiment import prepare_experiment
import pytest

def test_experiment_paths_are_isolated_and_safe(tmp_path):
    paths=prepare_experiment(output_root=tmp_path,experiment_name='student_baseline',seed=42,resume=False,overwrite=False)
    assert paths.root == tmp_path/'student_baseline'/'seed_42'
    assert paths.best_checkpoint != paths.last_checkpoint and paths.checkpoints.exists() and paths.logs.exists()
    for bad in ('',' ','../escape','a/b','/absolute'):
        with pytest.raises(ValueError): prepare_experiment(output_root=tmp_path,experiment_name=bad,seed=1,resume=False,overwrite=False)
