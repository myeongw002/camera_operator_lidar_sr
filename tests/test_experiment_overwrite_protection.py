import pytest
from camera_operator_sr.training.experiment import prepare_experiment, write_json_atomic

def test_existing_experiment_requires_explicit_resume_or_overwrite(tmp_path):
    first=prepare_experiment(output_root=tmp_path,experiment_name='x',seed=1,resume=False,overwrite=False); write_json_atomic(first.manifest,{'old':True}); first.last_checkpoint.write_text('old')
    with pytest.raises(FileExistsError): prepare_experiment(output_root=tmp_path,experiment_name='x',seed=1,resume=False,overwrite=False)
    assert first.last_checkpoint.read_text()=='old'
    with pytest.raises(ValueError): prepare_experiment(output_root=tmp_path,experiment_name='x',seed=1,resume=True,overwrite=True)
    with pytest.warns(UserWarning): overwritten=prepare_experiment(output_root=tmp_path,experiment_name='x',seed=1,resume=False,overwrite=True)
    assert not overwritten.last_checkpoint.exists()
