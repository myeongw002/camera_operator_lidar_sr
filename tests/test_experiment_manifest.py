from camera_operator_sr.training.experiment import build_manifest, hash_file_sha256

def test_manifest_tracks_split_paths_hashes_and_core_metadata(tmp_path):
    train=tmp_path/'train.txt'; val=tmp_path/'val.txt'; train.write_text('00/1\n'); val.write_text('01/2\n')
    manifest=build_manifest(experiment_name='student',experiment_type='student',seed=42,deterministic=True,dataset_root='data',train_split=str(train),validation_split=str(val),train_frames=1,validation_frames=1,model_config={'x':1},geometry={'width':4},depth_mode='none',advantage_config=None)
    assert manifest['schema_version']==1 and manifest['splits']['train']['sha256']==hash_file_sha256(train)
    old=manifest['splits']['train']['sha256']; train.write_text('00/changed\n'); assert hash_file_sha256(train)!=old
