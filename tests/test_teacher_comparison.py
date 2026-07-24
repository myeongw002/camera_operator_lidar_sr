from camera_operator_sr.evaluation.teacher_comparison import build_superiority

def test_teacher_superiority_reports_raw_delta():
    result=build_superiority([{ "model":"baseline","region":"gt_camera_visible","range_mae":2.0,"range_count":3},{"model":"teacher_correct","region":"gt_camera_visible","range_mae":1.0,"range_count":3}])
    assert result["correct_better_than_baseline"] and result["delta_mae_correct_minus_baseline"]==-1.0
