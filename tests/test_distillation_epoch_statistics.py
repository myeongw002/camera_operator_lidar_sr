import json

from camera_operator_sr.training.statistics import (add_distillation_statistics, empty_distillation_statistics,
    finalize_distillation_statistics)


def test_query_weighted_statistics_do_not_average_batch_means():
    totals = empty_distillation_statistics()
    # Two batches have 100 and 1 eligible query respectively: mean-of-means would be wrong.
    add_distillation_statistics(totals, {key: __import__("torch").tensor(value) for key, value in {
        "range_advantage_sum": 100, "range_advantage_count": 100, "return_advantage_sum": 100,
        "return_advantage_count": 100, "range_kd_active_count": 100, "range_kd_eligible_count": 100,
        "return_kd_active_count": 100, "return_kd_eligible_count": 100}.items()})
    add_distillation_statistics(totals, {key: __import__("torch").tensor(value) for key, value in {
        "range_advantage_sum": 0, "range_advantage_count": 1, "return_advantage_sum": 0,
        "return_advantage_count": 1, "range_kd_active_count": 0, "range_kd_eligible_count": 1,
        "return_kd_active_count": 0, "return_kd_eligible_count": 1}.items()})
    row = finalize_distillation_statistics(totals)
    assert row["mean_range_advantage"] == 100 / 101
    assert row["range_kd_active_ratio"] == 100 / 101
    assert row["mean_range_advantage"] != .5


def test_zero_eligible_statistics_are_explicit_nulls(tmp_path):
    row = finalize_distillation_statistics(empty_distillation_statistics())
    path = tmp_path / "metrics.jsonl"; path.write_text(json.dumps(row) + "\n")
    stored = json.loads(path.read_text())
    assert stored["mean_range_advantage"] is None and stored["range_advantage_count"] == 0
