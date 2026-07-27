from types import SimpleNamespace

import torch

from camera_operator_sr.evaluation.relation_metrics import RelationMetricAccumulator, anchor_selection_statistics


def relation_output(*, anchor_valid, final_weights, correction, anchor_ranges=None):
    anchor_valid = torch.tensor(anchor_valid, dtype=torch.bool).reshape(1, 1, -1, 2)
    width = anchor_valid.shape[-2]
    anchor_ranges = torch.tensor(anchor_ranges or [[8., 12.]] * width).reshape(1, 1, width, 2)
    final_weights = torch.tensor(final_weights).reshape(1, 1, width, 2)
    correction = torch.tensor(correction).reshape(1, 1, 1, width)
    prediction = (final_weights * anchor_ranges).sum(dim=-1)[:, None]
    return SimpleNamespace(
        prior_weights=torch.full_like(final_weights, 0.5), final_weights=final_weights,
        correction=correction, anchor_ranges=anchor_ranges, anchor_valid=anchor_valid,
        has_anchor=torch.ones(1, 1, 1, width, dtype=torch.bool), predicted_range=prediction,
    )


def test_correction_mean_uses_only_supported_two_anchor_queries():
    output = relation_output(
        anchor_valid=[[1, 1], [1, 0], [0, 1]],
        final_weights=[[.8, .2], [1., 0.], [0., 1.]], correction=[2., 100., -100.],
        anchor_ranges=[[8., 12.], [10., 0.], [0., 10.]],
    )
    accumulator = RelationMetricAccumulator()
    target = torch.tensor([[[[9., 10., 10.]]]])
    accumulator.update(output, target, torch.ones_like(target, dtype=torch.bool), torch.ones_like(target, dtype=torch.bool))
    values = accumulator.result()
    assert values["supported_count"] == 3
    assert values["both_anchor_count"] == 1
    assert values["mean_abs_correction"] == 2.0


def test_correction_mean_is_null_when_only_one_anchor_is_supported():
    output = relation_output(
        anchor_valid=[[1, 0], [0, 1]], final_weights=[[1., 0.], [0., 1.]], correction=[5., -7.],
        anchor_ranges=[[10., 0.], [0., 10.]],
    )
    accumulator = RelationMetricAccumulator()
    target = torch.tensor([[[[10., 10.]]]])
    accumulator.update(output, target, torch.ones_like(target, dtype=torch.bool), torch.ones_like(target, dtype=torch.bool))
    values = accumulator.result()
    assert values["supported_count"] == 2
    assert values["both_anchor_count"] == 0
    assert values["mean_abs_correction"] is None
    assert values["empty_group"] is False


def test_anchor_selection_excludes_final_weight_and_gt_error_ties_but_counts_clear_choice():
    target = torch.tensor([[[[9.]]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    final_tie = relation_output(anchor_valid=[[1, 1]], final_weights=[[.5, .5]], correction=[0.])
    gt_tie = relation_output(anchor_valid=[[1, 1]], final_weights=[[.8, .2]], correction=[0.], anchor_ranges=[[8., 10.]])
    clear = relation_output(anchor_valid=[[1, 1]], final_weights=[[.8, .2]], correction=[0.])
    assert anchor_selection_statistics(final_tie, target, mask) == (0, 0)
    assert anchor_selection_statistics(gt_tie, torch.tensor([[[[9.]]]]), mask) == (0, 0)
    assert anchor_selection_statistics(clear, target, mask) == (1, 1)
