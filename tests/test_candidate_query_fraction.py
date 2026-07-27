import pytest
import torch

from camera_operator_sr.geometry.candidate_graph import build_candidate_index


@pytest.mark.parametrize(
    ("input_elevation", "target_elevation", "expected"),
    [
        ([-1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.5, 0.0]),
        ([-2.0, -0.5, 3.0], [-1.25, 1.25], [0.5, 0.5]),
        ([3.0, -0.5, -2.0], [-1.25, 1.25], [0.5, 0.5]),
    ],
)
def test_query_fraction_uses_actual_elevation_for_ascending_and_descending_grids(input_elevation, target_elevation, expected):
    index = build_candidate_index(torch.tensor(input_elevation), torch.tensor(target_elevation), width=8)
    assert torch.allclose(index.query_fraction, torch.tensor(expected))


def test_exact_observed_row_has_one_source_and_zero_fraction():
    index = build_candidate_index(torch.tensor([1.0, -1.0]), torch.tensor([-1.0, 0.0, 1.0]), width=8)
    assert index.query_fraction.tolist() == [0.0, 0.5, 0.0]
    assert index.row_indices[0, 1].item() == index.row_indices[0, 4].item()
    assert not index.geometric_valid[0, 4]
