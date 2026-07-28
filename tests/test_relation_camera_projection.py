import torch

from camera_operator_sr.geometry.candidate_graph import build_candidate_index
from camera_operator_sr.geometry.candidate_projection import candidate_lidar_points


def test_candidate_projection_preserves_six_slots_and_wrapped_source_azimuth():
    index = build_candidate_index(torch.tensor([-.4, .4]), torch.tensor([0.0]), width=4)
    ranges = torch.ones(1, 1, 4, 6)
    valid = torch.ones_like(ranges, dtype=torch.bool)
    azimuth = torch.tensor([0.0, .2, .7, 1.5])
    xyz, mask = candidate_lidar_points(ranges, valid, index, torch.tensor([-.4, .4]), azimuth)
    # slots are lower[-1,0,+1], upper[-1,0,+1], so column zero gathers 3,0,1.
    expected = azimuth[torch.tensor([3, 0, 1, 3, 0, 1])]
    observed = torch.atan2(xyz[0, 0, 0, :, 1], xyz[0, 0, 0, :, 0])
    assert index.row_indices.tolist() == [[0, 0, 0, 1, 1, 1]]
    assert torch.all(mask)
    assert torch.allclose(observed, expected)
