import torch

from camera_operator_sr.geometry.candidate_graph import build_candidate_index, gather_candidate_values


def test_relation_candidate_slots_are_fixed_lower_then_upper_centers_with_wrapping():
    index = build_candidate_index(torch.tensor([-1.0, 1.0]), torch.tensor([0.0]), width=4, horizontal_radius=1)
    assert index.column_offsets.tolist() == [-1, 0, 1, -1, 0, 1]
    assert (index.lower_center_slot, index.upper_center_slot) == (1, 4)
    values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
    gathered = gather_candidate_values(values, index)
    assert gathered[0, 0, 0, :, 0].tolist() == [3.0, 0.0, 1.0, 7.0, 4.0, 5.0]
    assert gathered[0, 0, 3, :, 0].tolist() == [2.0, 3.0, 0.0, 6.0, 7.0, 4.0]
    assert gathered[0, 0, 0, index.lower_center_slot, 0].item() == 0.0
    assert gathered[0, 0, 0, index.upper_center_slot, 0].item() == 4.0
