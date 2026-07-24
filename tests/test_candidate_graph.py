import torch

from camera_operator_sr.geometry.candidate_graph import build_candidate_index, gather_candidate_values


def test_candidate_graph_wrap_and_order():
    index = build_candidate_index(torch.tensor([-1.0, 1.0]), torch.tensor([0.0]), width=4)
    assert index.column_offsets.tolist() == [-1, 0, 1, -1, 0, 1]
    values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
    gathered = gather_candidate_values(values, index)
    assert gathered.shape == (1, 1, 4, 6, 1)
    assert gathered[0, 0, 0, :, 0].tolist() == [3.0, 0.0, 1.0, 7.0, 4.0, 5.0]
