import torch

from camera_operator_sr.geometry.candidate_graph import build_candidate_index
from camera_operator_sr.models.operator_decoder import OperatorDecoder


def test_all_invalid_candidates_are_safe():
    decoder = OperatorDecoder(feature_dim=4, hidden_dim=8)
    output = decoder(torch.zeros(1, 4, 2, 8), torch.zeros(1, 1, 2, 8), torch.zeros(1, 1, 2, 8), build_candidate_index(torch.tensor([-1.0, 1.0]), torch.tensor([0.0]), 8))
    assert torch.isfinite(output.predicted_range).all()
    assert output.anchor_weights.eq(0).all()
    assert output.return_probability.lt(1e-6).all()
