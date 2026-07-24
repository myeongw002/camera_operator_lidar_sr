import torch


@torch.no_grad()
def fuse_observed_rows(output, batch: dict, exact_tolerance: float = 1e-5):
    """Preserve real 16-beam observations in exported 64-row predictions."""
    prediction, probability = output.predicted_range.clone(), output.return_probability.clone()
    input_elevation = batch["lidar"]["elevation"]
    target_elevation = batch["target"]["elevation"]
    if input_elevation.ndim == 1:
        input_elevation = input_elevation[None]
    if target_elevation.ndim == 1:
        target_elevation = target_elevation[None]
    for batch_index in range(prediction.shape[0]):
        target_index = (target_elevation[batch_index, :, None] - input_elevation[batch_index, None, :]).abs().argmin(dim=0)
        exact = (target_elevation[batch_index, target_index] - input_elevation[batch_index]).abs().le(exact_tolerance)
        rows = target_index[exact]
        source = torch.arange(input_elevation.shape[1], device=prediction.device)[exact]
        prediction[batch_index, :, rows] = batch["lidar"]["range"][batch_index, :, source]
        probability[batch_index, :, rows] = batch["lidar"]["valid"][batch_index, :, source]
    return prediction, probability
