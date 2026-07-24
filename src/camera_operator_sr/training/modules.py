import torch
from torch import nn

from camera_operator_sr.data.masks import build_generated_row_mask
from camera_operator_sr.geometry.visibility import build_camera_query_frustum_mask, build_gt_visible_valid_mask
from camera_operator_sr.geometry.validation import assert_shared_geometry
from camera_operator_sr.losses.advantage_mask import AdvantageConfig, build_distillation_masks, compute_range_advantage, compute_return_advantage
from camera_operator_sr.losses.total_loss import LossWeights, distillation_total, supervised_total


def generated_mask_for(batch: dict) -> torch.Tensor:
    elevation = batch["lidar"]["elevation"]
    target_elevation = batch["target"]["elevation"]
    if elevation.ndim == 2:
        assert_shared_geometry(elevation, "input elevation")
    if target_elevation.ndim == 2:
        assert_shared_geometry(target_elevation, "target elevation")
    input_angles = elevation[0] if elevation.ndim == 2 else elevation
    target_angles = target_elevation[0] if target_elevation.ndim == 2 else target_elevation
    row_mask = build_generated_row_mask(input_angles, target_angles).to(batch["target"]["range"].device)
    return row_mask[None].expand(batch["target"]["range"].shape[0], 1, -1, batch["target"]["range"].shape[-1])


def teacher_masks_for(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GT-visible-valid and geometric frustum masks, respectively."""
    target = batch["target"]
    from camera_operator_sr.data.range_image import range_image_to_pointcloud
    elevation = target["elevation"][0] if target["elevation"].ndim == 2 else target["elevation"]
    azimuth = batch["lidar"]["azimuth"][0] if batch["lidar"]["azimuth"].ndim == 2 else batch["lidar"]["azimuth"]
    xyz = range_image_to_pointcloud(target["range"].squeeze(1), target["valid"].squeeze(1), elevation, azimuth)
    image_size = batch["camera"]["relative_depth"].shape[-2:]
    visible = build_gt_visible_valid_mask(xyz, target["valid"], batch["calibration"]["K"], batch["calibration"]["T_cam_lidar"], image_size)
    frustum = build_camera_query_frustum_mask(elevation, azimuth, batch["calibration"]["K"], batch["calibration"]["T_cam_lidar"], image_size)
    return visible, frustum


class StudentModule(nn.Module):
    def __init__(self, model: nn.Module, weights: LossWeights = LossWeights()):
        super().__init__()
        self.model, self.weights = model, weights

    def forward(self, batch: dict) -> dict:
        output = self.model(batch)
        generated = generated_mask_for(batch) * output.has_candidate.to(generated_mask_for(batch).dtype)
        return supervised_total(output, batch, generated, self.weights, return_mask=generated)


class TeacherModule(StudentModule):
    def forward(self, batch: dict) -> dict:
        output = self.model(batch)
        visible, frustum = teacher_masks_for(batch)
        generated = generated_mask_for(batch)
        eligible = output.has_candidate.to(generated.dtype)
        return supervised_total(output, batch, generated * visible * eligible, self.weights, return_mask=generated * frustum * eligible)


class DistillModule(StudentModule):
    def __init__(self, student: nn.Module, teacher: nn.Module, baseline: nn.Module, weights: LossWeights = LossWeights(), advantage: AdvantageConfig = AdvantageConfig()):
        super().__init__(student, weights)
        self.teacher, self.baseline = teacher.eval(), baseline.eval()
        self.advantage = advantage
        for frozen in (self.teacher, self.baseline):
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)

    def forward(self, batch: dict) -> dict:
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            baseline_output = self.baseline(batch)
        student_output = self.model(batch)
        generated = generated_mask_for(batch)
        target = batch["target"]
        visible, frustum = teacher_masks_for(batch)
        range_advantage = compute_range_advantage(baseline_output.predicted_range, teacher_output.predicted_range, target["range"], target["valid"], margin=self.advantage.range_margin, temperature=self.advantage.range_temperature, mode=self.advantage.mode)
        return_advantage = compute_return_advantage(baseline_output.return_logits, teacher_output.return_logits, target["valid"], margin=self.advantage.return_margin, temperature=self.advantage.return_temperature, mode=self.advantage.mode)
        masks = build_distillation_masks(generated, visible, frustum, target["valid"], range_advantage, return_advantage, student_output.has_candidate)
        supervised = generated * student_output.has_candidate.to(generated.dtype)
        values = distillation_total(student_output, teacher_output, batch, supervised, masks, self.weights)
        range_eligible = (generated * visible * target["valid"] * student_output.has_candidate.to(generated.dtype)).bool()
        return_eligible = (generated * frustum * student_output.has_candidate.to(generated.dtype)).bool()
        values.update(
            range_advantage_sum=(range_advantage * range_eligible).sum(), range_advantage_count=range_eligible.sum(),
            return_advantage_sum=(return_advantage * return_eligible).sum(), return_advantage_count=return_eligible.sum(),
            range_kd_active_count=(masks.operator.gt(0) & range_eligible).sum(), return_kd_active_count=(masks.return_mask.gt(0) & return_eligible).sum(),
            range_kd_eligible_count=range_eligible.sum(), return_kd_eligible_count=return_eligible.sum(),
            mean_range_advantage=(range_advantage * range_eligible).sum() / range_eligible.sum().clamp_min(1), mean_return_advantage=(return_advantage * return_eligible).sum() / return_eligible.sum().clamp_min(1),
            range_kd_active_ratio=(masks.operator.gt(0) & range_eligible).sum() / range_eligible.sum().clamp_min(1), return_kd_active_ratio=(masks.return_mask.gt(0) & return_eligible).sum() / return_eligible.sum().clamp_min(1),
        )
        return values
