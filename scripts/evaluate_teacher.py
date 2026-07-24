#!/usr/bin/env python3
"""Compare baseline and optional teacher controls on an explicitly supplied test split."""
import argparse, csv, json, warnings
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from camera_operator_sr.data.collate import collate_frames
from camera_operator_sr.data.dataset import ProcessedTrainingDataset
from camera_operator_sr.evaluation.evaluator import RangeAccumulator, ReturnAccumulator, OperatorAccumulator
from camera_operator_sr.evaluation.operator_metrics import operator_metric_sums
from camera_operator_sr.evaluation.region_metrics import build_region_masks
from camera_operator_sr.evaluation.boundary_metrics import build_range_boundary_mask
from camera_operator_sr.evaluation.teacher_comparison import build_superiority
from camera_operator_sr.geometry.visibility import build_camera_query_frustum_mask, build_gt_visible_valid_mask
from camera_operator_sr.data.range_image import range_image_to_pointcloud
from camera_operator_sr.models.student import LidarOperatorStudent
from camera_operator_sr.models.teacher import CameraGuidedOperatorTeacher
from camera_operator_sr.training.checkpoint import extract_dataset_geometry, load_project_checkpoint, validate_checkpoint_geometry, validate_checkpoint_pair, validate_teacher_depth_mode
from camera_operator_sr.training.modules import generated_mask_for

def move(x, d): return x.to(d) if isinstance(x, torch.Tensor) else {k: move(v, d) for k,v in x.items()} if isinstance(x, dict) else x
def load(path, cls, device):
    checkpoint=load_project_checkpoint(path,map_location=device); model=cls(**checkpoint["model_config"]).to(device).eval(); model.load_state_dict(checkpoint["model"]); return model, checkpoint
def main():
    p=argparse.ArgumentParser(); p.add_argument("--baseline-checkpoint",type=Path,required=True); p.add_argument("--teacher-correct",type=Path,required=True); p.add_argument("--teacher-none",type=Path); p.add_argument("--teacher-frame-shuffled",type=Path); p.add_argument("--teacher-spatial-shuffled",type=Path); p.add_argument("--teacher-constant",type=Path); p.add_argument("--teacher-oracle",type=Path); p.add_argument("--dataset-root",type=Path,required=True); p.add_argument("--split-file",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--allow-depth-mode-mismatch",action="store_true"); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); a=p.parse_args()
    if not a.split_file.exists(): raise FileNotFoundError(a.split_file)
    specs=[("baseline",a.baseline_checkpoint,"none",LidarOperatorStudent),("teacher_correct",a.teacher_correct,"correct",CameraGuidedOperatorTeacher)]+[(f"teacher_{name}",getattr(a,f"teacher_{name}"),name,CameraGuidedOperatorTeacher) for name in ("none","frame_shuffled","spatial_shuffled","constant","oracle") if getattr(a,f"teacher_{name}")]
    baseline_metadata=load_project_checkpoint(a.baseline_checkpoint,map_location="cpu"); rows=[]
    model_modes={}
    for name,path,expected_depth_mode,cls in specs:
        model,checkpoint=load(path,cls,a.device); checkpoint_depth_mode=checkpoint.get("depth_mode")
        if checkpoint_depth_mode is None: raise ValueError(f"{name} checkpoint does not contain depth_mode metadata")
        if name != "baseline":
            validate_checkpoint_pair(checkpoint,baseline_metadata,left_name=name,right_name="baseline")
            try:
                validate_teacher_depth_mode(checkpoint,expected_depth_mode,allow_mismatch=False)
            except ValueError as error:
                message=f"Teacher depth mode mismatch\nmodel slot: {name}\ncheckpoint depth mode: {checkpoint_depth_mode}\nexpected evaluation depth mode: {expected_depth_mode}"
                if not a.allow_depth_mode_mismatch: raise ValueError(message) from error
                warnings.warn("Depth mode mismatch was explicitly allowed.\n\n" + message.replace("expected ", ""), UserWarning, stacklevel=1)
        model_modes[name]={"checkpoint_depth_mode":checkpoint_depth_mode,"evaluation_depth_mode":expected_depth_mode}
        dataset=ProcessedTrainingDataset(a.dataset_root,split_file=a.split_file,depth_mode=expected_depth_mode); geometry=extract_dataset_geometry(dataset[0],candidate_horizontal_radius=model.horizontal_radius); validate_checkpoint_geometry(checkpoint,geometry); metrics={}
        with torch.no_grad():
            for batch in DataLoader(dataset,batch_size=1,collate_fn=collate_frames):
                batch=move(batch,a.device); validate_checkpoint_geometry(checkpoint,batch); output=model(batch); generated=generated_mask_for(batch); target=batch["target"]; size=batch["calibration"]["image_size"][0].tolist(); frustum=build_camera_query_frustum_mask(target["elevation"],batch["lidar"]["azimuth"],batch["calibration"]["K"],batch["calibration"]["T_cam_lidar"],tuple(size)) if min(size)>0 else None
                elevation=target["elevation"][0] if target["elevation"].ndim==2 else target["elevation"]; azimuth=batch["lidar"]["azimuth"][0] if batch["lidar"]["azimuth"].ndim==2 else batch["lidar"]["azimuth"]
                xyz=range_image_to_pointcloud(target["range"].squeeze(1),target["valid"].squeeze(1),elevation,azimuth)
                visible=build_gt_visible_valid_mask(xyz,target["valid"],batch["calibration"]["K"],batch["calibration"]["T_cam_lidar"],tuple(size)) if min(size)>0 else torch.zeros_like(target["valid"])
                boundary=build_range_boundary_mask(target["range"],target["valid"])
                for region,mask in build_region_masks(batch["lidar"]["azimuth"],camera_frustum=frustum,gt_camera_visible=visible,camera_boundary=boundary).items():
                    r,v,o=metrics.setdefault(region,(RangeAccumulator(),ReturnAccumulator(),OperatorAccumulator())); evaluation=generated*mask.to(generated.device); r.update(output.predicted_range,target["range"],evaluation*target["valid"]); v.update(output.return_probability,target["valid"],evaluation); o.update(operator_metric_sums(output.anchor_range,output.predicted_range,output.residual,output.local_scale,output.candidate_ranges,output.candidate_valid,output.anchor_weights,target["range"],target["valid"],evaluation))
        for region,(r,v,o) in metrics.items(): rows.append({"model":name,"depth_mode":expected_depth_mode,"checkpoint_depth_mode":checkpoint_depth_mode,"evaluation_depth_mode":expected_depth_mode,"region":region}|r.result()|v.result()|o.result())
    a.output_root.mkdir(parents=True,exist_ok=True); fields=sorted({k for row in rows for k in row});
    with (a.output_root/"teacher_comparison.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
    superiority=build_superiority(rows); superiority["models"]=model_modes
    (a.output_root/"teacher_superiority.json").write_text(json.dumps(superiority,indent=2)+"\n")
    for name in ("beam_metrics.csv","distance_metrics.csv"): (a.output_root/name).write_text("Use evaluate_sr.py for detailed per-bin metrics.\n")
if __name__=="__main__": main()
