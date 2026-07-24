#!/usr/bin/env python3
"""Train the camera-guided teacher with exact resume semantics."""
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from camera_operator_sr.data.collate import collate_frames
from camera_operator_sr.data.dataset import ProcessedRangeDataset
from camera_operator_sr.losses.total_loss import LossWeights
from camera_operator_sr.models.teacher import CameraGuidedOperatorTeacher
from camera_operator_sr.training.checkpoint import (extract_dataset_geometry, load_project_checkpoint,
    save_checkpoint, validate_checkpoint_geometry)
from camera_operator_sr.training.experiment import (append_jsonl, assert_resume_compatible, build_manifest,
    invocation_record, prepare_experiment, source_checkpoint_metadata, write_json_atomic)
from camera_operator_sr.training.modules import TeacherModule, generated_mask_for, teacher_masks_for
from camera_operator_sr.training.reproducibility import capture_rng_state, dataloader_generator, seed_everything
from camera_operator_sr.training.resume import is_historical_best, restore_training_state
from camera_operator_sr.training.validation import ValidationRangeAccumulator


def _move(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    return {key: _move(item, device) for key, item in value.items()} if isinstance(value, dict) else value


def _arguments(args): return {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset-root", required=True); parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--depth-mode", choices=("correct", "frame_shuffled", "spatial_shuffled", "constant", "none", "oracle"), default="correct")
    parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--experiment-name", required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deterministic", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-split"); parser.add_argument("--val-split"); parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    baseline = load_project_checkpoint(args.baseline_checkpoint, map_location=args.device)
    model = CameraGuidedOperatorTeacher(**baseline["model_config"]).to(args.device)
    dataset = ProcessedRangeDataset(args.dataset_root, split_file=args.train_split, depth_mode=args.depth_mode)
    validation = ProcessedRangeDataset(args.dataset_root, split_file=args.val_split, depth_mode=args.depth_mode) if args.val_split else None
    geometry = extract_dataset_geometry(dataset[0], candidate_horizontal_radius=model.horizontal_radius).as_dict()
    validate_checkpoint_geometry(baseline, geometry)
    common = {key: value for key, value in baseline["model"].items() if key.startswith(("encoder.", "decoder."))}
    expected = {key: value for key, value in model.state_dict().items() if key.startswith(("encoder.", "decoder."))}
    if set(common) != set(expected) or any(value.shape != expected[key].shape for key, value in common.items()):
        raise ValueError("baseline checkpoint is incompatible with teacher LiDAR branch")
    model.load_state_dict(common, strict=False)
    weights = LossWeights(); loss_config = asdict(weights); assert loss_config
    paths = prepare_experiment(output_root=args.output_root, experiment_name=args.experiment_name, seed=args.seed, resume=args.resume, overwrite=args.overwrite)
    requested_manifest = build_manifest(experiment_name=args.experiment_name, experiment_type="teacher", seed=args.seed,
        deterministic=args.deterministic, dataset_root=args.dataset_root, train_split=args.train_split, validation_split=args.val_split,
        train_frames=len(dataset), validation_frames=len(validation) if validation else 0, model_config=model.model_config,
        geometry=geometry, depth_mode=args.depth_mode, advantage_config=None)
    sources = source_checkpoint_metadata(baseline=args.baseline_checkpoint)
    requested_manifest["source_checkpoints"] = sources; requested_manifest["loss_config"] = loss_config
    cli_arguments = _arguments(args)
    if args.resume:
        manifest = json.loads(paths.manifest.read_text()); assert_resume_compatible(manifest, requested_manifest)
        experiment_config = json.loads(paths.config.read_text())
        if experiment_config.get("loss_config") != loss_config: raise ValueError("Resume configuration mismatch: loss_config")
        invocation_index = len(paths.invocations.read_text().splitlines())
    else:
        manifest = requested_manifest; experiment_config = dict(cli_arguments, loss_config=loss_config); invocation_index = 0
        write_json_atomic(paths.manifest, manifest); write_json_atomic(paths.config, experiment_config)
    current_invocation = invocation_record(invocation_index=invocation_index, resume=args.resume, overwrite=args.overwrite, arguments=cli_arguments)
    append_jsonl(paths.invocations, current_invocation)
    metadata = {"experiment_name": args.experiment_name, "experiment_type": "teacher", "seed": args.seed, "deterministic": args.deterministic,
                "run_config": experiment_config, "experiment_config": experiment_config, "current_invocation": current_invocation, "manifest": manifest}
    module = TeacherModule(model, weights); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate); generator = dataloader_generator(args.seed)
    start, step, best, best_epoch, best_step = 0, 0, float("inf"), 0, 0
    if args.resume:
        state = restore_training_state(load_project_checkpoint(paths.last_checkpoint, map_location=args.device), model=model,
            optimizer=optimizer, dataloader_generator=generator, experiment_type="teacher")
        start, step, best, best_epoch, best_step = state.start_epoch, state.global_step, state.best_validation_score, state.best_epoch, state.best_global_step
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_frames, generator=generator)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate_frames) if validation else None
    for epoch in range(start, args.epochs):
        model.train(); total = 0.0
        for batch in loader:
            loss = module(_move(batch, args.device)); optimizer.zero_grad(set_to_none=True); loss["loss"].backward(); optimizer.step(); step += 1; total += float(loss["loss"].detach())
        score = count = None; is_best = False
        if validation_loader:
            model.eval(); accumulator = ValidationRangeAccumulator()
            with torch.no_grad():
                for batch in validation_loader:
                    batch = _move(batch, args.device); output = model(batch); visible, _ = teacher_masks_for(batch)
                    accumulator.update(output.predicted_range, batch["target"]["range"], generated_mask_for(batch) * batch["target"]["valid"] * visible * output.has_candidate.to(batch["target"]["valid"].dtype))
            score, count = accumulator.score(), accumulator.count
            if not math.isfinite(score) or count == 0: raise ValueError("validation score must be finite with non-zero count")
            is_best = is_historical_best(score, best)
            if is_best: best, best_epoch, best_step = score, epoch + 1, step
        checkpoint = dict(epoch=epoch + 1, global_step=step, sample=dataset[0], optimizer=optimizer, dataset_split=args.train_split,
            depth_mode=args.depth_mode, validation_score=score, validation_count=count, experiment_metadata=metadata, loss_config=loss_config,
            rng_state=capture_rng_state(generator), best_validation_score=best, best_epoch=best_epoch, best_global_step=best_step, source_checkpoints=sources)
        save_checkpoint(paths.last_checkpoint, model, **checkpoint)
        if is_best: save_checkpoint(paths.best_checkpoint, model, **checkpoint)
        append_jsonl(paths.metrics, {"epoch": epoch, "global_step": step, "training_loss": total / max(len(loader), 1),
            "validation_metric": "global_query_weighted_range_mae", "validation_score": score, "validation_count": count, "is_best": is_best})
        score_text = f"{score:.6f}" if score is not None else "n/a"
        print(f"[teacher:{args.depth_mode}] epoch {epoch + 1}/{args.epochs} step={step} loss={total / max(len(loader), 1):.6f} val_mae={score_text} best_mae={best:.6f} best={is_best}", flush=True)


if __name__ == "__main__": main()
