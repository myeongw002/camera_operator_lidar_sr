#!/usr/bin/env python3
"""Train the checkpointed L0 relation model without changing legacy training."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from camera_operator_sr.data.collate import collate_frames
from camera_operator_sr.data.dataset import ProcessedRangeDataset
from camera_operator_sr.losses.relation import relation_supervised_loss
from camera_operator_sr.models.relation import RelationLidarModel
from camera_operator_sr.training.checkpoint import extract_dataset_geometry, load_project_checkpoint, save_checkpoint
from camera_operator_sr.training.experiment import append_jsonl, assert_resume_compatible, build_manifest, invocation_record, prepare_experiment, write_json_atomic
from camera_operator_sr.training.modules import generated_mask_for
from camera_operator_sr.training.reproducibility import capture_rng_state, dataloader_generator, seed_everything
from camera_operator_sr.training.resume import is_historical_best, restore_training_state
from camera_operator_sr.training.validation import ValidationRangeAccumulator


def move(value, device): return value.to(device) if isinstance(value, torch.Tensor) else {key: move(item, device) for key, item in value.items()} if isinstance(value, dict) else value


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name, kwargs in (("--dataset-root", {"required": True}), ("--output-root", {"required": True, "type": Path}), ("--experiment-name", {"required": True}), ("--seed", {"required": True, "type": int})):
        parser.add_argument(name, **kwargs)
    parser.add_argument("--train-split"); parser.add_argument("--val-split")
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--horizontal-radius", type=int, default=1); parser.add_argument("--point-hidden-dim", type=int, default=24)
    parser.add_argument("--relation-hidden-dim", type=int, default=64); parser.add_argument("--correction-limit", type=float, default=3.0)
    parser.add_argument("--correction-reg-weight", type=float, default=1e-3); parser.add_argument("--use-intensity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    dataset = ProcessedRangeDataset(args.dataset_root, split_file=args.train_split, depth_mode="none")
    validation = ProcessedRangeDataset(args.dataset_root, split_file=args.val_split, depth_mode="none") if args.val_split else None
    model = RelationLidarModel(horizontal_radius=args.horizontal_radius, point_hidden_dim=args.point_hidden_dim, relation_hidden_dim=args.relation_hidden_dim, correction_limit=args.correction_limit, use_intensity=args.use_intensity).to(args.device)
    loss_config = {"huber_delta": 0.1, "correction_reg_weight": args.correction_reg_weight}
    geometry = extract_dataset_geometry(dataset[0], candidate_horizontal_radius=model.horizontal_radius).as_dict()
    paths = prepare_experiment(output_root=args.output_root, experiment_name=args.experiment_name, seed=args.seed, resume=args.resume, overwrite=args.overwrite)
    manifest = build_manifest(experiment_name=args.experiment_name, experiment_type="relation_l0", seed=args.seed, deterministic=args.deterministic, dataset_root=args.dataset_root, train_split=args.train_split, validation_split=args.val_split, train_frames=len(dataset), validation_frames=len(validation) if validation else 0, model_config=model.model_config, geometry=geometry, depth_mode="none", advantage_config=None)
    manifest["loss_config"] = loss_config; manifest["source_checkpoints"] = {}
    arguments = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    if args.resume:
        existing = json.loads(paths.manifest.read_text()); assert_resume_compatible(existing, manifest); manifest = existing
        config = json.loads(paths.config.read_text()); invocation_index = len(paths.invocations.read_text().splitlines())
    else:
        config = dict(arguments, loss_config=loss_config); write_json_atomic(paths.manifest, manifest); write_json_atomic(paths.config, config); invocation_index = 0
    invocation = invocation_record(invocation_index=invocation_index, resume=args.resume, overwrite=args.overwrite, arguments=arguments); append_jsonl(paths.invocations, invocation)
    metadata = {"experiment_name": args.experiment_name, "experiment_type": "relation_l0", "seed": args.seed, "deterministic": args.deterministic, "run_config": config, "experiment_config": config, "current_invocation": invocation, "manifest": manifest}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); generator = dataloader_generator(args.seed)
    start = step = best_epoch = best_step = 0; best = float("inf")
    if args.resume:
        state = restore_training_state(load_project_checkpoint(paths.last_checkpoint, map_location=args.device), model=model, optimizer=optimizer, dataloader_generator=generator, experiment_type="relation_l0")
        start, step, best, best_epoch, best_step = state.start_epoch, state.global_step, state.best_validation_score, state.best_epoch, state.best_global_step
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_frames, generator=generator)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate_frames) if validation else None
    for epoch in range(start, args.epochs):
        model.train(); total = 0.0
        for batch in loader:
            losses = relation_supervised_loss(model(move(batch, args.device)), move(batch, args.device), correction_reg_weight=args.correction_reg_weight)
            optimizer.zero_grad(set_to_none=True); losses["loss"].backward(); optimizer.step(); step += 1; total += float(losses["loss"].detach())
        score = count = None; is_best = False
        if validation_loader:
            model.eval(); accumulator = ValidationRangeAccumulator()
            with torch.no_grad():
                for batch in validation_loader:
                    batch = move(batch, args.device); output = model(batch); accumulator.update(output.predicted_range, batch["target"]["range"], generated_mask_for(batch) * batch["target"]["valid"] * output.has_anchor)
            score, count = accumulator.score(), accumulator.count
            if not math.isfinite(score) or count == 0: raise ValueError("validation score must be finite with non-zero supported count")
            is_best = is_historical_best(score, best)
            if is_best: best, best_epoch, best_step = score, epoch + 1, step
        common = dict(epoch=epoch + 1, global_step=step, sample=dataset[0], optimizer=optimizer, dataset_split=args.train_split, depth_mode="none", validation_score=score, validation_count=count, experiment_metadata=metadata, loss_config=loss_config, rng_state=capture_rng_state(generator), best_validation_score=best, best_epoch=best_epoch, best_global_step=best_step)
        save_checkpoint(paths.last_checkpoint, model, **common)
        if is_best: save_checkpoint(paths.best_checkpoint, model, **common)
        append_jsonl(paths.metrics, {"epoch": epoch + 1, "global_step": step, "training_loss": total / max(len(loader), 1), "validation_score": score, "validation_count": count, "is_best": is_best})


if __name__ == "__main__": main()
