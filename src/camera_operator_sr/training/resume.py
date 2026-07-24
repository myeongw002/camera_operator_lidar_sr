"""Shared exact-resume validation and restoration for all training stages."""
from dataclasses import dataclass

from torch import nn
from torch.optim import Optimizer
import torch

from .checkpoint import CHECKPOINT_SCHEMA_VERSION
from .reproducibility import restore_rng_state


TRAINING_CHECKPOINT_KEYS = {
    "checkpoint_schema_version", "model", "optimizer", "epoch", "global_step",
    "geometry", "model_config", "loss_config", "validation_metric",
    "current_validation_score", "validation_count", "best_validation_score",
    "best_epoch", "best_global_step", "rng_state", "experiment_name",
    "experiment_type", "experiment_config", "current_invocation",
    "invocation_index", "manifest", "split_hashes", "source_checkpoints",
}


@dataclass(frozen=True)
class ResumeState:
    start_epoch: int
    global_step: int
    best_validation_score: float
    best_epoch: int
    best_global_step: int


def validate_training_checkpoint(checkpoint: dict, *, experiment_type: str) -> None:
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Resume requires checkpoint schema version 3")
    missing = sorted(TRAINING_CHECKPOINT_KEYS - checkpoint.keys())
    if missing:
        raise ValueError("Schema 3 checkpoint missing resume metadata: " + ", ".join(missing))
    if checkpoint["experiment_type"] != experiment_type:
        raise ValueError(
            f"Resume experiment type mismatch: {checkpoint['experiment_type']} != {experiment_type}"
        )
    if not checkpoint["loss_config"]:
        raise ValueError("Schema 3 checkpoint has empty loss_config")
    if checkpoint["rng_state"] is None:
        raise ValueError("Schema 3 checkpoint has no RNG state")


def restore_training_state(checkpoint: dict, *, model: nn.Module, optimizer: Optimizer,
                           dataloader_generator: torch.Generator, experiment_type: str) -> ResumeState:
    """Restore state before creating a DataLoader iterator."""
    validate_training_checkpoint(checkpoint, experiment_type=experiment_type)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(checkpoint["rng_state"], dataloader_generator)
    return ResumeState(
        start_epoch=int(checkpoint["epoch"]),
        global_step=int(checkpoint["global_step"]),
        best_validation_score=float(checkpoint["best_validation_score"]),
        best_epoch=int(checkpoint["best_epoch"]),
        best_global_step=int(checkpoint["best_global_step"]),
    )


def is_historical_best(current_validation_score: float, best_validation_score: float) -> bool:
    """MAE uses strict improvement so ties preserve the historical best file."""
    return current_validation_score < best_validation_score
