"""Small, explicit reproducibility helpers for training entry points."""
import random
import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(); generator.manual_seed(seed); return generator


def capture_rng_state(dataloader_generator: torch.Generator) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "dataloader_generator": dataloader_generator.get_state()}


def restore_rng_state(state: dict, dataloader_generator: torch.Generator) -> None:
    required={"python","numpy","torch_cpu","torch_cuda","dataloader_generator"}
    if not required <= state.keys(): raise ValueError("Checkpoint RNG state is incomplete")
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"]); dataloader_generator.set_state(state["dataloader_generator"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None: torch.cuda.set_rng_state_all(state["torch_cuda"])
