from pathlib import Path

import numpy as np


def save_relative_depth(path: str | Path, depth: np.ndarray, valid: np.ndarray, model_name: str, input_resolution: tuple[int, int]) -> None:
    """Store unnormalised relative inverse depth and reproducibility metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth=depth.astype(np.float16), valid=valid.astype(np.uint8), model_name=model_name, input_resolution=np.asarray(input_resolution), output_type="relative_inverse_depth")


def load_relative_depth(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if str(data["output_type"]) != "relative_inverse_depth":
            raise ValueError("expected relative_inverse_depth")
        return {key: data[key] for key in data.files}
