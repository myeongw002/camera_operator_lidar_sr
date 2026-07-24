from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .split import load_split


class ProcessedTrainingDataset(Dataset):
    """Dataset contract for files emitted by prepare_range_images.

    Every frame directory contains target/input arrays and a `meta.npz` with
    elevation/azimuth/calibration. Camera depth arrays are optional for student use.
    """
    def __init__(self, root: str | Path, frame_dirs: list[str | Path] | None = None, split_file: str | Path | None = None, depth_mode: str = "none", shuffle_seed: int = 42, shuffle_different_sequence: bool = True, minimum_frame_gap: int = 100):
        self.root = Path(root)
        if frame_dirs and split_file:
            raise ValueError("provide frame_dirs or split_file, not both")
        self.frames = [Path(p) for p in frame_dirs] if frame_dirs else load_split(self.root, split_file) if split_file else sorted(self.root.glob("*/*"))
        self.frames = [path for path in self.frames if (path / "target_range.npy").exists()]
        if depth_mode not in {"correct", "frame_shuffled", "spatial_shuffled", "constant", "none", "oracle"}:
            raise ValueError(f"unsupported depth mode: {depth_mode}")
        self.depth_mode = depth_mode
        self.shuffle_seed, self.shuffle_different_sequence, self.minimum_frame_gap = shuffle_seed, shuffle_different_sequence, minimum_frame_gap
        self._frame_shuffle = self._build_frame_shuffle() if depth_mode == "frame_shuffled" else None
        if not self.frames:
            raise FileNotFoundError(f"no processed frames under {self.root}")

    @staticmethod
    def _load(path: Path, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.load(path / name), dtype=dtype)

    def __len__(self) -> int:
        return len(self.frames)

    def _build_frame_shuffle(self) -> list[int]:
        if len(self.frames) < 2: raise ValueError("frame_shuffled depth requires at least two frames")
        generator = torch.Generator().manual_seed(self.shuffle_seed)
        choices: list[int] = []
        for index, frame in enumerate(self.frames):
            sequence = frame.parent.name
            other_sequences = [candidate for candidate, path in enumerate(self.frames) if path.parent.name != sequence]
            if other_sequences and self.shuffle_different_sequence:
                pool = other_sequences
            else:
                try: frame_id = int(frame.name)
                except ValueError: frame_id = index
                pool = [candidate for candidate, path in enumerate(self.frames) if candidate != index and (path.parent.name != sequence or abs(int(path.name) - frame_id) >= self.minimum_frame_gap)]
            if not pool: raise ValueError(f"no valid shuffled-depth source for {frame}; provide another sequence or a larger split")
            choices.append(pool[torch.randint(len(pool), (1,), generator=generator).item()])
        return choices

    def _depth(self, index: int, fallback_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.depth_mode == "none":
            height, width = fallback_size
            return torch.zeros(1, max(height, 1), max(width, 1)), torch.zeros(1, max(height, 1), max(width, 1))
        source = self.frames[index]
        if self.depth_mode == "frame_shuffled":
            source = self.frames[self._frame_shuffle[index]]
        names = ("oracle_relative_depth.npy", "oracle_depth_valid.npy") if self.depth_mode == "oracle" else ("relative_depth.npy", "depth_valid.npy")
        if not (source / names[0]).exists() or not (source / names[1]).exists():
            raise FileNotFoundError(f"depth mode {self.depth_mode} requires {source / names[0]}")
        depth, valid = self._load(source, names[0]), self._load(source, names[1])
        if self.depth_mode == "spatial_shuffled":
            generator = torch.Generator().manual_seed(index)
            permutation = torch.randperm(depth.numel(), generator=generator)
            depth, valid = depth.flatten()[permutation].reshape_as(depth), valid.flatten()[permutation].reshape_as(valid)
        elif self.depth_mode == "constant":
            values = depth[valid.bool()]
            depth = torch.full_like(depth, values.median() if values.numel() else 0)
        return depth.unsqueeze(0), valid.unsqueeze(0)

    def __getitem__(self, index: int) -> dict:
        path = self.frames[index]
        meta = np.load(path / "meta.npz")
        target_range = self._load(path, "target_range.npy").unsqueeze(0)
        target_valid = self._load(path, "target_valid.npy").unsqueeze(0)
        sample = {
            "lidar": {"range": self._load(path, "input_range.npy").unsqueeze(0), "intensity": self._load(path, "input_intensity.npy").unsqueeze(0), "valid": self._load(path, "input_valid.npy").unsqueeze(0), "elevation": torch.as_tensor(meta["input_elevation"], dtype=torch.float32), "azimuth": torch.as_tensor(meta["azimuth"], dtype=torch.float32)},
            "target": {"range": target_range, "valid": target_valid, "elevation": torch.as_tensor(meta["target_elevation"], dtype=torch.float32)},
            "calibration": {"K": torch.as_tensor(meta["K"], dtype=torch.float32), "T_cam_lidar": torch.as_tensor(meta["T_cam_lidar"], dtype=torch.float32), "image_size": torch.as_tensor(meta["image_size"] if "image_size" in meta else [0, 0], dtype=torch.long)},
            "meta": {"sequence": path.parent.name, "frame_id": path.name},
        }
        image_size = tuple(int(value) for value in (meta["image_size"] if "image_size" in meta else [1, 1]))
        depth, depth_valid = self._depth(index, image_size)
        sample["camera"] = {"relative_depth": depth, "depth_valid": depth_valid}
        sample["meta"]["depth_mode"] = self.depth_mode
        return sample


class LidarInferenceDataset(Dataset):
    """GT-free dataset used by the deployable LiDAR-only student."""
    def __init__(self, root: str | Path, frame_dirs: list[str | Path] | None = None):
        self.root = Path(root)
        self.frames = [Path(path) for path in frame_dirs] if frame_dirs else sorted(self.root.glob("*/*"))
        self.frames = [path for path in self.frames if (path / "input_range.npy").exists() and (path / "meta.npz").exists()]
        if not self.frames:
            raise FileNotFoundError(f"no inference frames under {self.root}")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        path = self.frames[index]
        meta = np.load(path / "meta.npz")
        return {"lidar": {"range": torch.as_tensor(np.load(path / "input_range.npy"), dtype=torch.float32).unsqueeze(0), "intensity": torch.as_tensor(np.load(path / "input_intensity.npy"), dtype=torch.float32).unsqueeze(0), "valid": torch.as_tensor(np.load(path / "input_valid.npy"), dtype=torch.float32).unsqueeze(0), "elevation": torch.as_tensor(meta["input_elevation"], dtype=torch.float32), "azimuth": torch.as_tensor(meta["azimuth"], dtype=torch.float32)}, "target": {"elevation": torch.as_tensor(meta["target_elevation"], dtype=torch.float32)}, "meta": {"sequence": path.parent.name, "frame_id": path.name}}


# Backward-compatible name for callers written before inference was separated.
ProcessedRangeDataset = ProcessedTrainingDataset
