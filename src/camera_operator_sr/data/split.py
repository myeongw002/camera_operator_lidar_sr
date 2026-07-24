from pathlib import Path


def load_split(root: str | Path, split_file: str | Path) -> list[Path]:
    root = Path(root)
    entries = [line.strip() for line in Path(split_file).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    frames: list[Path] = []
    for entry in entries:
        candidate = root / entry
        if candidate.is_dir() and (candidate / "target_range.npy").exists():
            frames.append(candidate)
            continue
        # A sequence entry selects every frame under that sequence.
        sequence = root / entry
        if sequence.is_dir():
            frames.extend(sorted(path for path in sequence.iterdir() if path.is_dir()))
            continue
        raise FileNotFoundError(f"split entry does not exist: {entry}")
    return frames
