#!/usr/bin/env python3
"""Precompute raw relative inverse depth using a fixed Hugging Face depth model."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _valid_output(target: Path) -> bool:
    try:
        depth = np.load(target / "relative_depth.npy", allow_pickle=False)
        valid = np.load(target / "depth_valid.npy", allow_pickle=False).astype(bool)
        return depth.shape == valid.shape and bool(valid.any()) and bool(np.isfinite(depth[valid]).all())
    except (OSError, ValueError, KeyError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--model-output", choices=("depth", "inverse_depth"), default="depth")
    parser.add_argument("--device", type=int, default=0 if torch.cuda.is_available() else -1)
    parser.add_argument("--frame-list", type=Path, help="newline-delimited image stems selected by the pipeline")
    parser.add_argument("--batch-size", type=int, default=1, help="Depth backend currently supports one image per inference call.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError("--batch-size currently supports only 1 for this depth backend")
    selected = None if args.frame_list is None else {line.strip() for line in args.frame_list.read_text().splitlines() if line.strip()}
    if selected is not None and not selected:
        raise ValueError("--frame-list contains no frames")
    paths = [path for extension in ("*.png", "*.jpg", "*.jpeg") for path in args.image_root.glob(extension)]
    paths = sorted(path for path in paths if selected is None or path.stem in selected)
    if not paths:
        raise ValueError("no selected images found")
    pending = [path for path in paths if args.overwrite or not _valid_output(args.output_root / path.stem)]
    if not pending:
        return
    from transformers import pipeline
    estimator = pipeline("depth-estimation", model=args.model, device=args.device)
    for image_path in pending:
        image = Image.open(image_path).convert("RGB")
        result = estimator(image)
        predicted = result["predicted_depth"]
        depth = predicted.detach().float() if isinstance(predicted, torch.Tensor) else torch.as_tensor(np.asarray(predicted), dtype=torch.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(0)
        depth = F.interpolate(depth[None, None], size=(image.height, image.width), mode="bilinear", align_corners=False).squeeze()
        if args.model_output == "depth":
            depth = depth.reciprocal()
        depth_np = depth.cpu().numpy()
        valid = np.isfinite(depth_np) & (depth_np > 0)
        target = args.output_root / image_path.stem
        target.mkdir(parents=True, exist_ok=True)
        # Inverse depth can be finite in float32 yet exceed float16's maximum
        # (65,504).  Saving it as float16 silently turns valid pixels into inf,
        # which later corrupts teacher input.  Preserve the model result in
        # float32; pipeline partial-resume validation will repair old files.
        np.save(target / "relative_depth.npy", np.where(valid, depth_np, 0).astype(np.float32))
        np.save(target / "depth_valid.npy", valid.astype(np.uint8))
        np.savez_compressed(target / "depth_meta.npz", model_name=args.model, input_resolution=np.asarray([image.height, image.width]), output_type="relative_inverse_depth")


if __name__ == "__main__":
    main()
