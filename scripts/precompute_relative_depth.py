#!/usr/bin/env python3
"""Precompute raw relative inverse depth using a fixed Hugging Face depth model."""
import argparse
import os
import time
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
    print(f"[depth] selected frames: {len(paths)}; to_compute: {len(pending)}; reused: {len(paths) - len(pending)}", flush=True)
    if not pending:
        return
    from transformers import pipeline
    estimator = pipeline("depth-estimation", model=args.model, device=args.device)
    debug = os.environ.get("CAMERA_OPERATOR_SR_DEBUG") == "1"
    for index, image_path in enumerate(pending, start=1):
        frame_started = time.perf_counter()
        print(f"[depth] frame {index}/{len(pending)}: {image_path.stem} START", flush=True)
        if debug: print(f"[debug:depth] frame {index}/{len(pending)} {image_path.stem}: image_load_start", flush=True)
        image = Image.open(image_path).convert("RGB")
        if debug: print(f"[debug:depth] frame {index}/{len(pending)} {image_path.stem}: image_loaded size={image.width}x{image.height}", flush=True)
        inference_started = time.perf_counter()
        if debug: print(f"[debug:depth] frame {index}/{len(pending)} {image_path.stem}: inference_start", flush=True)
        result = estimator(image)
        print(f"[depth] frame {index}/{len(pending)}: {image_path.stem} inference_done elapsed={time.perf_counter() - inference_started:.2f}s", flush=True)
        if debug: print(f"[debug:depth] frame {index}/{len(pending)} {image_path.stem}: postprocess_start", flush=True)
        predicted = result["predicted_depth"]
        depth = predicted.detach().float() if isinstance(predicted, torch.Tensor) else torch.as_tensor(np.asarray(predicted), dtype=torch.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(0)
        depth = F.interpolate(depth[None, None], size=(image.height, image.width), mode="bilinear", align_corners=False).squeeze()
        if args.model_output == "depth":
            depth = depth.reciprocal()
        depth_np = depth.cpu().numpy()
        valid = np.isfinite(depth_np) & (depth_np > 0)
        if debug: print(f"[debug:depth] frame {index}/{len(pending)} {image_path.stem}: postprocess_done valid={int(valid.sum())}/{valid.size}", flush=True)
        target = args.output_root / image_path.stem
        target.mkdir(parents=True, exist_ok=True)
        if debug: print(f"[debug:depth] frame {index}/{len(pending)} {image_path.stem}: save_start target={target}", flush=True)
        # Inverse depth can be finite in float32 yet exceed float16's maximum
        # (65,504).  Saving it as float16 silently turns valid pixels into inf,
        # which later corrupts teacher input.  Preserve the model result in
        # float32; pipeline partial-resume validation will repair old files.
        np.save(target / "relative_depth.npy", np.where(valid, depth_np, 0).astype(np.float32))
        np.save(target / "depth_valid.npy", valid.astype(np.uint8))
        np.savez_compressed(target / "depth_meta.npz", model_name=args.model, input_resolution=np.asarray([image.height, image.width]), output_type="relative_inverse_depth")
        print(f"[depth] frame {index}/{len(pending)}: {image_path.stem} saved total={time.perf_counter() - frame_started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
