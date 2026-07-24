from __future__ import annotations

import platform
import sys

import torch


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available inside the container.", file=sys.stderr)
        return 1

    count = torch.cuda.device_count()
    print(f"CUDA device count: {count}")
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        print(
            f"[{index}] {props.name} | compute capability "
            f"{props.major}.{props.minor} | VRAM {props.total_memory / 2**30:.2f} GiB"
        )

    device = torch.device("cuda:0")
    x = torch.randn(2048, 2048, device=device)
    y = torch.randn(2048, 2048, device=device)
    z = x @ y
    torch.cuda.synchronize()
    print(f"GPU matmul OK: shape={tuple(z.shape)}, mean={z.mean().item():.6f}")

    # The RTX 50 family is expected to report compute capability 12.0.
    major, minor = torch.cuda.get_device_capability(0)
    if major < 12:
        print(
            f"WARNING: detected compute capability {major}.{minor}; "
            "this image was selected primarily for Blackwell/RTX 50 support."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
