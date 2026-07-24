# Docker environment for Camera-Guided Local Operator LiDAR SR

## Fixed environment

- Ubuntu 22.04
- NVIDIA CUDA 12.8.1 + cuDNN development image
- Python 3.10 (Ubuntu system Python)
- PyTorch 2.9.1 / torchvision 0.24.1 with CUDA 12.8 wheels
- Open3D, OpenCV, Hydra, TensorBoard, W&B, JupyterLab

The `devel` CUDA image is used because this research project may later compile CUDA/C++ extensions. For pure inference, a smaller runtime image can be created later.

## 1. Host prerequisites

The host needs:

1. A current NVIDIA driver that supports the RTX 5070.
2. Docker Engine with the Compose plugin.
3. NVIDIA Container Toolkit configured for Docker.

The CUDA toolkit does not need to be installed separately on the host. The NVIDIA driver remains on the host, while CUDA user-space libraries are provided by the container.

Verify the host first:

```bash
nvidia-smi
docker --version
docker compose version
```

Verify Docker GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## 2. Copy the files into the project root

Expected layout:

```text
camera_operator_lidar_sr/
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── .env
├── docker/
├── src/
├── scripts/
├── configs/
├── data/        # optional local mount
└── outputs/     # training outputs
```

## 3. Configure host paths

```bash
cp .env.example .env
id -u
id -g
```

Set `USER_ID` and `GROUP_ID` in `.env` to the values returned by `id -u` and `id -g`.

For datasets stored elsewhere:

```dotenv
DATA_ROOT=/absolute/path/to/datasets
OUTPUT_ROOT=/absolute/path/to/outputs
CACHE_ROOT=/absolute/path/to/model-cache
```

Do not put a trailing slash after the path.

## 4. Build

```bash
make build
```

Equivalent command:

```bash
docker compose build
```

## 5. Verify RTX 5070 and PyTorch

```bash
make gpu-check
```

Expected fields include:

```text
PyTorch: 2.9.1+cu128
PyTorch CUDA runtime: 12.8
CUDA available: True
... RTX 5070 ... compute capability 12.0
GPU matmul OK
```

## 6. Open a shell

```bash
make shell
```

Inside the container:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
python -c "import open3d as o3d; print(o3d.__version__)"
```

## 7. Install the local project package

After creating `pyproject.toml` in the project root:

```bash
make shell
pip install -e .
```

The source folder is also exposed through `PYTHONPATH=/workspace/src`, so editable installation is not mandatory during the earliest implementation stage.

## 8. Training commands

Examples:

```bash
make shell
python scripts/train_student.py --config configs/train/student_baseline.yaml
python scripts/train_teacher.py --config configs/train/teacher.yaml
python scripts/train_distill.py --config configs/train/distill.yaml
```

Data is available at `/data` and outputs should be written to `/outputs`.

## 9. Jupyter and TensorBoard

```bash
make jupyter
```

Open `http://localhost:8888`.

```bash
make tensorboard
```

Open `http://localhost:6006`.

## 10. VS Code Dev Containers

Install the VS Code Dev Containers extension, open the project directory, and choose:

```text
Dev Containers: Reopen in Container
```

## 11. Memory settings for the RTX 5070 12 GB

Start conservatively:

- AMP: bf16 when supported, otherwise fp16
- Student batch size: 2-4
- Teacher batch size: 1-2
- Distillation batch size: 1-2 because teacher, baseline, and student may coexist
- DataLoader workers: 2-4
- OpenBLAS/OMP/MKL threads: 1
- Gradient accumulation instead of increasing batch size

The Compose file uses `ipc: host` and a 16 GB shared-memory allowance to reduce DataLoader shared-memory failures.

## 12. Common failures

### `could not select device driver` or GPU unavailable

The NVIDIA Container Toolkit is absent or not configured. Reinstall/configure it, restart Docker, and rerun the Docker GPU smoke test.

### `no kernel image is available for execution`

An older PyTorch/CUDA build was installed. Confirm:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

Rebuild without cache if the image does not show `+cu128`:

```bash
make rebuild
```

### Files owned by root

Set `USER_ID=$(id -u)` and `GROUP_ID=$(id -g)` in `.env`, then rebuild.

### DataLoader bus error

Reduce workers, verify `ipc: host`, and check host RAM. The initial thread variables are deliberately fixed to 1.

### Open3D visualization fails

The image uses `opencv-python-headless` and is intended for training servers. Save visualizations to files. Interactive GUI forwarding requires extra X11/Wayland configuration and is intentionally excluded.
