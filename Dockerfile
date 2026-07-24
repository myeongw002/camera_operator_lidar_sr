# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_NAME=research
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_HOME=/home/${USER_NAME}/.cache/torch \
    HF_HOME=/home/${USER_NAME}/.cache/huggingface \
    MPLCONFIGDIR=/home/${USER_NAME}/.cache/matplotlib \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        build-essential \
        ninja-build \
        cmake \
        pkg-config \
        git \
        git-lfs \
        curl \
        wget \
        ca-certificates \
        unzip \
        rsync \
        tmux \
        htop \
        nano \
        less \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

RUN python -m pip install --upgrade pip setuptools wheel

# Install CUDA 12.8 PyTorch wheels explicitly from the official PyTorch index.
RUN python -m pip install \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt requirements-dev.txt /tmp/requirements/
RUN python -m pip install -r /tmp/requirements/requirements.txt \
    && python -m pip install -r /tmp/requirements/requirements-dev.txt

RUN groupadd --gid ${GROUP_ID} ${USER_NAME} \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash ${USER_NAME} \
    && mkdir -p /workspace /data /outputs \
                 /home/${USER_NAME}/.cache/torch \
                 /home/${USER_NAME}/.cache/huggingface \
                 /home/${USER_NAME}/.cache/matplotlib \
    && chown -R ${USER_NAME}:${USER_NAME} \
         /workspace /data /outputs /home/${USER_NAME}

COPY --chown=${USER_NAME}:${USER_NAME} docker/entrypoint.sh /usr/local/bin/project-entrypoint
COPY --chown=${USER_NAME}:${USER_NAME} docker/verify_cuda.py /usr/local/bin/verify_cuda.py
RUN chmod +x /usr/local/bin/project-entrypoint

USER ${USER_NAME}
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/project-entrypoint"]
CMD ["bash"]
