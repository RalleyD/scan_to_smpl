FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git curl wget build-essential \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.4 + CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
RUN pip install --no-cache-dir \
    smplx>=0.1.28 \
    opencv-python>=4.8 \
    numpy>=1.24 \
    scipy>=1.10 \
    "transformers>=4.45" \
    Pillow>=10.0 \
    open3d>=0.17 \
    trimesh>=4.0 \
    click>=8.0 \
    tqdm \
    pytest \
    ruff \
    mypy

# Kaolin for PyTorch 2.4 + CUDA 12.1
RUN pip install --no-cache-dir \
    kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# Viz (optional but useful during development)
RUN pip install --no-cache-dir pyvista matplotlib

WORKDIR /workspace
