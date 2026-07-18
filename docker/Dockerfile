FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace

# ------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        ffmpeg \
        ninja-build \
        build-essential \
        python3-dev \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Python tooling
# ------------------------------------------------------------------
RUN pip install --upgrade \
    pip \
    wheel \
    setuptools==79.0.1

# ------------------------------------------------------------------
# PyTorch
# ------------------------------------------------------------------
RUN pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0

# ------------------------------------------------------------------
# OpenMMLab dependencies
# ------------------------------------------------------------------
RUN pip install mmengine==0.10.7
RUN pip install mmdet==3.3.0

# ------------------------------------------------------------------
# MMCV (CUDA 12.9)
# ------------------------------------------------------------------
RUN pip install --no-build-isolation \
    mmcv==2.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu129/torch2.8.0/index.html

# ------------------------------------------------------------------
# Patch mmdet version restriction (allow mmcv 2.2.0)
# ------------------------------------------------------------------
RUN sed -i \
    "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" \
    /opt/conda/lib/python3.11/site-packages/mmdet/__init__.py

# ------------------------------------------------------------------
# Clone MMDetection3D v1.4.0
# ------------------------------------------------------------------
RUN git clone \
    --branch v1.4.0 \
    https://github.com/open-mmlab/mmdetection3d.git \
    /workspace/official_mmdet3d

WORKDIR /workspace/official_mmdet3d

# ------------------------------------------------------------------
# Relax mmdet3d version checks
# ------------------------------------------------------------------
RUN sed -i "s/mmcv_maximum_version = .*/mmcv_maximum_version = '2.3.0'/" mmdet3d/__init__.py && \
    sed -i "s/mmdet_maximum_version = .*/mmdet_maximum_version = '3.4.0'/" mmdet3d/__init__.py && \
    sed -i "s/mmseg_maximum_version = .*/mmseg_maximum_version = '2.0.0'/" mmdet3d/__init__.py

# ------------------------------------------------------------------
# Install mmdet3d (build CUDA ops)
# ------------------------------------------------------------------
RUN pip install -v -e . --no-build-isolation

# ------------------------------------------------------------------
# Adjust BEVFusion ops for RTX 5090 (compute 12.0)
# ------------------------------------------------------------------
RUN sed -i \
    's/-gencode=arch=compute_[0-9]\+,code=sm_[0-9]\+/-gencode=arch=compute_120,code=sm_120/g' \
    projects/BEVFusion/setup.py

# ------------------------------------------------------------------
# Compile BEVFusion custom CUDA ops
# ------------------------------------------------------------------
RUN python projects/BEVFusion/setup.py build_ext --inplace

# ------------------------------------------------------------------
# PyTorch 2.8 checkpoint compatibility
# ------------------------------------------------------------------
RUN sed -i \
    's/torch.load(filename, map_location=map_location)/torch.load(filename, map_location=map_location, weights_only=False)/' \
    /opt/conda/lib/python3.11/site-packages/mmengine/runner/checkpoint.py

# ------------------------------------------------------------------
# Download nuScenes metadata
# ------------------------------------------------------------------
RUN mkdir -p data/nuscenes && \
    wget -O data/nuscenes/nuscenes_infos_train.pkl \
    https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl && \
    wget -O data/nuscenes/nuscenes_infos_val.pkl \
    https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
ENV PYTHONPATH="/workspace/official_mmdet3d"

WORKDIR /workspace/official_mmdet3d

CMD ["/bin/bash"]