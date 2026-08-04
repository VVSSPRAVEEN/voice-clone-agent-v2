# Backend image for Voice Clone Agent
# PyTorch with CUDA 12.1 base, Python 3.11
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/data/models/hf \
    XDG_CACHE_HOME=/app/data/.cache \
    COQUI_TOS_AGREED=1 \
    VLLM_USE_MODELSCOPE=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        ffmpeg libsndfile1 espeak-ng \
        git curl ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation

# Copy source
COPY app /app/app
COPY scripts /app/scripts

# Pre-create data dirs
RUN mkdir -p /app/data/speakers /app/data/calls /app/data/models

EXPOSE 8000

# Healthcheck-friendly default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
