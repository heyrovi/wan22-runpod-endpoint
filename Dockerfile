FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# System Dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
WORKDIR /workspace

# Requirements installieren
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# WAN2.2 Repository klonen und Dependencies installieren
RUN git clone https://github.com/Wan-Video/Wan2.2.git /workspace/Wan2.2 && \
    cd /workspace/Wan2.2 && \
    pip install -e .

# Handler Script kopieren
COPY handler.py /workspace/

# Model Cache Directory
RUN mkdir -p /workspace/models

# Huggingface Cache setzen
ENV HF_HOME="/workspace/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"

# RunPod Handler als Entrypoint
CMD ["python", "-u", "/workspace/handler.py"]
