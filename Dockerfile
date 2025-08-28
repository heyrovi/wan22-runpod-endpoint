FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system deps
RUN apt-get update && apt-get install -y \
    git wget ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Install Python deps
RUN pip install --no-cache-dir \
    runpod \
    huggingface-hub \
    opencv-python \
    Pillow \
    numpy \
    imageio[ffmpeg] \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Download WAN2.2
RUN git clone https://github.com/Wan-Video/Wan2.2.git

# Install WAN2.2 dependencies
WORKDIR /Wan2.2
RUN pip install --no-cache-dir \
    diffusers==0.25.0 \
    transformers==4.35.0 \
    accelerate==0.24.0 \
    imageio==2.33.0 \
    decord==0.6.0 \
    scipy \
    einops \
    omegaconf \
    safetensors \
    easydict \
    pandas \
    torchsde \
    xformers==0.0.22.post7

# Copy handler
COPY handler.py /handler.py

# Download models at build time
RUN mkdir -p /models && \
    huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir /models/Wan2.2-I2V-A14B --local-dir-use-symlinks False && \
    huggingface-cli download Wan-AI/Wan2.2-VAE --local-dir /models/vae --local-dir-use-symlinks False

CMD ["python", "-u", "/handler.py"]
