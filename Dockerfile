# 1. Use the official NVIDIA CUDA base (Stable & Supported on SageMaker)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install System Dependencies (Python 3.10 is default in Ubuntu 22.04)
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        wget \
        curl && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Install Python Libraries
# We install 'mineru[core,vllm]' which automatically pulls the correct compatible vLLM version
RUN pip install uv

RUN uv pip install --system fastapi uvicorn boto3 loguru python-multipart "mineru[core]"

# 6. Runtime Configuration
# NOW we tell the app to use the local models we just downloaded
# ENV MINERU_MODEL_SOURCE=local
# ENV MINERU_DEVICE=cuda

# RUN mineru-models-download -s huggingface -m vlm

# 7. Setup Workspace
WORKDIR /app

# 8. Copy your SageMaker adapter script
COPY serve.py /app/serve.py

# 9. Expose port
EXPOSE 8080

# 10. Entrypoint - Run FastAPI app with uvicorn
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]