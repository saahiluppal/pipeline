FROM vllm/vllm-openai:v0.10.1.1

# Install libgl for opencv support & Noto fonts for Chinese characters
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install mineru + FastAPI deps
RUN python3 -m pip install -U \
        fastapi \
        uvicorn \
        boto3 \
        loguru \
        python-multipart \
        'mineru[core]>=2.7.0' \
        --break-system-packages && \
    python3 -m pip cache purge

# Download models
RUN /bin/bash -c "mineru-models-download -s huggingface -m all"

# Setup workspace
WORKDIR /app

# Copy FastAPI app
COPY serve.py /app/serve.py

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose both ports
EXPOSE 30000
EXPOSE 8080

# Single entrypoint
ENTRYPOINT ["/app/start.sh"]
