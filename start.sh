#!/usr/bin/env bash
set -e

# Optional: ensure local models
export MINERU_MODEL_SOURCE=local

echo "Starting mineru-openai-server..."
mineru-openai-server --host 0.0.0.0 --port 30000 &

echo "Starting FastAPI server..."
uvicorn serve:app --host 0.0.0.0 --port 8080

# uvicorn stays in foreground (PID 1)