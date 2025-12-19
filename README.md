# Pipeline Service

A FastAPI service for document and image analysis using the MinerU library.

## Overview

This service provides document parsing capabilities with support for multiple backends:
- **pipeline**: General-purpose parsing backend
- **vlm-transformers**: Vision Language Model backend using transformers

## API Endpoints

### POST `/analyze-image`
Analyzes an image file and returns structured output (markdown, JSON).

**Parameters:**
- `image`: Image file to analyze (multipart/form-data)
- `backend`: Either `"pipeline"` or `"vlm-transformers"`

**Response:**
- Returns markdown, model output, content list, and middle JSON files

### GET `/health`
Health check endpoint.

### POST `/cleanup`
Cleans up the data directory.

## Running the Service

### Using Docker

```bash
docker build -t pipeline-service .
docker run -p 8080:8080 pipeline-service
```

The service will be available at `http://localhost:8080`.

## Requirements

- CUDA 12.1.1
- Python 3.10+
- MinerU library