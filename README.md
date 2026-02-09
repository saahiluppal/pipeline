# Pipeline Service

A FastAPI service for document and image analysis powered by the [MinerU](https://github.com/opendatalab/MinerU) library with a built-in vLLM backend.

## Overview

The container runs two processes:

| Process | Port | Description |
|---------|------|-------------|
| **MinerU OpenAI Server** | 30000 | vLLM-based VLM inference server (started automatically) |
| **FastAPI App** | 8080 | Public-facing API that orchestrates parsing via MinerU |

Supported parsing backends (used internally by `parse_doc`):

- **pipeline** — General-purpose OCR/layout pipeline.
- **vlm-auto-engine** — VLM via local compute (auto-detected engine).
- **vlm-http-client** — VLM via a remote OpenAI-compatible server.
- **hybrid-auto-engine** — Next-gen hybrid (pipeline + VLM) via local compute (default).
- **hybrid-http-client** — Hybrid via a remote server with minimal local compute.

The API endpoints use the `vlm-http-client` backend by default, routing requests to the co-located MinerU OpenAI server on port 30000.

## API Endpoints

### `POST /analyze-image`

Analyze an image file (JPEG, PNG, GIF, or WebP) and return structured output.

**Parameters (multipart/form-data):**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | file | *(required)* | The image file to analyze |
| `download_dir` | bool | `false` | If `true`, return a zip archive of the full output directory instead of JSON |

**Response (JSON):**

```json
{
  "success": true,
  "output_dir": "data/<task-id>",
  "files": {
    "markdown": "...",
    "model_output": { ... },
    "content_list": { ... },
    "middle_output": { ... }
  },
  "time_elapsed": 12.345
}
```

When `download_dir=true`, the response is a zip file download instead.

---

### `POST /analyze-document`

Analyze a PDF document and return structured output.

**Parameters (multipart/form-data):**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `document` | file | *(required)* | The PDF file to analyze |
| `download_dir` | bool | `false` | If `true`, return a zip archive of the full output directory instead of JSON |
| `start_page` | int | `0` | Start page index (0-based, inclusive) |
| `end_page` | int | `null` | End page index (0-based, inclusive). `null` processes all pages to the end |
| `page_index` | int | `null` | Process only this single page (0-based). Overrides `start_page`/`end_page` when provided |

**Response:** Same structure as `/analyze-image` (with `document.*` filenames).

---

### `GET /health`

Health check that reports both API and VLM backend status.

**Response:**

```json
{
  "api_status": "ok",
  "backend_status": "ok"
}
```

If the backend is unreachable, `backend_status` will be `"down"` and an `error` field is included.

---

### `GET /cleanup`

Remove all task subdirectories and files under the `data/` directory.

**Response:**

```json
{
  "success": true,
  "message": "Data directory cleaned successfully"
}
```

## Running the Service

### Using Docker

```bash
docker build -t pipeline-service .
docker run --gpus all -p 8080:8080 -p 30000:30000 pipeline-service
```

The FastAPI service will be available at `http://localhost:8080` and the internal VLM server at `http://localhost:30000`.

### Startup

The container entrypoint (`start.sh`) launches:

1. `mineru-openai-server` on port 30000 (background)
2. `uvicorn serve:app` on port 8080 (foreground)

## Requirements

- GPU with CUDA support (base image: `vllm/vllm-openai:v0.10.1.1`)
- Python 3.10+
- MinerU `>=2.7.0` (with `core` extras)
- FastAPI, Uvicorn, Loguru, python-multipart, boto3

## Version History

| Version | Commit | Highlights |
|---------|--------|------------|
| **v1** | [`4a04302`](https://github.com/saahiluppal/pipeline/tree/4a043025532a43d43d1aeb76b76cd65236358bf1) | Initial release — `POST /analyze-image` with selectable `pipeline` / `vlm-transformers` backends, basic `/health` and `/cleanup` endpoints. |
| **v2** | [`5277479`](https://github.com/saahiluppal/pipeline/tree/52774795baa3d8ebf87e9e113591c183d33ed61a) | Added `start.sh` entrypoint with co-located MinerU OpenAI server; switched to `vlm-http-client` backend; added `POST /analyze-document` with page-range support (`start_page`, `end_page`, `page_index`); enhanced `/health` to probe VLM backend status; added `download_dir` zip-download option on both analyze endpoints; changed `/cleanup` from POST to GET. |
| **v2.1** | [`16d1eaf`](https://github.com/saahiluppal/pipeline/tree/16d1eaf701dd2155ee8b3f267e5fcfbc6397a10d) | Added `start_page`/`end_page` range and `page_index` single-page parameter to `/analyze-document`; updated README to reflect current API surface, backends, architecture, and Docker instructions. |
