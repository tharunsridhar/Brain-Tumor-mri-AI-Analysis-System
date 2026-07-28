# NeuroScan AI FastAPI Deployment

## Local Windows

Run from the repository root:

```powershell
.\deploy\start.ps1 -HostName 127.0.0.1 -Port 8000
```

Open `http://127.0.0.1:8000`.

## Docker

Run from the repository root:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

The Docker image does not bake in `.keras` model files. The compose file mounts the local `model/` folder into the container at runtime.

## Important Environment Variables

- `GROQ_API_KEY`: enables Groq report generation. If absent, the API returns a local fallback report.
- `NEUROSCAN_CORS_ORIGINS`: comma-separated allowed origins, or `*` for local testing.
- `NEUROSCAN_MAX_UPLOAD_MB`: upload limit in MB. Default is `25`.
- `NEUROSCAN_MODELS_DIR`: optional custom model directory. Default is `model/`.

## Endpoints

- `GET /health`
- `GET /ready`
- `POST /api/analyze`
- `GET /api/history`
- `GET /api/reports`
- `GET /api/reports/{filename}`
- `GET /api/model-info`
- `GET /docs`
