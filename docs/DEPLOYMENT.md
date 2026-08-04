# Deployment

## Quick start (Docker)

```bash
git clone <repo>
cd voice-clone-agent
cp .env.example .env
make build
make download          # ~5 GB of model weights
make up
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

## Prerequisites

### Host machine

- NVIDIA GPU with at least 6 GB VRAM (RTX 3060 6 GB is the minimum target)
- NVIDIA driver 525+ and CUDA 12.1+ runtime (provided by the Docker image)
- Docker 20.10+ with the NVIDIA Container Toolkit installed
- ~10 GB free disk for models + Docker images

### Install NVIDIA Container Toolkit (if not already)

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

## Native (without Docker)

If you can't use Docker, run directly in a Python venv:

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000
streamlit run streamlit_app/main.py --server.port=8501
```

You'll need:
- Python 3.11+
- ffmpeg installed (`apt install ffmpeg`)
- NVIDIA CUDA 12.1+ and PyTorch with CUDA support (pip-installed version
  should work if your system CUDA matches)

## Configuration

All runtime config is in `.env` (see `.env.example`). Most important:

| Variable              | Default | Notes                                                      |
|-----------------------|---------|------------------------------------------------------------|
| `PIPELINE_MODE`       | parallel | `parallel` (streaming) or `sequential`                    |
| `MAX_CONCURRENT_CALLS` | 1       | Increase requires more VRAM                                |
| `STT_MODEL`           | medium  | tiny/base/small/medium/large-v3                            |
| `STT_LANGUAGE`        | te      | Telugu default; `auto` for autodetect                      |
| `TTS_ENGINE`          | xtts    | `xtts` (Coqui) or `sherpa` (VITS)                          |
| `LLM_ENABLED`         | false   | Set true + configure `LLM_API_BASE` for LLM replies        |

## Model download

`make download` runs `scripts/download_models.py` which pulls:

- `Systran/faster-whisper-medium` (~1.5 GB, CT2 format)
- `silero_vad.onnx` (~2 MB)
- Coqui XTTS v2 weights (~1.8 GB, downloaded on first TTS call)
- (optional with `--include-sherpa`) AI4Bharat Telugu VITS (~100 MB)

Models land in `./data/models/` (host) which is bind-mounted into the
backend container at `/app/data/models/`.

## GPU memory management

Each worker calls `torch.cuda.empty_cache()` after inference to release
fragmented memory back to the allocator. The pipeline also has an
`unload()` method on each worker that fully releases the model.

For `PIPELINE_MODE=parallel` (default), all four stages hold their models
simultaneously. On a 6 GB card with the default config (STT=medium INT8 +
XTTS), this can hit ~7 GB. Mitigations (pick one):

1. `PIPELINE_MODE=sequential` — stages load/unload in turn (peak ~4 GB)
2. `TTS_ENGINE=sherpa` — sherpa-onnx VITS uses ~1 GB instead of 4 GB
3. `STT_MODEL=small` — uses ~1.5 GB instead of 3 GB

For `MAX_CONCURRENT_CALLS > 1`, definitely use `TTS_ENGINE=sherpa` to
stay under 6 GB.

## Health checks

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "ok",
  "device": "cuda",
  "pipeline_mode": "parallel",
  "llm_enabled": false,
  "tts_engine": "xtts",
  "vram_total_mb": 6144,
  "vram_used_mb": 0,
  "vram_free_mb": 6144,
  "speakers_count": 0,
  "calls_count": 0
}
```

The Docker Compose `healthcheck` calls this every 30 seconds. The
container is considered healthy once `/health` returns 200.

## Logging

Logs go to stdout in JSON-ish format via `loguru`. To tail:

```bash
make logs          # all services
docker compose logs -f backend
```

To ship to a log aggregator, mount a log driver in `docker-compose.yml`
or use `loguru`'s sink feature to forward to Loki/Filebeat.

## Monitoring (optional)

The current build emits stdout logs only. For production monitoring:

1. Add a `/metrics` endpoint with `prometheus-fastapi-instrumentator`
2. Run `prometheus` + `grafana` containers in `docker-compose.yml`
3. Suggested panels:
   - VRAM used / total
   - Active WebSocket connections
   - Pipeline latency percentiles (STT, LLM, TTS, end-to-end)
   - Calls per hour, average duration
   - Error rate by stage

## Backups

- `data/speakers/` — speaker registry (small, ~MB total)
- `data/calls/` — call audio + transcripts (can grow large)
- `data/calls.db` — SQLite metadata index
- `data/models/` — model weights (regenerable via `make download`)

Suggested backup strategy: nightly cron job that tars `data/speakers/`,
`data/calls/`, and `data/calls.db` to off-host storage (S3, Backblaze B2,
etc.). Skip `data/models/` — it's reproducible.

## Scaling

The current build targets single-machine deployment. For multi-worker
scaling:

1. Add Redis container (commented out in `docker-compose.yml`)
2. Replace in-memory `_uploads` dict in `main.py` with Redis hash
3. Replace `ConnectionManager` set with Redis-backed active-call counter
4. Run multiple backend replicas behind a load balancer (nginx/traefik)
5. Sticky sessions required for WebSocket (`/ws/call`) — use IP hash or
   cookie-based routing
6. Each replica needs its own GPU; you cannot share a 6 GB GPU across
   replicas with the current model sizes

## Security

For local-only deployment (`AUTH_MODE=none`), the backend has no auth.
For exposing beyond localhost:

1. Set `AUTH_MODE=apikey` and a strong `API_KEY`
2. Put both services behind a TLS-terminating reverse proxy (Caddy / nginx)
3. Add CORS restrictions (currently `*`)
4. Rate-limit the WebSocket endpoint to prevent abuse
5. Add user authentication if multi-tenant

## Troubleshooting

### "CUDA out of memory"

- Switch to `PIPELINE_MODE=sequential`
- Or `TTS_ENGINE=sherpa`
- Or smaller STT model (`STT_MODEL=small`)

### "Coqui XTTS license agreement"

Set `COQUI_TOS_AGREED=1` in your environment (already set in the Dockerfile).
First call to `TTS(...)` will download the model.

### "Silero VAD model not found"

Run `make download` first. The ONNX model lives at
`data/models/silero-vad/silero_vad.onnx`.

### WebSocket connection refused

- Check `MAX_CONCURRENT_CALLS` — additional callers get 1013'd
- Check that the backend is healthy: `curl http://localhost:8000/health`
- Check browser console for CORS / mixed-content errors (frontend is HTTP,
  backend is HTTP — both must be same scheme)

### Streamlit audio_input returns None

- Ensure you're using a recent Streamlit (1.30+)
- Ensure HTTPS or `localhost` (browsers block mic access on plain HTTP
  from non-localhost origins)
