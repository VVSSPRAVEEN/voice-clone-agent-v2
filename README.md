# Voice Clone Agent (Telugu / English) — RTX 3060 6GB

A self-hosted, zero-shot voice cloning agent that supports bilingual
(Telugu + English) speech-to-speech conversation, call recording, transcript
playback, and 300-minute long-audio handling. Designed to fit on a single
RTX 3060 6 GB GPU.

> **Note on local LLM:** The LLM stage is an OpenAI-compatible API hook. On
> this machine it is a **local vLLM server** (Qwen2.5-3B-Instruct-AWQ,
> `http://127.0.0.1:8001`, set `LLM_ENABLED=true` + `LLM_API_URL` in
> `backend/.env`); any other OpenAI-compatible endpoint (Ollama, LM Studio,
> OpenAI) works too. When no LLM is configured, the pipeline falls back to a
> simple rule-based responder so the rest of the system (STT, voice cloning,
> call logging, UI) remains fully functional.

---

## Quick start

```bash
# 0) Weights: already cached under D:\hf-models (config.py sets
#    HF_HOME=D:/hf-models automatically at import time). Optional manual
#    (re)download:  cd backend && .venv\Scripts\python.exe scripts\download_models.py

# 1) Backend (FastAPI + WebSocket)  -> http://localhost:8000
cd backend && .venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2) Frontend (Streamlit UI)        -> http://localhost:8501
cd frontend\streamlit_app && ..\..\backend\.venv\Scripts\python.exe -m streamlit run main.py --server.port 8501

# 3) Optional local LLM (vLLM, Qwen2.5-3B-Instruct-AWQ) -> http://127.0.0.1:8001
#    Sized for the 6 GB GPU: --enforce-eager (no CUDA graphs),
#    --gpu-memory-utilization 0.7, --max-model-len 1024 (small KV cache).
#    Skip this step to use the rule-based fallback.
D:\vllm-venv\Scripts\python.exe -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-3B-Instruct-AWQ `
  --port 8001 --enforce-eager --gpu-memory-utilization 0.7 --max-model-len 1024

# Backend:  http://localhost:8000    (FastAPI + WebSocket)
# Frontend: http://localhost:8501    (Streamlit UI)

# (The Makefile's `make dev-backend` / `make dev-frontend` run the same two
#  commands; all other make targets are Docker-based.)
```

## What's inside

| Component        | Choice                                  | VRAM   |
|------------------|-----------------------------------------|--------|
| STT              | faster-whisper `medium` INT8            | ~3 GB  |
| VAD              | Silero VAD (ONNX, CPU)                  | ~0 GB  |
| TTS (cloning)    | Coqui XTTS v2 (multilingual, zero-shot) | ~4 GB  |
| TTS (alt)        | sherpa-onnx VITS (Telugu/English)       | ~1 GB  |
| TTS (hybrid)     | Praxy (te/ta) / XTTS (en/hi) / Edge     | ~0-4 GB|
| LLM (optional)   | External OpenAI-compatible API          | 0 GB   |
| Diarization      | (Hook only — not enabled by default)    | —      |

The pipeline runs in **parallel streaming mode** by default: VAD, STT, LLM,
and TTS stages run as concurrent asyncio tasks connected by queues, so a new
utterance can be processed while the previous one is still being spoken.
Switch to `PIPELINE_MODE=sequential` for the older one-segment-at-a-time
behaviour.

## Repository layout

```
voice-clone-agent/
├── docker-compose.yml
├── .env.example
├── Makefile
├── README.md
├── PLAN.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py              # FastAPI + WebSocket endpoints
│   │   ├── config.py            # Pydantic settings
│   │   ├── models.py            # Pydantic schemas
│   │   ├── pipeline.py          # Orchestrator (parallel / sequential)
│   │   ├── speaker_registry.py  # Speaker embedding cache + CRUD
│   │   ├── call_logger.py       # SQLite + audio file storage
│   │   ├── vad_worker.py        # Silero VAD
│   │   ├── stt_worker.py        # faster-whisper
│   │   ├── llm_worker.py        # OpenAI-compatible API + fallback
│   │   ├── tts_worker.py        # Coqui XTTS / sherpa-onnx
│   │   └── websocket_handler.py # Streaming protocol
│   └── scripts/
│       ├── download_models.py
│       └── benchmark.py
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── streamlit_app/
│       ├── main.py
│       ├── pages/
│       │   ├── 1_Speaker_Lab.py
│       │   ├── 2_Call_Center.py
│       │   ├── 3_History.py
│       │   └── 4_Settings.py
│       ├── components/
│       └── utils/
├── evaluation/
│   ├── evaluate_stt.py
│   ├── evaluate_tts.py
│   └── evaluate_e2e.py
├── data/                        # gitignored runtime data
│   ├── speakers/                # embeddings + reference audio
│   ├── calls/                   # audio + transcripts
│   └── models/                  # downloaded weights
└── docs/
    ├── ARCHITECTURE.md
    ├── DEPLOYMENT.md
    ├── API.md
    └── EVALUATION.md
```

## Key features

- **Zero-shot voice cloning** — Upload 3-10 s of reference audio, generate
  speech in that voice in Telugu or English.
- **Live call mode** — Browser microphone via WebRTC, real-time transcript
  and bot audio via WebSocket.
- **300-minute audio** — Chunked streaming STT, never loads full audio into
  RAM; JSONL transcript with per-segment timestamps.
- **Speaker registry** — Multiple speakers, each with cached embedding and
  reference audio.
- **Call history** — SQLite metadata + filesystem audio; playback UI with
  timestamp seeking.
- **VRAM-aware** — Models load/unload with `torch.cuda.empty_cache()` between
  heavy stages to stay under 5 GB peak.
- **Parallel streaming** — VAD → STT → LLM → TTS as concurrent asyncio tasks
  (default), or sequential mode for determinism.

## Configuration

All runtime config lives in `backend/.env` (see `.env.example` at the repo root). The most important
toggles:

| Variable           | Default      | Notes                                       |
|--------------------|--------------|---------------------------------------------|
| `PIPELINE_MODE`    | `parallel`   | `parallel` (streaming) or `sequential`      |
| `MAX_CONCURRENT_CALLS` | `1`      | Higher values require more VRAM             |
| `STT_MODEL`        | `medium`     | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `STT_LANGUAGE`     | `te`         | Telugu; use `auto` for autodetect           |
| `TTS_ENGINE`       | `edge`       | `xtts` / `sherpa` / `ai4bharat` / `edge` / `hybrid` (auto-routing) |
| `AUTO_CPU_FALLBACK`| `true`       | Auto-run STT/Praxy/XTTS on CPU when VRAM is low; `false` = always CUDA |
| `LLM_ENABLED`      | `true`       | Set `false` to use the rule-based fallback; configure `LLM_API_URL` when enabled |
| `LLM_API_URL`      | `http://127.0.0.1:8001` | Local vLLM OpenAI endpoint (worker appends `/v1/chat/completions`) |
| `HF_HOME`          | `D:/hf-models`| Read by `huggingface_hub` (not pydantic); set at the top of `config.py` |

## Hybrid TTS & GPU memory

With `TTS_ENGINE=hybrid` each reply is routed by language instead of a single
engine:

- **Telugu / Tamil** (with a speaker reference clip) → **Praxy engine**:
  Chatterbox multilingual base + `Praxel/praxy-voice-r6` LoRA, cloning the
  uploaded speaker's voice. If Praxy fails to load it is auto-disabled and
  Telugu/Tamil falls back to Edge-TTS.
- **English / Hindi** → **XTTS v2** (zero-shot cloning from the reference
  clip).
- **Everything else** → **Edge-TTS** (online preset voices, no cloning).

- **GPU memory:** vLLM holds ~4.3 GB of the 6 GB card, so free VRAM is
  tight. whisper (STT), Praxy, and XTTS check free VRAM and **auto-run on
  CPU** when it is insufficient (thresholds: STT 1400 MiB, Praxy 3600 MiB,
  XTTS 2500 MiB). Set `AUTO_CPU_FALLBACK=false` to restore the old
  always-CUDA behaviour.
- **Praxy engine:** ResembleAI Chatterbox multilingual + R6 LoRA cloning of
  the uploaded speaker voice for Telugu/Tamil, 24 kHz output.
- **Weights on D:** — `HF_HOME=D:/hf-models` is set at the top of
  `config.py`, so the Chatterbox/Praxy weights are cached on D: and are not
  re-downloaded after the first run.

## Local LLM (optional)

To wire up an LLM without changing this codebase:

1. Run any OpenAI-compatible server locally - e.g. the vLLM command from
   Quick start - or:
   ```bash
   ollama run qwen2.5:3b-instruct
   # serves on http://localhost:11434/v1
   ```
2. In `backend/.env`:
   ```
   LLM_ENABLED=true
   LLM_API_URL=http://127.0.0.1:8001   # local vLLM; no /v1 suffix - worker appends it
   LLM_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ
   ```

If `LLM_ENABLED=false`, the pipeline uses a rule-based responder (echo +
canned acknowledgement in the detected language).

## Evaluation

Place 10-minute Telugu/English test clips in `evaluation/test_clips/` and
matching reference transcripts in `evaluation/ground_truth/`, then:

```bash
make eval-stt      # WER / CER / latency
make eval-tts      # speaker similarity, ASR round-trip WER, MOS
make eval-e2e      # turn latency, hallucination rate
```

See `docs/EVALUATION.md` for full methodology.

## Documentation

- `docs/ARCHITECTURE.md` — System design and data flow
- `docs/DEPLOYMENT.md` — Production deployment, GPU, monitoring
- `docs/API.md` — REST + WebSocket reference
- `docs/EVALUATION.md` — Metrics and methodology

## License

- Code: MIT
- Models: see each model's individual license (Coqui XTTS v2 = CPML, faster-whisper = MIT, Silero VAD = MIT, sherpa-onnx = Apache 2.0)

## Roadmap

- [ ] Twilio phone-call integration (Phase 2)
- [ ] Pyannote 3.1 diarization for multi-speaker calls
- [ ] Prometheus + Grafana monitoring
- [ ] Redis-backed multi-worker queue
