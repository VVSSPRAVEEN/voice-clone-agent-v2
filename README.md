# Voice Clone Agent (Telugu / English) — RTX 3060 6GB

A self-hosted, zero-shot voice cloning agent that supports bilingual
(Telugu + English) speech-to-speech conversation, call recording, transcript
playback, and 300-minute long-audio handling. Designed to fit on a single
RTX 3060 6 GB GPU.

> **Note on local LLM:** Per project decision, the local LLM (Qwen2.5-3B via
> llama.cpp) was removed because of deployment difficulty. The LLM stage is
> now an **optional external API hook** (any OpenAI-compatible endpoint such
> as Ollama, vLLM, LM Studio, or OpenAI itself). When no LLM is configured,
> the pipeline falls back to a simple rule-based responder so the rest of the
> system (STT, voice cloning, call logging, UI) remains fully functional.

---

## Quick start

```bash
git clone <this repo>
cd voice-clone-agent
cp .env.example .env                # review / edit
make build                          # build docker images
make download                       # download all model weights (~5 GB)
make up                             # start backend + frontend
# Backend:  http://localhost:8000    (FastAPI + WebSocket)
# Frontend: http://localhost:8501    (Streamlit UI)
```

## What's inside

| Component        | Choice                                  | VRAM   |
|------------------|-----------------------------------------|--------|
| STT              | faster-whisper `medium` INT8            | ~3 GB  |
| VAD              | Silero VAD (ONNX, CPU)                  | ~0 GB  |
| TTS (cloning)    | Coqui XTTS v2 (multilingual, zero-shot) | ~4 GB  |
| TTS (alt)        | sherpa-onnx VITS (Telugu/English)       | ~1 GB  |
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

All runtime config lives in `.env` (see `.env.example`). The most important
toggles:

| Variable           | Default      | Notes                                       |
|--------------------|--------------|---------------------------------------------|
| `PIPELINE_MODE`    | `parallel`   | `parallel` (streaming) or `sequential`      |
| `MAX_CONCURRENT_CALLS` | `1`      | Higher values require more VRAM             |
| `STT_MODEL`        | `medium`     | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `STT_LANGUAGE`     | `te`         | Telugu; use `auto` for autodetect           |
| `TTS_ENGINE`       | `xtts`       | `xtts` (Coqui) or `sherpa` (VITS)           |
| `LLM_ENABLED`      | `false`      | Set `true` and configure `LLM_API_BASE`     |

## Local LLM (optional)

To wire up an LLM without changing this codebase:

1. Run any OpenAI-compatible server locally, e.g.
   ```bash
   ollama run qwen2.5:3b-instruct
   # serves on http://localhost:11434/v1
   ```
2. In `.env`:
   ```
   LLM_ENABLED=true
   LLM_API_BASE=http://host.docker.internal:11434/v1
   LLM_API_KEY=ollama
   LLM_MODEL=qwen2.5:3b-instruct
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
