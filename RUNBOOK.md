# Voice Clone Agent — RUNBOOK

Complete operations guide: what the system is, how to run every piece on this
Windows machine, all environment variables, and how to test.

---

## 1. What this is

Local Telugu/English voice agent:

```
mic / audio file  →  VAD (Silero)  →  STT (faster-whisper, GPU)
                                    →  LLM (Qwen2.5-3B via local vLLM, bilingual)
                                    →  TTS (XTTS voice-clone | Edge-TTS Telugu)
                                    →  audio out
```

- Live calls over WebSocket (`/ws/call`)
- Long audio files end-to-end (`POST /analyze`) — STT the whole file, one
  intelligent LLM reply, spoken back in the speaker's reference voice
- Streamlit UI on port 8501

## 2. Machines & prerequisites

- RTX 3060 Laptop 6 GB, driver 595.97, Python 3.12.7 (Anaconda)
- ffmpeg: `D:\tools\ffmpeg-9.0-essentials_build\bin` (must be on PATH)
- Two venvs, both already set up:
  - Backend: `D:\voice-clone-agent\backend\.venv`
  - vLLM:    `D:\vllm-venv`

## 3. How to run (3 processes)

### A. vLLM (LLM server, port 8001)

```powershell
$env:HF_HOME = "D:\voice-clone-agent\data\models\hf"
$env:PYTHONUTF8 = "1"
D:\vllm-venv\Scripts\python.exe -m vllm.entrypoints.cli.main serve `
  Qwen/Qwen2.5-3B-Instruct-AWQ --port 8001 --gpu-memory-utilization 0.7 `
  --max-model-len 1024 --enforce-eager
```

> Must launch via `D:\vllm-venv\Scripts\python.exe -m vllm.entrypoints.cli.main`
> (NOT the `vllm.exe` shim). Resident VRAM ≈ 4.5 GB of 6 GB.

### B. Backend (port 8000)

```powershell
$env:Path = "D:\tools\ffmpeg-9.0-essentials_build\bin;$env:Path"
$env:PYTHONUTF8 = "1"
cd D:\voice-clone-agent\backend
D:\voice-clone-agent\backend\.venv\Scripts\python.exe -m uvicorn app.main:app `
  --host 0.0.0.0 --port 8000
```

Health check: `curl http://127.0.0.1:8000/health` → `tts_engine:"edge"`.

### C. Streamlit UI (port 8501)

```powershell
cd D:\voice-clone-agent\frontend
D:\voice-clone-agent\backend\.venv\Scripts\streamlit.exe run streamlit_app\main.py
```

Open http://localhost:8501 (pages: Speaker Lab, Call Center, History, Settings).

## 4. Environment variables (`backend/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8000` | Backend bind |
| `PIPELINE_MODE` | `parallel` | `parallel` (streaming queues) or `sequential` |
| `MAX_CONCURRENT_CALLS` | `1` | Concurrent call limit |
| `DATA_DIR` | `D:\voice-clone-agent\data` | Data root |
| `SPEAKERS_DIR` | `...\data\speakers` | Speaker ref clips |
| `CALLS_DIR` | `...\data\calls` | Call transcripts/audio |
| `MODELS_DIR` | `...\data\models` | Model cache (HF, faster-whisper) |
| `DB_PATH` | `...\data\calls.db` | SQLite call registry |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU id |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:128` | CUDA allocator |
| `FORCE_CPU` | `false` | Force CPU override |
| `STT_MODEL` | `medium` | faster-whisper size |
| `STT_COMPUTE_TYPE` | `int8` | int8/fp16/float32 |
| `STT_DEVICE_CUDA` | `true` | GPU for whisper |
| `STT_LANGUAGE` | `auto` | `auto` = per-segment Telugu/English detect; `te` = force |
| `STT_BEAM_SIZE` | `1` | 1 fast / 5 accurate |
| `VAD_THRESHOLD` | `0.5` | Silero VAD energy threshold |
| `VAD_MIN_SPEECH_MS` | `250` | Min speech to trigger |
| `VAD_MAX_SPEECH_MS` | `30000` | Max speech per segment |
| `VAD_SILENCE_MS` | `500` | Silence to end segment |
| `TTS_ENGINE` | `edge` | `edge` \| `xtts` \| `sherpa` \| `ai4bharat` |
| `XTTS_MODEL` | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui XTTS v2 |
| `XTTS_DEVICE` | `cpu` | `cpu` (safe with vLLM) or `cuda` |
| `XTTS_LANGUAGE` | `te` | Fallback TTS language iso code |
| `edge_voice` | `te-IN-MohanNeural` | Edge Telugu male (female: `te-IN-ShrutiNeural`) |
| `LLM_ENABLED` | `true` | Use vLLM; `false` = rule-based fallback |
| `LLM_MODEL` | `Qwen/Qwen2.5-3B-Instruct-AWQ` | vLLM served model |
| `LLM_API_URL` | `http://127.0.0.1:8001` | vLLM OpenAI endpoint |
| `LLM_SYSTEM_PROMPT` | *bilingual agent* | System prompt |
| `LLM_MAX_TOKENS` | `128` | Reply length cap |
| `LLM_TEMPERATURE` | `0.7` | Sampling temp |
| `SAMPLE_RATE` | `16000` | STT/VAD rate |
| `TTS_SAMPLE_RATE` | `24000` | TTS native rate |
| `CHANNELS` / `CHUNK_MS` | `1` / `20` | Audio format |
| `STORAGE_BACKEND` | `local` | `local` or `minio` |
| `AUTH_MODE` | `none` | `none` or `apikey` |
| `API_KEY` | `change-me` | Used when AUTH_MODE=apikey |
| `HF_TOKEN` | *(secret)* | HuggingFace token for gated models |

> `HF_TOKEN` and `API_KEY` are secrets. `.env` is gitignored — never commit it.
> Use `backend/.env.example` as the committed template.

## 5. Voice cloning

`docs/VOICE_CLONING.md` covers it in depth. Summary:
- Speaker = a folder `data/speakers/{id}/` with `ref.wav` (3-10 s) + `meta.json`.
- `TTS_ENGINE=xtts` → true zero-shot cloning from `ref.wav`.
- `TTS_ENGINE=edge` (current) → preset neural voices, **no cloning**, but best
  Telugu. Reply language is auto-detected by script (Telugu/Tamil/Hindi/English).

## 6. Endpoints (REST)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, VRAM, engine, counts |
| GET | `/settings` | Config snapshot |
| GET/POST/DELETE | `/speakers...` | Speaker CRUD |
| POST | `/tts/synthesize` | One-shot TTS (voice clone) |
| POST | `/analyze` | Long file → STT → LLM → TTS reply (returns `call_id`) |
| POST | `/upload/chunked/*` | Chunked upload for 300+ min audio (STT-only) |
| GET | `/calls` , `/calls/{id}/transcript` , `/calls/{id}/audio` | History |
| WS | `/ws/call` | Live call streaming |

Example analyze:
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "speaker_id=test2" -F "audio=@meeting.wav" -F "title=meeting"
# -> {"call_id":"call_...","analysis":"started"}
curl http://127.0.0.1:8000/calls/call_.../transcript
```

## 7. Tests / evaluation

- `evaluation/e2e_te.py` — live-pipeline Telugu E2E (`import os; os._exit(0)` at end).
- `evaluation/e2e_en.py` — English variant.
- `evaluation/audio_compare.py` — acoustic compare between two WAVs
  (pitch, pace, jitter/shimmer smoothness, F1/F2 accent, MFCC similarity):

```bash
cd D:\voice-clone-agent\evaluation
python audio_compare.py fileA.wav fileB.wav
```

- Run E2E with CWD = `backend` (`.env` is located there) and `PYTHONUTF8=1`.

## 8. Known limits

- XTTS has no Telugu; Edge-TTS won't clone → Telugu+clone = future `praxy`
  (`Praxel/praxy-voice-r6`, Chatterbox LoRA) or `DhVaani-0.5` / `IndicF5`.
- XTTS on this GPU runs CPU-only when vLLM is resident (~1.5 GB VRAM free).
- faster-whisper `medium` on synthetic clips can mis-detect language (Hindi for
  some Telugu frames) — real speech is more reliable; tune `STT_LANGUAGE`.