# Architecture

## Overview

Voice Clone Agent is a self-hosted, bilingual (Telugu/English) speech-to-speech
agent that performs zero-shot voice cloning on a single RTX 3060 6 GB GPU.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend (port 8501)                       │
│  Speaker Lab │ Call Center │ History │ Settings                          │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │ HTTP REST + WebSocket
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (port 8000)                            │
│  REST:  /speakers /calls /tts /upload/chunked /health /settings         │
│  WS:    /ws/call (live call streaming)                                   │
│  Conn:  ConnectionManager (MAX_CONCURRENT_CALLS gate)                    │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                Pipeline Orchestrator  (pipeline.py)                      │
│   parallel mode (default): asyncio queues between stages                 │
│   sequential mode:        one segment fully through per stage            │
└──┬──────────┬──────────┬──────────┬──────────┬──────────────────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────┐
│ VAD  │→ │ STT  │→ │ LLM  │→ │ TTS  │→ │ Audio    │
│Silero│  │faster│  │ HTTP │  │ XTTS │  │ chunks   │
│ (CPU)│  │-whisp│  │  API │  │  v2  │  │ via WS   │
└──────┘  └──────┘  └──────┘  └──────┘  └──────────┘
            ~3GB      0GB       ~4GB
            VRAM                VRAM
```

## Components

### Frontend (Streamlit)

Four pages:

1. **Speaker Lab** (`pages/1_Speaker_Lab.py`) — register speakers by uploading
   a 3-10 s reference clip, then test voice cloning with arbitrary text.
2. **Call Center** (`pages/2_Call_Center.py`) — start a live WebSocket call.
   Records audio chunks via `st.audio_input`, sends them to the backend over
   WebSocket, displays the live transcript and bot audio.
3. **History** (`pages/3_History.py`) — list, replay, and read transcripts of
   past calls. Audio playback + JSONL transcript download.
4. **Settings** (`pages/4_Settings.py`) — backend config snapshot, live VRAM
   usage, and chunked-upload UI for 300+ minute audio files.

### Backend (FastAPI)

**REST endpoints** (see `API.md` for full reference):
- `GET /health` — liveness + VRAM + counts
- `GET /settings` — read-only config snapshot
- `GET/POST/DELETE /speakers[/{id}]` — speaker registry CRUD
- `POST /tts/synthesize` — single-utterance TTS with cloning
- `GET /calls`, `GET /calls/{id}`, `GET /calls/{id}/transcript`,
  `GET /calls/{id}/audio`, `DELETE /calls/{id}`
- `POST /upload/chunked/init|{upload_id}/{chunk_index}|complete` — long audio

**WebSocket**: `WS /ws/call` — bidirectional audio streaming for live calls.

### Pipeline (`pipeline.py`)

The orchestrator supports two modes:

#### Parallel mode (default, `PIPELINE_MODE=parallel`)

```
              ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
audio_stream →│  VAD     │───▶│  STT     │───▶│  LLM     │───▶│  TTS     │──▶ audio out
              │ (async)  │ Q1 │ (async)  │ Q2 │ (async)  │ Q3 │ (async)  │
              └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

Each stage is a long-running asyncio task that pulls from an input
`asyncio.Queue(maxsize=8)` and pushes to an output queue. Segments flow
through the pipeline as soon as each stage finishes, so a new utterance
can be transcribed while the previous one is being spoken by TTS.

#### Sequential mode (`PIPELINE_MODE=sequential`)

Each segment fully traverses VAD → STT → LLM → TTS before the next segment
starts processing. Simpler and more deterministic, but higher perceived
latency.

### Workers

#### VAD (`vad_worker.py`)
- Engine: Silero VAD (ONNX, CPU)
- VRAM: 0
- Inputs: 16 kHz int16 PCM stream
- Outputs: speech segments `{t0, t1, samples, is_final}`

#### STT (`stt_worker.py`)
- Engine: faster-whisper `medium`, INT8 quantized
- VRAM: ~3 GB
- Supports both single-segment (`transcribe_pcm`) and streaming-file
  (`transcribe_file`) transcription
- Language: configurable, default `te` (Telugu)

#### LLM (`llm_worker.py`)
- Engine: external OpenAI-compatible HTTP API (optional)
- VRAM: 0
- When `LLM_ENABLED=false`, falls back to a rule-based bilingual responder
  that picks a canned Telugu or English acknowledgement based on the
  detected language of the user's utterance
- Why external? The original plan called for Qwen2.5-3B via llama.cpp, but
  local LLM deployment was deemed too difficult. The HTTP API hook lets you
  plug in Ollama, vLLM, LM Studio, or OpenAI itself without code changes.

#### TTS (`tts_worker.py`)
- Engine: Coqui XTTS v2 (default) or sherpa-onnx VITS (fallback)
- VRAM: ~4 GB (XTTS) or ~1 GB (sherpa)
- Zero-shot cloning: pass `speaker_ref_wav` path; XTTS conditions on the
  3-10 s reference clip at synthesis time
- Streams audio in 20 ms chunks for low-latency playback

### Speaker Registry (`speaker_registry.py`)

On-disk layout:
```
data/speakers/{speaker_id}/
    meta.json     # display_name, language, created_at, ref_duration_s
    ref.wav       # 16 kHz mono reference clip
```

For XTTS no separate embedding file is needed — the reference clip is the
conditioning input. The `ref.wav` is loaded fresh on each synthesis call
(small file, negligible overhead).

### Call Logger (`call_logger.py`)

On-disk layout:
```
data/calls/{call_id}/
    audio.wav          # mixed call audio (16 kHz mono)
    transcript.jsonl   # one JSON line per segment
    meta.json          # call metadata
```

SQLite index at `data/calls.db` for fast listing. Schema:

```sql
CREATE TABLE calls (
    call_id TEXT PRIMARY KEY,
    speaker_id TEXT NOT NULL,
    title TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_s REAL DEFAULT 0,
    audio_path TEXT,
    transcript_path TEXT,
    segment_count INTEGER DEFAULT 0
);
```

### WebSocket Handler (`websocket_handler.py`)

Protocol:
- Client sends text frame `{"type":"hello","speaker_id":"...","title":"..."}`
- Server responds `{"type":"hello","data":{"message":"connected"}}`
- Client streams binary frames of int16 PCM 16 kHz mono
- Server sends text frames: `transcript`, `llm`, `audio` (with `pcm_int16`
  field as a JSON list), `audio_end`, `status`, `error`, `call_end`
- Client sends `{"type":"end"}` to gracefully end

`ConnectionManager` enforces `MAX_CONCURRENT_CALLS`. Additional callers get
a 1013 "try again later" close code.

## VRAM budget

| Mode            | VAD | STT  | LLM | TTS  | Peak |
|-----------------|-----|------|-----|------|------|
| Parallel + XTTS | 0   | 3 GB | 0   | 4 GB | 7 GB ⚠️ |
| Parallel + sherpa | 0 | 3 GB | 0   | 1 GB | 4 GB ✅ |
| Sequential + XTTS | 0 | 3→0  | 0   | 0→4  | 4 GB ✅ |
| Sequential + sherpa | 0 | 3→0 | 0  | 0→1  | 3 GB ✅ |

For a 6 GB card with the default `parallel + xtts` config, the STT stage
uses INT8 quantization and aggressive `torch.cuda.empty_cache()` calls
between heavy transcriptions. If you hit OOM, either:

1. Switch to `PIPELINE_MODE=sequential` (recommended), or
2. Switch to `TTS_ENGINE=sherpa` (smaller VRAM footprint), or
3. Use a smaller STT model: `STT_MODEL=small` (1.5 GB) or `STT_MODEL=base` (0.5 GB)

## Long audio handling (300+ minutes)

For uploaded long recordings, the pipeline bypasses VAD/LLM/TTS and runs
STT-only in streaming mode:

1. Frontend splits the file into 5-min chunks
2. POSTs each chunk to `/upload/chunked/{upload_id}/{chunk_index}`
3. POSTs `/upload/chunked/complete` — backend concatenates chunks and
   starts a background transcription task
4. `STTWorker.transcribe_file` reads the audio in 30-second windows,
   transcribes each window, and appends a JSONL line to the call's
   transcript file
5. Frontend can poll `GET /calls/{call_id}` to see segment count growing

Memory: peak RAM usage is bounded by the 30-second audio window (~1 MB at
16 kHz mono int16) plus the model weights. A 5-hour recording uses the
same peak memory as a 1-minute recording.
