# Live Calls & Voice Cloning — User Guide

This document covers the live-call experience (mic → voice-cloned reply),
the kinship-aware greeting behaviour, and how to keep the stack healthy.

## Architecture at a glance

```
Browser mic (Streamlit UI, :8501)
        │  WebSocket  /ws/call  (16 kHz int16 PCM chunks)
        ▼
Backend (:8000)
   ├─ VAD  → detects speech segments
   ├─ STT  → faster-whisper (Telugu/English)
   ├─ LLM  → local vLLM (:8001, Qwen2.5-3B-Instruct-AWQ)
   └─ TTS  → hybrid router:
        te/ta  → Praxy (Chatterbox + R6 LoRA) — true voice cloning, 24 kHz
        en/hi  → Coqui XTTS v2 — zero-shot cloning, 24 kHz
        other  → Edge-TTS (online, no cloning)
```

## Using the Call Center (UI)

1. Open `http://localhost:8501` → **Call Center** (use localhost — mic only
   works on localhost or HTTPS).
2. Pick the speaker whose cloned voice should answer.
3. Press **Start call session**.
4. Press **Click to record**, speak Telugu or English, press stop.
5. Wait 10–60 s — the bot's answer arrives as audio + transcript.

Engine warm-up: the backend preloads Edge, Praxy and XTTS in the background
at startup (see *Preloading*). During the first ~3 minutes after a backend
restart, the first call may still wait for Praxy to finish loading.

## Kinship-aware greetings

The LLM system prompt contains a Telugu kinship table. When a caller greets
the agent with a relation term, the agent replies with the correct
reciprocal term, e.g.:

| Caller says | Agent replies with |
|---|---|
| mavayya (maternal uncle) | menalludaa (nephew) |
| babai / chinnanna / peddananna (paternal uncle) | menalludaa |
| athamma / peddamma / pinni (aunt) | menalludaa |
| anna (elder brother) | thammudaa (younger brother) |
| akka (elder sister) | chellellaa (younger sister) |
| thata (grandfather) / ammamma (grandmother) | manavaduu (grandson) |
| bava / vadhina (in-laws) | maradalaa (sister-in-law) |
| mamayya / atta (in-laws) | alludaa / kodalaa (son/daughter-in-law) |

The table lives in `backend/app/config.py` (`llm_system_prompt`). Add or
change entries there — no code changes needed.

## Registering a speaker from the mic

**Speaker Lab** page → *Register new speaker* → click **Record from mic**
(3–10 s clean speech), or upload a file. The clip is stored as the
reference voice for cloning.

## Preloading (why the first call is fast)

`backend/app/tts_worker.py` → `preload()` runs at backend startup as a
background task (`backend/app/main.py` → `_preload_engines`):

- Edge-TTS: instant
- Praxy: loads the Chatterbox backbone + R6 LoRA (on CUDA if ≥3.6 GB free,
  else CPU — CPU takes ~3–6 min)
- XTTS v2: loads on CPU (~15 s)

`/health` responds while preloading. Watch `D:\backend3_err.log` (or the
uvicorn console) for `Praxy preload complete` / `Engine preload complete`.

## Connection loss handling

- If the browser drops the connection mid-call, the backend now **cancels
  the pipeline** instead of wasting a long CPU synthesis on a ghost client
  (`backend/app/websocket_handler.py`).
- The UI shows `Connection closed ... press Start call session to reconnect`
  when the socket dies, and tells you to reconnect + record again.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Microphone not allowed" | Use `http://localhost:8501`, not the LAN IP. Check the browser site settings → Microphone → Allow. For other devices, put the UI behind HTTPS (Caddy/nginx). |
| No reply after recording | First call right after backend restart → Praxy still preloading (wait, or watch the log). Otherwise check the backend log for `WS send failed` (client disconnected). |
| Reply is in Edge voice, not cloned | Praxy failed to load/route (log: `Praxy TTS failed ... falling back to Edge-TTS`). Check free VRAM (`/health` → vram_free_mb); Praxy needs ≥3.6 GB. |
| vLLM not answering | vLLM must run with `HF_HOME=D:/hf-models` and `HF_HUB_OFFLINE=1` (see README quick start). Backend falls back to canned replies when the LLM is down. |
| Backend unresponsive | Restart it (kill `uvicorn app.main` processes, rerun the uvicorn command). First 3 min after boot are preload-heavy. |

## Running everything (3 terminals)

```powershell
# 1. Backend (:8000)
cd D:\voice-clone-agent\backend
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. LLM (:8001) — optional but recommended
$env:HF_HOME="D:\hf-models"; $env:HF_HUB_OFFLINE="1"
D:\vllm-venv\Scripts\python.exe -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-3B-Instruct-AWQ --port 8001 --gpu-memory-utilization 0.7 --max-model-len 1024 --enforce-eager

# 3. UI (:8501)
cd D:\voice-clone-agent\frontend\streamlit_app
D:\voice-clone-agent\backend\.venv\Scripts\python.exe -m streamlit run main.py --server.port 8501 --server.headless true
```

Config lives in `D:\voice-clone-agent\backend\.env` (API key `change-me`,
`TTS_ENGINE=hybrid`, etc.).
