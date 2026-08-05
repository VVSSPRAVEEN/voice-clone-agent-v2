# Live Calls & Voice Cloning — User Guide

This document covers the live-call experience (mic → voice-cloned reply),
the kinship-aware greeting behaviour, and how to keep the stack healthy.

## Architecture at a glance

```
Browser mic (Streamlit UI, :8501)
        │  WebSocket /ws/call (int16 PCM, 16 kHz mono; JSON events back)
        ▼
Backend (:8000) — single sequential pipeline, NO VAD
   ├─ STT  → faster-whisper medium/int8 (Telugu/English) — runs on CPU
   ├─ LLM  → local vLLM (:8001, Qwen2.5-3B-Instruct-AWQ)
   └─ TTS  → hybrid router:
        te/ta  → Praxy (Chatterbox + R6 LoRA) — true voice cloning, 24 kHz
        en/hi  → Coqui XTTS v2 — zero-shot cloning, 24 kHz
        other  → Edge-TTS (online, no cloning)
```

How a turn works (no VAD anywhere): while the mic stream is open the
backend buffers audio. When **~1.5 s of silence** passes (or the client
sends an `end` frame) it runs the whole utterance through
STT → LLM → TTS and streams the reply back in 20 ms PCM chunks. A
"Backend status" panel in the UI shows each stage live
(`listening → stt_transcribing → llm_thinking → tts_synthesizing`).

## Using the Call Center (UI)

1. Open `http://localhost:8501` → **Call Center** (use localhost — mic only
   works on localhost or HTTPS).
2. Pick the speaker whose cloned voice should answer.
3. Choose a call mode:

   - **📝 Record clip** — press **Start call session**, press **Click to
     record**, speak, stop. The clip is sent automatically (no manual end
     needed), the reply arrives as audio + transcript, and the next
     recording reconnects automatically.
   - **🔴 Live streaming (WebRTC)** — hands-free full-duplex call. Click
     **▶ Start**, grant mic access, and just talk. Pause >1.5 s between
     sentences; the agent replies in the cloned voice, multi-turn on one
     connection. Stop with the component's ⏹ button.

4. Expected timing (warm engines): STT ~10–20 s (CPU), LLM ~1–2 s,
   TTS ~15–30 s (XTTS CPU) / ~70 s (Praxy first synthesis).

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
- STT: faster-whisper medium on CPU (~3 min first time, faster from cache)

`/health` responds while preloading. Watch the uvicorn console (current
deployment logs to `D:\backendN_err.log` / `D:\backendN_out.log` — the
number increments per restart) for `Praxy preload complete` /
`Engine preload complete`.

## Connection handling & multi-turn

- The WebSocket stays open across turns: after each reply the server sends
  `{"type":"call_end"}` and keeps listening on the same socket. A client
  `{"type":"end"}` frame or a disconnect ends the session.
- If the browser drops the connection mid-turn, the backend **cancels the
  pipeline** instead of wasting a long synthesis on a ghost client
  (`backend/app/websocket_handler.py`).
- The UI shows `Connection closed ... press Start call session to
  reconnect` when the socket dies; record-clip mode reconnects
  automatically for the next recording.

## GPU memory budget (6 GB card)

Everything is tuned to fit a 6 GB GPU:

- vLLM runs with `--gpu-memory-utilization 0.45` (~2.7 GB)
- Praxy on CUDA (~3.1 GB)
- STT deliberately on CPU (frees ~1.2 GB) — see `backend/.env`
  `STT_DEVICE_CUDA=false`

Do not raise the vLLM utilization above ~0.5 while Praxy is loaded —
the GPU thrashes and LLM replies jump from ~1 s to 60 s.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Microphone not allowed" | Use `http://localhost:8501`, not the LAN IP. Check the browser site settings → Microphone → Allow. For other devices, put the UI behind HTTPS (Caddy/nginx). |
| No reply after recording | Check the **Backend status** panel: stuck at `listening` = audio not arriving (mic permission); `stt_transcribing` for >60 s = first CPU STT load; `no_speech_detected` = quiet clip. |
| WebRTC mode: mic echoes | Old page cache — hard-refresh (Ctrl+F5). The component returns silence while no reply audio is queued. |
| Reply is in Edge voice, not cloned | Praxy failed to load/route (log: `Praxy TTS failed ... falling back to Edge-TTS`). Check free VRAM (`/health` → vram_free_mb); Praxy needs ≥3.6 GB. |
| vLLM not answering | vLLM must run with `HF_HOME=D:/hf-models` and `HF_HUB_OFFLINE=1` (see README quick start). Backend falls back to canned replies when the LLM is down (`llm(fallback)` in the status panel). |
| "Server busy 1013" | `MAX_CONCURRENT_CALLS=1` — only one call at a time; close the other tab/session. |
| Backend unresponsive | Restart it (kill `uvicorn app.main` processes, rerun the uvicorn command). First 3 min after boot are preload-heavy. |

## Running everything (3 terminals)

```powershell
# 1. Backend (:8000)
cd D:\voice-clone-agent\backend
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. LLM (:8001) — optional but recommended
$env:HF_HOME="D:\hf-models"; $env:HF_HUB_OFFLINE="1"
D:\vllm-venv\Scripts\python.exe -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-3B-Instruct-AWQ --port 8001 --gpu-memory-utilization 0.45 --max-model-len 1024 --enforce-eager

# 3. UI (:8501)
cd D:\voice-clone-agent\frontend\streamlit_app
D:\voice-clone-agent\backend\.venv\Scripts\python.exe -m streamlit run main.py --server.port 8501 --server.headless true
```

Config lives in `D:\voice-clone-agent\backend\.env` (API key `change-me`,
`TTS_ENGINE=hybrid`, `STT_DEVICE_CUDA=false`, `PIPELINE_MODE=sequential`).
The frontend requires `websockets>=13` in the backend venv (client pings are
disabled to prevent socket drops).
