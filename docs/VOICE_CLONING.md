# Voice Cloning — How It Works

This document explains exactly how "voice cloning" works in this project, what
is and is not cloned, and how to get real cloning working.

---

## TL;DR

| Engine | Voice cloning? | How |
|---|---|---|
| `xtts` (Coqui XTTS v2) | **Yes — real zero-shot cloning** | 3–10 s reference clip is fed to XTTS at synthesis time; it extracts a speaker embedding and speaks in that voice |
| `edge` (Microsoft Edge-TTS, online) | **No** | Fixed neural voices (`te-IN-MohanNeural`, `en-US-JennyNeural`, ...). The reference clip is **ignored** |
| `sherpa` (sherpa-onnx VITS) | No | Fixed preset voices |
| `ai4bharat` (FastPitch+HiFi-GAN) | No (and currently broken in this fork) | Fixed speaker id |

> **Current state of this machine:** `TTS_ENGINE=edge`, so cloning is NOT
> active right now — Edge-TTS speaks with its own preset voices. Switch
> `TTS_ENGINE=xtts` in `backend/.env` to enable real cloning (see below).

---

## 1. The speaker concept

Every "speaker" in the system is just a folder:

```
data/speakers/test1/
├── meta.json     # speaker_id, display_name, language, ref_duration_s, created_at
└── ref.wav       # the 3-10 s reference recording (16 kHz mono, converted if needed)
```

Created via:

```bash
curl -X POST http://localhost:8000/speakers \
  -F "speaker_id=arjun" \
  -F "display_name=Arjun" \
  -F "language=te" \
  -F "ref_audio=@/path/to/arjun_voice.wav"
```

- Any non-WAV upload (mp3, m4a, ogg) is converted to 16 kHz mono WAV via
  pydub/ffmpeg (`speaker_registry.py:create`).
- `ref_duration_s` is recorded — 3–10 s is the sweet spot for XTTS cloning.

Relevant file: `backend/app/speaker_registry.py`.

---

## 2. How XTTS cloning works (the real one)

Coqui XTTS v2 is a zero-shot voice cloning model. There is **no training
step** — cloning happens entirely at synthesis time:

1. You provide a 3–10 s clip of the target voice (`ref.wav`).
2. XTTS's **voice encoder** converts that clip into a fixed-length speaker
   embedding (a vector describing timbre, pitch, accent, speaking style).
3. That embedding conditions the autoregressive decoder, so every sentence
   synthesized *for that speaker* is spoken in that voice.

In this codebase, the flow is:

```
pipeline.py
  └─ tts.synthesize(text, speaker_ref_wav=..., language=...)
       └─ tts_worker.py  _synth_xtts()
            └─ self._xtts.tts(
                 text=text,
                 speaker_wav=str(speaker_ref_wav),   # <- the reference clip
                 language=lang_code,
               )
```

- `speaker_ref_wav` is always resolved from the speaker registry before the
  pipeline runs (`pipeline.py:82-83`, `analyze_file` too).
- Cloning works for **any voice you have a clip for** — no identity
  registration, no training, instant.
- XTTS supports 17 languages including **English and Hindi but NOT Telugu**
  (see `docs/` notes on Telugu).

To enable:

```bash
# backend/.env
TTS_ENGINE=xtts
XTTS_DEVICE=cpu        # or cuda (needs ~4 GB VRAM free; vLLM uses ~4.5 GB of 6 GB)
XTTS_LANGUAGE=te       # fallback language code for voice selection
```

> Note: with vLLM resident on the 6 GB GPU there is ~1.5 GB free, so XTTS
> typically runs on **CPU** on this machine (~25 s per sentence). This is
> correct behavior — the system already handles it.

---

## 3. How Edge-TTS fits in (the current default)

Edge-TTS (`TTS_ENGINE=edge`) is a free online neural TTS from Microsoft. It:

- Has **native Telugu voices** (`te-IN-MohanNeural` male, `te-IN-ShrutiNeural`
  female) and English (`en-US-JennyNeural`).
- Does **not** clone your reference voice. `speaker_ref_wav` is accepted but
  unused.

Voice selection is **auto-detected per reply** (`tts_worker.py:_pick_edge_voice`):

| Reply content | Voice used |
|---|---|
| Contains Telugu script (U+0C00–U+0C7F) | `SETTINGS.edge_voice` (default `te-IN-MohanNeural`) |
| Contains Tamil script | `ta-IN-PallaviNeural` |
| Contains Devanagari | `hi-IN-MadhurNeural` |
| Pure Latin text + STT said English | `en-US-JennyNeural` |
| Otherwise | `SETTINGS.edge_voice` |

This script-based detection is deliberately **independent of the LLM's
language tag**, which can be wrong for code-mixed replies (e.g.
Telugu text + one English word).

Relevant file: `backend/app/tts_worker.py` (`_pick_edge_voice`, `_synth_edge`).

---

## 4. What "clone" means in each call path

### Real-time call (`run_streaming`, WebSocket `/ws/call`)
- Client connects with `speaker_id` → registry resolves `ref.wav` → every bot
  reply is synthesized with that reference (XTTS) or the detected preset
  voice (Edge).

### Long audio file (`analyze_file`, POST `/analyze`)
- Upload the file (chunked upload for 300+ min) → STT transcribes the whole
  thing in 30 s chunks → the full transcript is sent to the LLM once → the
  single intelligent reply is synthesized with the speaker's voice.
- Same `speaker_ref_wav` resolution as streaming.

### Direct TTS endpoint (`POST /tts/synthesize`)
- One-shot: `text + speaker_id (+ language)` → WAV bytes with
  `X-TTS-Engine` / `X-TTS-Latency-ms` headers.

---

## 5. Limitations (be honest with users)

- **Telugu voice cloning is not possible on the current stack.** XTTS lacks
  Telugu; Edge-TTS won't clone. Options:
  - Today: Telugu = Edge-TTS preset voices (very good quality, fixed voice).
  - Future: `Praxel/praxy-voice-r6` (Chatterbox LoRA + 8–11 s Telugu
    reference clip) does true zero-shot Telugu cloning — vet it before
    integrating.
- A poor reference clip (noisy, music, multiple speakers, <3 s or >10 s)
  degrades clone quality.
- XTTS is CPU-bound here (~25 s/sentence) — that is an inference-speed
  tradeoff, not a correctness one.

---

## 6. Files that matter

| File | Role |
|---|---|
| `backend/app/speaker_registry.py` | Store/convert/validate reference clips |
| `backend/app/tts_worker.py` | XTTS/Edge/sherpa synthesis + script-based voice routing |
| `backend/app/pipeline.py` | `run_streaming` (live), `analyze_file` (long files) |
| `backend/app/main.py` | REST/WS endpoints (speakers, TTS, upload, analyze) |
| `backend/.env` | `TTS_ENGINE`, `XTTS_DEVICE`, `edge_voice`, `HF_TOKEN`, ... |
