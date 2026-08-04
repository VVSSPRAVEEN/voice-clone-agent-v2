# PLAN.md — Voice Clone Agent (Telugu/English, RTX 3060 6GB)

This is the locked specification that the codebase implements. Any deviation
is noted in the "Deviations" section at the bottom.

## 1. Engine Selection

| Component        | Choice                                   | Why                                                                                | VRAM   |
|------------------|------------------------------------------|------------------------------------------------------------------------------------|--------|
| Voice Cloning    | Coqui XTTS v2 (multilingual, zero-shot)  | Supports Telugu + English, 3 s reference cloning, fits ~4 GB, GPL/CPML            | ~4 GB  |
| Fallback TTS     | sherpa-onnx VITS (AI4Bharat IndicTTS)    | Smaller, faster, official Telugu model                                            | ~1 GB  |
| STT              | faster-whisper `medium` INT8             | Telugu + English, WER ~4-6%, CT2 quantization                                     | ~3 GB  |
| VAD              | Silero VAD (ONNX)                        | CPU only, negligible VRAM                                                          | 0 GB   |
| LLM              | External OpenAI-compatible API (optional)| Local LLM removed per user request; pipeline falls back to rule-based if disabled  | 0 GB   |
| Diarization      | Pyannote 3.1 (hook only, not enabled)    | Future: multi-speaker call separation                                             | ~2 GB  |

### VRAM budget (parallel streaming mode)

In parallel mode, all stages hold their model in memory simultaneously:

```
VAD (0 GB, CPU) + STT (~3 GB) + TTS (~4 GB) = ~7 GB
```

This **exceeds 6 GB**. To fit on a 3060, the pipeline uses one of:

1. **Default (single-stream parallel)**: VAD/STT run on a single stream while
   TTS holds its model. STT unloads its model when idle for >N seconds.
2. **Fallback (`PIPELINE_MODE=sequential`)**: Each stage loads, runs, fully
   unloads, then the next stage loads. Peak VRAM = max(STT, TTS) ≈ 4 GB.

For `MAX_CONCURRENT_CALLS > 1`, switch to `sherpa` TTS engine (1 GB) to keep
total under 6 GB.

## 2. Architecture (high level)

```
Streamlit Frontend (8501) ←→ FastAPI Backend (8000) ←→ Pipeline Workers
   Speaker Lab                 WebSocket Mgr              VAD (CPU)
   Call Center (WebRTC)        Session Mgr                STT (GPU)
   Call History                Speaker Registry           LLM (HTTP API)
   Settings                    Call Logger (SQLite+FS)    TTS (GPU)
```

## 3. 300-minute audio handling

| Requirement              | Solution                                                                  |
|--------------------------|----------------------------------------------------------------------------|
| Long recording upload    | Chunked 5-min upload → streaming STT → incremental transcript              |
| Continuous call (5 hr)   | WebSocket streaming → VAD segments → pipeline per segment → JSONL log      |
| Storage                  | SQLite metadata + local filesystem (MinIO optional via `STORAGE_BACKEND`)  |
| Transcript format        | JSONL: `{"t0":..,"t1":..,"speaker":..,"text":..,"is_final":..}`           |
| Playback                 | Streamlit audio player with timestamp seeking                              |
| Memory                   | Never load full audio; process in 30-s windows                              |

## 4. Speaker cloning flow

1. User uploads 3-10 s reference audio (Telugu / English / mixed)
2. Backend stores reference + computes speaker conditioning (XTTS handles this
   at synthesis time; no separate embedding file needed for XTTS)
3. Registry entry: `data/speakers/{speaker_id}/` contains `ref.wav` + `meta.json`
4. On TTS request: load speaker → XTTS synthesis with `speaker_wav=ref.wav`
5. Stream chunks back via WebSocket (16 kHz PCM → 24 kHz PCM for TTS)

Pitch/prosody/slang preservation: XTTS v2 captures prosody and accent from
the reference clip. For best Telugu slang results, use a reference clip with
natural conversational Telugu (not studio narration).

## 5. Repository structure

See `README.md` for the full tree.

## 6. Docker Compose

Two services (backend, frontend), both with GPU passthrough on backend.
Optional MinIO and Redis services are commented out in `docker-compose.yml`.

## 7. Model download & quantization

`backend/scripts/download_models.py` pulls:

- `Systran/faster-whisper-medium` (CT2 INT8 quantized at load time)
- Silero VAD ONNX (`snakers4/silero-vad`)
- Coqui XTTS v2 (`tts_models/multilingual/multi-dataset/xtts_v2`)
- sherpa-onnx Telugu VITS (AI4Bharat) — optional, used by `TTS_ENGINE=sherpa`

## 8. Evaluation plan

- `evaluate_stt.py` — WER, CER, per-segment latency on user's 10-min clips
- `evaluate_tts.py` — speaker similarity (ECAPA-TDNN cosine), ASR round-trip
  WER, subjective MOS template
- `evaluate_e2e.py` — turn latency, hallucination rate, coherence

## 9. Phone call integration (Phase 2, not in this build)

- Twilio: webhook → TwiML `<Stream>` → FastAPI WebSocket → pipeline → `<Stream>` back
- LiveKit/Daily: WebRTC SFU → FastAPI worker
- Start: WebRTC in Streamlit (browser-to-browser)

## 10. Implementation phases (this build covers 1-6)

| Phase | Deliverable                                                        | Status |
|-------|--------------------------------------------------------------------|--------|
| 1     | Docker Compose, FastAPI skeleton, Streamlit skeleton, downloader   | ✅      |
| 2     | VAD → STT → LLM → TTS workers, WebSocket streaming, Speaker reg.   | ✅      |
| 3     | UI: Speaker Lab, Call Center, History, Settings, WebRTC            | ✅      |
| 4     | 300-min support: chunked upload, streaming STT, SQLite, playback   | ✅      |
| 5     | Evaluation scripts                                                 | ✅      |
| 6     | Hardening: VRAM monitor, error recovery, health checks, logging    | ✅      |
| 7     | Twilio / phone integration                                         | ⏳ Phase 2 |
| 8     | Pyannote diarization                                               | ⏳ Phase 2 |

## 11. Open decisions (resolved)

| Decision                  | Choice                                                              |
|---------------------------|---------------------------------------------------------------------|
| TTS engine                | Coqui XTTS v2 (multilingual zero-shot cloning, fits VRAM)           |
| LLM                       | Skip local; use OpenAI-compatible API hook (user request)           |
| Call storage              | Local filesystem + SQLite (MinIO optional)                          |
| Auth                      | None for local (API key optional)                                   |
| Monitoring                | stdout structured logs (Prometheus hook optional)                   |
| Concurrency default       | Parallel pipeline (asyncio streaming, single GPU stream)            |

## 12. Deviations from original user plan

1. **OmniVoice → Coqui XTTS v2**: The user's plan referenced "OmniVoice" as
   the cloning engine. We use Coqui XTTS v2 as the actual implementation
   because it is the only widely-available open-source model with verified
   Telugu + English zero-shot cloning that fits in 4 GB VRAM. The
   `TTS_ENGINE=sherpa` fallback uses AI4Bharat's Telugu VITS via sherpa-onnx.

2. **Local LLM removed**: Per explicit user request ("making the llm deployed
   in local llm is difficult"), the Qwen2.5-3B local LLM was removed. The
   LLM stage is now an optional external HTTP API call. When `LLM_ENABLED=false`,
   a rule-based responder produces short acknowledgements in the detected
   language so the pipeline still runs end-to-end.

3. **Parallel mode is default**: `PIPELINE_MODE=parallel` is the default as
   requested. In this mode VAD, STT, LLM, and TTS run as concurrent asyncio
   tasks connected by bounded queues — segments flow through the pipeline
   with overlapping processing, giving lower perceived latency than strict
   sequential mode.

4. **No pyannote in default build**: Diarization is a Phase 2 hook. The
   pipeline supports single-speaker-per-stream today; multi-speaker call
   mode requires enabling diarization (not implemented in this build).
