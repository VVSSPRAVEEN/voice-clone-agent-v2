# API Reference

Base URL: `http://localhost:8000`

All endpoints accept/return JSON unless noted. Binary audio endpoints
return `audio/wav`.

When `AUTH_MODE=apikey`, send `X-API-Key: <your-key>` header on every
request.

## Health & Settings

### `GET /health`

Returns liveness + VRAM + counts.

**Response:**
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

### `GET /settings`

Read-only snapshot of the backend configuration.

**Response:**
```json
{
  "pipeline_mode": "parallel",
  "max_concurrent_calls": 1,
  "stt_model": "medium",
  "stt_language": "te",
  "tts_engine": "xtts",
  "xtts_language": "te",
  "llm_enabled": false,
  "llm_model": "qwen2.5-3b-instruct",
  "device": "cuda",
  "vram_total_mb": 6144,
  "vram_used_mb": 0,
  "vram_free_mb": 6144
}
```

## Speakers

### `GET /speakers`

List all registered speakers.

**Response:** `SpeakerOut[]`
```json
[
  {
    "speaker_id": "ramesh_telugu",
    "display_name": "Ramesh (Telugu)",
    "language": "te",
    "ref_duration_s": 6.42,
    "created_at": "2026-01-15T10:30:00Z"
  }
]
```

### `POST /speakers`

Register a new speaker. **Multipart form data.**

**Form fields:**
- `speaker_id` (string) — unique identifier
- `display_name` (string) — human-readable name
- `language` (string) — default TTS language code (`te`, `en`, etc.)
- `ref_audio` (file) — 3-10 s reference clip (wav/mp3/m4a/ogg/flac)

**Response:** `SpeakerOut`

**Example (curl):**
```bash
curl -X POST http://localhost:8000/speakers \
  -F "speaker_id=ramesh_telugu" \
  -F "display_name=Ramesh (Telugu)" \
  -F "language=te" \
  -F "ref_audio=@ref.wav"
```

### `GET /speakers/{speaker_id}`

Get speaker metadata.

**Response:** `SpeakerMeta`
```json
{
  "speaker_id": "ramesh_telugu",
  "display_name": "Ramesh (Telugu)",
  "language": "te",
  "ref_audio_path": "/app/data/speakers/ramesh_telugu/ref.wav",
  "ref_duration_s": 6.42,
  "created_at": "2026-01-15T10:30:00Z"
}
```

### `DELETE /speakers/{speaker_id}`

Delete a speaker and their reference audio.

**Response:** `{"ok": true}`

### `GET /speakers/{speaker_id}/ref.wav`

Download the speaker's reference audio as a WAV file.

**Response:** `audio/wav` binary

## TTS

### `POST /tts/synthesize`

Synthesize a single utterance with voice cloning.

**Request body:**
```json
{
  "text": "నమస్తే, ఎలా ఉన్నారు?",
  "speaker_id": "ramesh_telugu",
  "language": "te"
}
```

(`language` is optional; defaults to the speaker's default language.)

**Response:** WAV audio bytes with headers:
- `X-TTS-Latency-ms`: synthesis time in ms
- `X-TTS-Engine`: `xtts` or `sherpa`

## Calls

### `GET /calls`

List calls, newest first.

**Query params:**
- `limit` (int, default 100, max 500)
- `offset` (int, default 0)

**Response:** `CallOut[]`
```json
[
  {
    "call_id": "call_abc123def456",
    "speaker_id": "ramesh_telugu",
    "title": "Test call 1",
    "started_at": "2026-01-15T10:30:00Z",
    "ended_at": "2026-01-15T10:32:15Z",
    "duration_s": 135.0,
    "audio_path": "/app/data/calls/call_abc123def456/audio.wav",
    "transcript_path": "/app/data/calls/call_abc123def456/transcript.jsonl",
    "segment_count": 12
  }
]
```

### `GET /calls/{call_id}`

Get call metadata.

**Response:** `CallOut`

### `GET /calls/{call_id}/transcript`

Get the call's transcript segments.

**Response:** `SegmentOut[]`
```json
[
  {
    "t0": 0.0,
    "t1": 2.5,
    "speaker": "user",
    "text": "నమస్తే, ఎలా ఉన్నారు?",
    "is_final": true
  },
  {
    "t0": 2.5,
    "t1": 5.0,
    "speaker": "bot",
    "text": "నేను బాగున్నాను, ధన్యవాదాలు.",
    "is_final": true
  }
]
```

### `GET /calls/{call_id}/audio`

Download the call's recorded audio as a WAV file.

**Response:** `audio/wav` binary

### `DELETE /calls/{call_id}`

Delete a call and its audio/transcript files.

**Response:** `{"ok": true}`

## Chunked Upload (Long Audio)

For audio files longer than a few minutes, upload in chunks. The backend
concatenates them and starts a background transcription task.

### `POST /upload/chunked/init`

Initialize a chunked upload session.

**Request body:**
```json
{
  "filename": "5hr_interview.wav",
  "total_chunks": 60,
  "call_id": null
}
```

**Response:**
```json
{
  "upload_id": "up_abc123def456",
  "call_id": "call_xyz789abc012"
}
```

### `POST /upload/chunked/{upload_id}/{chunk_index}`

Upload one chunk. **Multipart form data** with field `chunk`.

**Response:**
```json
{
  "received": 1,
  "total": 60
}
```

### `POST /upload/chunked/complete`

Finalize the upload. Concatenates chunks and kicks off background STT.

**Request body:**
```json
{
  "upload_id": "up_abc123def456",
  "speaker_id": "upload",
  "title": "5-hour interview"
}
```

**Response:**
```json
{
  "call_id": "call_xyz789abc012",
  "audio_path": "/app/data/calls/call_xyz789abc012/audio.wav",
  "transcription": "started"
}
```

After completion, poll `GET /calls/{call_id}` to watch `segment_count`
grow as transcription progresses.

## WebSocket: `/ws/call`

Bidirectional audio streaming for live calls.

### Client → Server

1. **Text frame (hello):**
   ```json
   {"type":"hello","speaker_id":"ramesh_telugu","title":"Test call","sample_rate":16000}
   ```

2. **Binary frames:** int16 PCM 16 kHz mono chunks (20-50 ms each)

3. **Text frame (end):**
   ```json
   {"type":"end"}
   ```

### Server → Client (text frames)

- `{"type":"hello","data":{"message":"connected"}}`
- `{"type":"vad","data":{"event":"speech_end","t0":..,"t1":..}}`
- `{"type":"transcript","data":{"t0":..,"t1":..,"speaker":"user","text":"..","language":"te","latency_ms":..,"is_final":true}}`
- `{"type":"llm","data":{"text":"..","source":"llm|fallback","latency_ms":..}}`
- `{"type":"audio","data":{"pcm_int16":[..],"sample_rate":24000,"engine":"xtts"}}`
- `{"type":"audio_end","data":{"t0":..,"t1":..,"latency_ms":..}}`
- `{"type":"status","data":{"message":"pipeline_started"}}`
- `{"type":"error","data":{"stage":"stt","message":".."}}`
- `{"type":"call_end","call_id":"call_..","data":{"call_id":"call_.."}}`

### Connection limits

- `MAX_CONCURRENT_CALLS` (default 1) — additional callers get HTTP 1013
  close with an error JSON
- 30-second timeout waiting for the initial `hello` frame

### Example (Python)

```python
import asyncio, json, websockets, numpy as np

async def call():
    async with websockets.connect("ws://localhost:8000/ws/call") as ws:
        await ws.send(json.dumps({
            "type": "hello",
            "speaker_id": "ramesh_telugu",
            "sample_rate": 16000,
        }))
        # Send audio
        pcm = np.zeros(16000, dtype=np.int16)  # 1s silence
        await ws.send(pcm.tobytes())
        await ws.send(json.dumps({"type": "end"}))
        # Receive responses
        async for raw in ws:
            msg = json.loads(raw)
            print(msg)

asyncio.run(call())
```
