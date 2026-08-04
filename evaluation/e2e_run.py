import asyncio
import sys
import wave
import numpy as np

sys.path.insert(0, r"D:\voice-clone-agent\backend")

from app.config import SETTINGS
from app.vad_worker import VADWorker
from app.stt_worker import STTWorker
from app.llm_worker import LLMWorker
from app.tts_worker import TTSWorker
from app.speaker_registry import SpeakerRegistry
from app.call_logger import CallLogger
from app.pipeline import Pipeline


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != target_sr:
        import scipy.signal
        samples = scipy.signal.resample_poly(samples, target_sr, sr)
    return (samples * 32767.0).astype(np.int16)


async def main():
    print(f"data_dir={SETTINGS.data_dir}")
    vad = VADWorker()
    stt = STTWorker()
    llm = LLMWorker()
    tts = TTSWorker()
    reg = SpeakerRegistry(SETTINGS.speakers_dir)
    calls = CallLogger(SETTINGS.db_path)

    pipeline = Pipeline(vad, stt, llm, tts, reg, calls)
    samples = load_wav(r"D:\voice-clone-agent\evaluation\test_clips\test_tone.wav")

    chunk_ms = 100
    n = int(16000 * chunk_ms / 1000)
    chunks = [samples[i:i + n] for i in range(0, len(samples), n)]

    async def stream():
        for c in chunks:
            if c.size:
                yield c
        yield np.zeros(0, dtype=np.int16)

    events = []

    async def on_event(ev):
        events.append(ev)
        kind = ev.kind
        d = ev.data
        if kind == "transcript":
            print(f"[STT] {d.get('text')!r} ({d.get('latency_ms')}ms)")
        elif kind == "llm":
            print(f"[LLM] {d.get('text')!r} source={d.get('source')} ({d.get('latency_ms')}ms)")
        elif kind == "audio_end":
            print(f"[TTS] {d.get('latency_ms')}ms, {d.get('t1',0)-d.get('t0',0):.1f}s audio")
        elif kind == "error":
            print(f"[ERROR] {d}")

    call_id = await pipeline.run_streaming(
        stream(), "test1", title="e2e-test", on_event=on_event
    )
    print(f"call_id={call_id}")
    print(f"events: {[e.kind for e in events]}")
    tts.unload()


asyncio.run(main())
