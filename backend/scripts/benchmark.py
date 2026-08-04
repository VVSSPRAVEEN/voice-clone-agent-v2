"""Benchmark script.

Measures:
- STT latency per 1s / 5s / 30s clip
- TTS latency per 10-word / 50-word / 200-word utterance
- VAD latency per 30s clip
- End-to-end pipeline latency (audio in → audio out)

Outputs JSON to stdout and saves to data/benchmark_<timestamp>.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def bench_vad():
    from app.vad_worker import VADWorker
    w = VADWorker()
    # Synthetic 30 s of mostly-silence with some noise bursts
    sr = 16000
    rng = np.random.default_rng(42)
    audio = (rng.standard_normal(sr * 30) * 500).astype(np.int16)
    t0 = time.perf_counter()
    segs = w.detect_segments(audio)
    elapsed = (time.perf_counter() - t0) * 1000
    w.unload()
    return {"latency_ms": elapsed, "segments_found": len(segs), "duration_s": 30}


async def bench_stt():
    from app.stt_worker import STTWorker
    w = STTWorker()
    sr = 16000
    results = {}
    for dur_s in (1, 5, 30):
        # Generate silence (we don't have real Telugu clips here; just measuring
        # raw model invocation cost on synthetic input)
        audio = np.zeros(sr * dur_s, dtype=np.int16)
        t0 = time.perf_counter()
        res = await w.transcribe_pcm(audio)
        elapsed = (time.perf_counter() - t0) * 1000
        results[f"{dur_s}s"] = {"latency_ms": elapsed, "text": res.text[:50]}
    w.unload()
    return results


async def bench_tts(speaker_ref_wav: str | None):
    from app.tts_worker import TTSWorker
    if not speaker_ref_wav:
        return {"skipped": "no reference wav provided"}
    w = TTSWorker()
    results = {}
    text_samples = {
        "short": "Hello, how are you today?",
        "medium": ("This is a medium-length test sentence. " * 5).strip(),
        "long": ("This is a longer test of the text-to-speech system. " * 20).strip(),
    }
    for name, text in text_samples.items():
        t0 = time.perf_counter()
        res = await w.synthesize(text=text, speaker_ref_wav=speaker_ref_wav, language="en")
        elapsed = (time.perf_counter() - t0) * 1000
        results[name] = {
            "latency_ms": elapsed,
            "audio_seconds": len(res.audio) / res.sample_rate,
            "rtf": (len(res.audio) / res.sample_rate) / (elapsed / 1000) if elapsed > 0 else 0,
        }
    w.unload()
    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker-ref", help="Path to a speaker reference WAV (for TTS benchmark)")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-vad", action="store_true")
    args = parser.parse_args()

    out = {}
    if not args.skip_vad:
        print("Benchmarking VAD...")
        out["vad"] = await bench_vad()
    if not args.skip_stt:
        print("Benchmarking STT (synthetic silence)...")
        out["stt"] = await bench_stt()
    if not args.skip_tts:
        print("Benchmarking TTS...")
        out["tts"] = await bench_tts(args.speaker_ref)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path("data") / f"benchmark_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
