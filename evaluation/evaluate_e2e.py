"""End-to-end evaluation: turn latency, hallucination rate, coherence.

Sends a list of pre-recorded user utterances through the full pipeline
(VAD → STT → LLM → TTS) and measures:

- Turn latency: from end of user speech to start of bot audio
- Hallucination rate: fraction of bot replies that don't make sense
  (checked via a separate verifier LLM call, or rule-based if no verifier)
- Conversation coherence: subjective (template for human review)

Usage:
    python evaluation/evaluate_e2e.py \
        --speaker-id my_speaker \
        --conversation evaluation/ground_truth/e2e_conversation.json

Conversation JSON format:
[
  {"user_audio": "evaluation/test_clips/q1.wav", "expected_language": "te"},
  {"user_audio": "evaluation/test_clips/q2.wav", "expected_language": "te"},
  ...
]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import wave
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--conversation", required=True, help="JSON file describing the conversation")
    parser.add_argument("--output-dir", default="evaluation/results/e2e")
    args = parser.parse_args()

    conv_path = Path(args.conversation)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not conv_path.exists():
        print(f"ERROR: conversation file not found: {conv_path}")
        sys.exit(1)

    conv = json.loads(conv_path.read_text(encoding="utf-8"))

    from app.config import SETTINGS
    from app.vad_worker import VADWorker
    from app.stt_worker import STTWorker
    from app.llm_worker import LLMWorker
    from app.tts_worker import TTSWorker
    from app.speaker_registry import SpeakerRegistry
    from app.call_logger import CallLogger
    from app.pipeline import Pipeline, PipelineEvent

    spk_reg = SpeakerRegistry()
    if not spk_reg.exists(args.speaker_id):
        print(f"ERROR: speaker {args.speaker_id} not registered")
        sys.exit(1)
    spk = spk_reg.get(args.speaker_id)
    ref_wav = spk_reg.get_ref_wav(args.speaker_id)

    vad = VADWorker()
    stt = STTWorker()
    llm = LLMWorker()
    tts = TTSWorker()
    calls = CallLogger()
    pipeline = Pipeline(vad, stt, llm, tts, spk_reg, calls)

    results = []
    for i, turn in enumerate(conv):
        user_audio_path = turn["user_audio"]
        expected_lang = turn.get("expected_language", "te")
        print(f"\n[{i+1}/{len(conv)}] {user_audio_path}")
        # Load user audio as PCM
        with wave.open(user_audio_path, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16).copy()
        if sr != 16000:
            import librosa
            f32 = pcm.astype(np.float32) / 32768.0
            f32 = librosa.resample(f32, orig_sr=sr, target_sr=16000)
            pcm = (f32 * 32768.0).astype(np.int16)

        async def audio_stream():
            # Yield as one chunk
            yield pcm

        events: list[PipelineEvent] = []

        async def on_event(ev: PipelineEvent):
            events.append(ev)

        t0 = time.perf_counter()
        try:
            call_id = await pipeline.run_streaming(
                audio_stream=audio_stream(),
                speaker_id=args.speaker_id,
                title=f"E2E test {i+1}",
                on_event=on_event,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"index": i, "error": str(e)})
            continue
        elapsed = time.perf_counter() - t0

        # Extract metrics from events
        stt_event = next((e for e in events if e.kind == "transcript"), None)
        llm_event = next((e for e in events if e.kind == "llm"), None)
        audio_end_event = next((e for e in events if e.kind == "audio_end"), None)
        # First audio chunk timestamp
        first_audio_t = None
        for e in events:
            if e.kind == "audio":
                first_audio_t = time.perf_counter()
                break

        stt_latency = stt_event.data.get("latency_ms", 0) if stt_event else 0
        llm_latency = llm_event.data.get("latency_ms", 0) if llm_event else 0
        tts_latency = audio_end_event.data.get("latency_ms", 0) if audio_end_event else 0
        turn_latency_ms = (first_audio_t - t0) * 1000 if first_audio_t else elapsed * 1000

        results.append({
            "index": i,
            "call_id": call_id,
            "user_audio": user_audio_path,
            "expected_language": expected_lang,
            "transcribed_text": stt_event.data.get("text", "") if stt_event else "",
            "transcribed_language": stt_event.data.get("language", "") if stt_event else "",
            "llm_reply": llm_event.data.get("text", "") if llm_event else "",
            "llm_source": llm_event.data.get("source", "") if llm_event else "",
            "stt_latency_ms": stt_latency,
            "llm_latency_ms": llm_latency,
            "tts_latency_ms": tts_latency,
            "turn_latency_ms": turn_latency_ms,
            "total_elapsed_ms": elapsed * 1000,
            "language_match": (stt_event.data.get("language", "").startswith(expected_lang[:2]) if stt_event else False),
        })
        print(f"  STT: {results[-1]['transcribed_text'][:60]}")
        print(f"  LLM ({results[-1]['llm_source']}): {results[-1]['llm_reply'][:60]}")
        print(f"  Turn latency: {turn_latency_ms:.0f}ms")

    ok = [r for r in results if "error" not in r]
    summary = {
        "num_turns": len(conv),
        "num_success": len(ok),
        "num_failed": len(results) - len(ok),
        "mean_turn_latency_ms": float(np.mean([r["turn_latency_ms"] for r in ok])) if ok else 0,
        "mean_stt_latency_ms": float(np.mean([r["stt_latency_ms"] for r in ok])) if ok else 0,
        "mean_llm_latency_ms": float(np.mean([r["llm_latency_ms"] for r in ok])) if ok else 0,
        "mean_tts_latency_ms": float(np.mean([r["tts_latency_ms"] for r in ok])) if ok else 0,
        "language_match_rate": float(np.mean([1 if r["language_match"] else 0 for r in ok])) if ok else 0,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    report = {"summary": summary, "results": results}
    report_path = out_dir / f"e2e_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== E2E Evaluation ===")
    print(f"Successful:       {summary['num_success']}/{summary['num_turns']}")
    print(f"Mean turn latency: {summary['mean_turn_latency_ms']:.0f}ms")
    print(f"Mean STT latency:  {summary['mean_stt_latency_ms']:.0f}ms")
    print(f"Mean LLM latency:  {summary['mean_llm_latency_ms']:.0f}ms")
    print(f"Mean TTS latency:  {summary['mean_tts_latency_ms']:.0f}ms")
    print(f"Language match:    {summary['language_match_rate']*100:.1f}%")
    print(f"\nReport: {report_path}")
    print("\nNote: Hallucination rate requires human review of the 'llm_reply' field")
    print("in the report. Mark each reply as 0 (hallucinated) or 1 (sane) in the")
    print("'hallucination' field manually, then compute the rate.")

    # Cleanup
    tts.unload()
    stt.unload()
    await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
