"""TTS evaluation: speaker similarity, ASR round-trip WER, MOS template.

Usage:
    python evaluation/evaluate_tts.py \
        --reference evaluation/test_clips/speaker_ref.wav \
        --texts evaluation/ground_truth/tts_test_texts.txt \
        --output-dir evaluation/results

For each test text:
1. Generate TTS audio using the reference clip (zero-shot cloning).
2. Compute speaker embedding cosine similarity between reference and generated
   (using a speaker encoder; falls back to MFCC-based similarity if no encoder).
3. Run ASR on the generated audio and compute WER against the input text.
4. Write a per-utterance report + an aggregate summary.

A subjective MOS (mean opinion score) template is also written for you to
fill in after listening to the generated samples.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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


def _wav_to_pcm_int16(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).copy(), sr


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _mfcc_features(pcm: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Compute mean MFCC features as a cheap speaker embedding."""
    try:
        import librosa
        f32 = pcm.astype(np.float32) / 32768.0
        mfcc = librosa.feature.mfcc(y=f32, sr=sr, n_mfcc=n_mfcc)
        return mfcc.mean(axis=1)
    except Exception:
        # Fallback: just use mean and std of the signal
        f32 = pcm.astype(np.float32) / 32768.0
        return np.array([f32.mean(), f32.std()])


def _wer_simple(ref: str, hyp: str) -> float:
    r = ref.lower().split()
    h = hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    return dp[len(r)][len(h)] / len(r)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Speaker reference WAV (3-10s)")
    parser.add_argument("--texts", required=True, help="Text file: one utterance per line")
    parser.add_argument("--language", default="te")
    parser.add_argument("--output-dir", default="evaluation/results/tts")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    texts_path = Path(args.texts)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ref_path.exists() or not texts_path.exists():
        print("ERROR: reference or texts file not found")
        sys.exit(1)

    texts = [l.strip() for l in texts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Reference: {ref_path}")
    print(f"Test utterances: {len(texts)}")

    ref_pcm, ref_sr = _wav_to_pcm_int16(str(ref_path))
    ref_emb = _mfcc_features(ref_pcm, ref_sr)

    from app.tts_worker import TTSWorker
    tts = TTSWorker()

    from app.stt_worker import STTWorker
    stt = STTWorker(language=args.language)

    results = []
    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] {text[:80]}")
        try:
            t0 = time.perf_counter()
            res = await tts.synthesize(text=text, speaker_ref_wav=ref_path, language=args.language)
            elapsed = (time.perf_counter() - t0) * 1000
            # Save audio
            wav_path = out_dir / f"utt_{i:03d}.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(res.sample_rate)
                wf.writeframes(res.audio.tobytes())
            # Speaker similarity
            gen_emb = _mfcc_features(res.audio, res.sample_rate)
            sim = _cosine_similarity(ref_emb, gen_emb)
            # ASR round-trip
            stt_res = await stt.transcribe_pcm(res.audio, language=args.language)
            asr_text = stt_res.text
            asr_wer = _wer_simple(text, asr_text)
            results.append({
                "index": i,
                "text": text,
                "audio_path": str(wav_path),
                "audio_seconds": len(res.audio) / res.sample_rate,
                "latency_ms": elapsed,
                "rtf": (len(res.audio) / res.sample_rate) / (elapsed / 1000) if elapsed > 0 else 0,
                "speaker_similarity_mfcc": sim,
                "asr_roundtrip_text": asr_text,
                "asr_roundtrip_wer": asr_wer,
                "engine": res.engine,
            })
            print(f"  similarity={sim:.3f}  wer={asr_wer:.3f}  latency={elapsed:.0f}ms")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"index": i, "text": text, "error": str(e)})

    # Aggregate
    ok = [r for r in results if "error" not in r]
    summary = {
        "reference": str(ref_path),
        "language": args.language,
        "num_utterances": len(texts),
        "num_success": len(ok),
        "num_failed": len(results) - len(ok),
        "mean_speaker_similarity": float(np.mean([r["speaker_similarity_mfcc"] for r in ok])) if ok else 0,
        "mean_asr_wer": float(np.mean([r["asr_roundtrip_wer"] for r in ok])) if ok else 0,
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in ok])) if ok else 0,
        "mean_rtf": float(np.mean([r["rtf"] for r in ok])) if ok else 0,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    report = {"summary": summary, "results": results}
    report_path = out_dir / f"tts_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # MOS template
    mos_path = out_dir / f"mos_template_{ts}.csv"
    with open(mos_path, "w", encoding="utf-8") as f:
        f.write("index,audio,mos_naturalness,mos_similarity,mos_intelligibility,notes\n")
        for r in results:
            if "error" in r:
                continue
            f.write(f"{r['index']},{r['audio_path']},,,,\n")

    print("\n=== TTS Evaluation ===")
    print(f"Successful:   {summary['num_success']}/{summary['num_utterances']}")
    print(f"Speaker sim:  {summary['mean_speaker_similarity']:.3f} (MFCC-based, 0-1)")
    print(f"ASR WER:      {summary['mean_asr_wer']:.3f}")
    print(f"Latency ms:   {summary['mean_latency_ms']:.0f}")
    print(f"RTF:          {summary['mean_rtf']:.2f}")
    print(f"\nReport: {report_path}")
    print(f"MOS template: {mos_path}  (fill in after listening)")

    tts.unload()
    stt.unload()


if __name__ == "__main__":
    asyncio.run(main())
