"""STT evaluation: WER, CER, per-segment latency.

Usage:
    python evaluation/evaluate_stt.py \
        --audio evaluation/test_clips/telugu_10min.wav \
        --ground-truth evaluation/ground_truth/telugu_10min.txt \
        --language te

Output: JSON report + writes to evaluation/results/stt_<timestamp>.json
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List

# Make backend importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


def normalize_text(t: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = t.lower().strip()
    # Keep Telugu unicode range, English letters, digits
    t = re.sub(r"[^\u0c00-\u0c7fa-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def wer(ref: str, hyp: str) -> float:
    """Word error rate (Levenshtein on word level)."""
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    # DP table
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


def cer(ref: str, hyp: str) -> float:
    """Character error rate."""
    r = normalize_text(ref).replace(" ", "")
    h = normalize_text(hyp).replace(" ", "")
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
    parser.add_argument("--audio", required=True, help="Path to test audio file")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth transcript (plain text)")
    parser.add_argument("--language", default="te", help="Language code (te, en, auto)")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: audio not found: {audio_path}")
        sys.exit(1)

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"ERROR: ground truth not found: {gt_path}")
        sys.exit(1)
    ref_text = gt_path.read_text(encoding="utf-8")

    print(f"Audio: {audio_path}")
    print(f"Ground truth length: {len(ref_text)} chars")
    print(f"Language: {args.language}")

    from app.stt_worker import STTWorker
    stt = STTWorker(language=args.language)

    print("Transcribing (streaming chunks)...")
    t0 = time.perf_counter()
    segments: List[dict] = []
    async for res in stt.transcribe_file(str(audio_path), language=args.language,
                                         chunk_seconds=args.chunk_seconds):
        segments.append({
            "t0": res.start, "t1": res.end,
            "text": res.text, "language": res.language,
            "latency_ms": res.latency_ms,
        })
        print(f"  [{res.start:6.2f} → {res.end:6.2f}] ({res.latency_ms:5.0f}ms) {res.text[:80]}")
    total_elapsed = time.perf_counter() - t0

    hyp_text = " ".join(s["text"] for s in segments)
    wer_score = wer(ref_text, hyp_text)
    cer_score = cer(ref_text, hyp_text)
    latencies = [s["latency_ms"] for s in segments]

    report = {
        "audio_path": str(audio_path),
        "ground_truth_path": str(gt_path),
        "language": args.language,
        "num_segments": len(segments),
        "total_audio_s": segments[-1]["t1"] if segments else 0,
        "total_processing_s": total_elapsed,
        "rtf": (segments[-1]["t1"] if segments else 0) / total_elapsed if total_elapsed > 0 else 0,
        "wer": wer_score,
        "cer": cer_score,
        "latency_ms": {
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
            "mean": sum(latencies) / len(latencies) if latencies else 0,
            "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        },
        "segments": segments,
        "hypothesis": hyp_text,
        "reference": ref_text,
    }

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"stt_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== STT Evaluation ===")
    print(f"Segments:    {report['num_segments']}")
    print(f"Audio:       {report['total_audio_s']:.1f}s")
    print(f"Processing:  {report['total_processing_s']:.1f}s (RTF={report['rtf']:.2f})")
    print(f"WER:         {wer_score:.4f} ({wer_score*100:.2f}%)")
    print(f"CER:         {cer_score:.4f} ({cer_score*100:.2f}%)")
    print(f"Latency ms:  min={report['latency_ms']['min']:.0f}  "
          f"mean={report['latency_ms']['mean']:.0f}  "
          f"p95={report['latency_ms']['p95']:.0f}  "
          f"max={report['latency_ms']['max']:.0f}")
    print(f"\nReport: {out_path}")

    stt.unload()


if __name__ == "__main__":
    asyncio.run(main())
