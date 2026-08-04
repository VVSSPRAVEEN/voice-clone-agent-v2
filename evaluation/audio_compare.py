"""Acoustic comparison between two audio files.

Measures, for each file:
  - pitch: median/mean/std F0, voice range
  - pace:  speaking rate estimate (voiced syllables / sec)
  - smoothness: jitter %, shimmer %, voicing continuity
  - accent:  F1/F2 formant centroids, spectral centroid, MFCC profile

Then reports per-metric deltas and an overall similarity score (0-100).

Usage:
    python audio_compare.py <fileA.wav> <fileB.wav> [--sr 16000] [--fmin 60] [--fmax 400]

Examples:
    python audio_compare.py ../data/speakers/test1/ref.wav   ../data/speakers/test2/ref.wav
    python audio_compare.py speech_te.wav ../data/speakers/test2/ref.wav
"""
from __future__ import annotations

import argparse
import sys

import librosa
import numpy as np


def _load(path: str, sr: int) -> np.ndarray:
    y, _sr = librosa.load(path, sr=sr, mono=True)
    if _sr != sr:
        y = librosa.resample(y, orig_sr=_sr, target_sr=sr)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    # Normalize peak to 0.9 to be comparable across files
    peak = np.max(np.abs(y)) or 1.0
    return y / peak


def _f0_stats(y, sr, fmin, fmax):
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=512)
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool) & ~np.isnan(f0)
    f0v = f0[voiced]
    if f0v.size < 5:
        vratio = float(voiced.mean()) if voiced.size else 0.0
        return dict(n=0, voicing=vratio, median=0.0, mean=0.0, std=0.0, range=0.0, f0v=f0v)
    return dict(
        n=int(f0v.size),
        voicing=float(voiced.mean()),
        median=float(np.median(f0v)),
        mean=float(f0v.mean()),
        std=float(f0v.std()),
        range=float(f0v.max() - f0v.min()),
        f0v=f0v,
    )


def _pace_smoothness(y, sr, f0v):
    """Speaking pace + smoothness.

    Pace = voiced syllables/sec (approximated by F0 onset events per second).
    Jitter  = avg |dF0| / avg F0  (pitch instability)
    Shimmer = avg |dAmp| / avg Amp (loudness instability)
    """
    dur = len(y) / sr
    # Pitch onsets = rising/falling transitions in F0 curve (syllable-ish nuclei)
    if f0v.size > 5:
        d = np.abs(np.diff(f0v))
        pace = float((d > 20.0).sum() / dur)
        jitter = float(d.mean() / (f0v[:-1].mean() or 1.0) * 100)
    else:
        pace, jitter = 0.0, 0.0
    # Amplitude envelope instability (shimmer)
    env = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
    rms = np.sqrt((env ** 2).mean(axis=0))
    rms = rms[rms > 1e-4]
    if rms.size > 5:
        shimmer = float(np.abs(np.diff(rms)).mean() / (rms[:-1].mean() or 1e-6) * 100)
    else:
        shimmer = 0.0
    return dict(pace_sylps=pace, jitter_pct=jitter, shimmer_pct=shimmer)


def _formant_centroids(y, sr):
    """F1/F2 formant estimates via LPC spectral peak picking. Coarse but
    adequate for accent/timbre comparison."""
    import scipy.linalg
    import scipy.signal
    # Downsample for faster LPC (formants < 4kHz)
    if sr > 8000:
        y8 = librosa.resample(y, orig_sr=sr, target_sr=8000)
        s8 = 8000
    else:
        y8, s8 = y, sr
    order = 18
    frame = y8[: min(len(y8), s8 * 1)]  # use first second
    if frame.size < s8:
        frame = y8
    # LPC via autocorrelation method
    autocorr = np.correlate(frame, frame, mode="full")[frame.size - 1:]
    r = autocorr[: order + 1]
    toeplitz = scipy.linalg.toeplitz(r[:-1])
    try:
        a_coeff = scipy.linalg.solve(toeplitz, -r[1:])
    except Exception:
        return dict(f1=0.0, f2=0.0, f3=0.0, f2_minus_f1=0.0)
    a = np.concatenate(([1.0], a_coeff))
    roots = np.roots(a)
    roots = roots[abs(roots) < 1.0]
    angles = np.angle(roots)
    freqs = sorted(abs(angles) * (s8 / (2 * np.pi)))
    freqs = [f for f in freqs if 100 < f < 4000]
    f1 = freqs[0] if len(freqs) > 0 else 0.0
    f2 = freqs[1] if len(freqs) > 1 else 0.0
    f3 = freqs[2] if len(freqs) > 2 else 0.0
    return dict(f1=f1, f2=f2, f3=f3, f2_minus_f1=f2 - f1)


def _mfcc_profile(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=256)
    return mfcc.mean(axis=1), mfcc.std(axis=1)


def analyze(path, sr, fmin, fmax):
    y = _load(path, sr)
    f0 = _f0_stats(y, sr, fmin, fmax)
    ps = _pace_smoothness(y, sr, f0["f0v"])
    fc = _formant_centroids(y, sr)
    mmean, mstd = _mfcc_profile(y, sr)
    spec = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    features = {
        "duration_s": round(len(y) / sr, 2),
        **f0,
        **ps,
        **fc,
    }
    features.pop("f0v", None)
    return y, features, mmean, mstd, float(spec)


def _sim(mfccA_mean, mfccA_std, mfccB_mean, mfccB_std):
    # Cosine on means, blend with std-ratio penalty => 0..1
    def norm(v):
        m = np.linalg.norm(v)
        return v / m if m else v

    c = float(np.dot(norm(mfccA_mean), norm(mfccB_mean)))
    c = max(0.0, min(1.0, (c + 1) / 2))
    stdsim = 1.0 - min(1.0, abs(mfccA_std.mean() - mfccB_std.mean()) / (max(mfccA_std.mean(), mfccB_std.mean(), 1e-6)))
    return 100.0 * (0.7 * c + 0.3 * stdsim)


_KEY = ["median", "mean", "std", "range", "voicing",
        "pace_sylps", "jitter_pct", "shimmer_pct",
        "f1", "f2", "f3", "f2_minus_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fileA")
    ap.add_argument("fileB")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--fmin", type=float, default=60.0)
    ap.add_argument("--fmax", type=float, default=400.0)
    args = ap.parse_args()

    yA, fa, mA, sA, specA = analyze(args.fileA, args.sr, args.fmin, args.fmax)
    yB, fb, mB, sB, specB = analyze(args.fileB, args.sr, args.fmin, args.fmax)
    score = _sim(mA, sA, mB, sB)
    fa["spec_centroid"] = round(specA, 1)
    fb["spec_centroid"] = round(specB, 1)
    _KEY.append("spec_centroid")

    print(f"\n{'Metric':<18}{'A':>12}{'B':>12}{'Delta A-B':>16}")
    print("-" * 60)
    for k in _KEY:
        if k == "voicing":
            print(f"{k:<18}{fa[k]*100:>10.1f}%{fb[k]*100:>11.1f}%{_fmt(fa[k], fb[k], pct=True)}")
        elif "pct" in k:
            print(f"{k:<18}{fa[k]:>12.4f}{fb[k]:>12.4f}{_fmt(fa[k], fb[k])}")
        else:
            print(f"{k:<18}{fa[k]:>12.2f}{fb[k]:>12.2f}{_fmt(fa[k], fb[k])}")

    print("-" * 60)
    print(
        f"\nSIMILARITY: {score:.1f}/100  (nA={fa['n']} voiced "
        f"frames, nB={fb['n']} voiced frames)\n"
    )
    print("Reading guide:")
    print("  Pitch  : median/mean/std/range in Hz. Large std = prosodic variety.")
    print("  Pace   : voiced pitch-events per second (approx syllables/sec).")
    print("  Jitter : low (<1%) smooth, high (>3%) rough/robotic.")
    print("  Shimmer: low (<3%) steady amplitude, high = shaky.")
    print("  F1/F2  : vowel timbre — big F1+F2 distance difference implies a")
    print("           different accent / vowel space. F2-F1 wider = brighter accent.")
    print("  Sim    : MFCC timbre match (cosine). >80 same-ish voice, <60 differ.")


def _fmt(a, b, pct=False):
    d = b - a
    sign = "+" if d >= 0 else ""
    if pct:
        return f"{d*100:>+14.1f}pp"
    return f"{sign}{d:>14.3f}".replace(" ", "") if False else f"{sign}{d:>13.3f}"


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)