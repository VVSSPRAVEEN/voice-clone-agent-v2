"""Download all model weights into ./data/models/.

Run via:  python scripts/download_models.py

Downloads:
- faster-whisper medium (CT2 INT8 quantized at load time)
- Silero VAD ONNX
- Coqui XTTS v2 (multilingual zero-shot)
- (optional) sherpa-onnx Telugu + English VITS models
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Ensure the app config is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS = {
    "faster-whisper": {
        "repo": "Systran/faster-whisper-medium",
        "local_dir": "models/faster-whisper",
        "method": "huggingface_ct2",
    },
    "silero-vad": {
        "repo": "snakers4/silero-vad",
        "local_dir": "models/silero-vad",
        "method": "url",
        "files": [
            ("silero_vad.onnx",
             "https://github.com/snakers4/silero-vad/raw/refs/heads/master/files/silero_vad.onnx"),
        ],
    },
    "xtts": {
        "repo": "coqui/XTTS-v2",
        "local_dir": "models/hf",
        "method": "coqui_tts",
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    },
    "sherpa-telugu": {
        "repo": "ai4bharat/indic-tts",
        "local_dir": "models/sherpa-onnx/telugu-vits",
        "method": "skip",  # Optional; enable with --include-sherpa
        "files": [],
    },
}


def download_url(url: str, dest: Path) -> None:
    print(f"  -> {url}")
    print(f"  -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  (already exists, skipping)")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "voice-clone-agent/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            buf = r.read(1024 * 1024)
            if not buf:
                break
            f.write(buf)
    print(f"  done ({dest.stat().st_size / 1e6:.1f} MB)")


def download_silero(root: Path) -> None:
    print("[silero-vad]")
    spec = MODELS["silero-vad"]
    for fname, url in spec["files"]:
        download_url(url, root / spec["local_dir"] / fname)


def download_faster_whisper(root: Path) -> None:
    print("[faster-whisper medium]")
    from huggingface_hub import snapshot_download
    local = root / MODELS["faster-whisper"]["local_dir"]
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODELS["faster-whisper"]["repo"],
        local_dir=str(local),
        local_dir_use_symlinks=False,
        allow_patterns=["*.bin", "*.json", "*.txt", "tokenizer/*", "vocabulary.*"],
    )
    print(f"  done -> {local}")


def download_xtts(root: Path) -> None:
    print("[Coqui XTTS v2]")
    # Coqui will download into the HF cache under COQUI_TOS_AGREED=1
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("HF_HOME", str(root / "models" / "hf"))
    from TTS.api import TTS
    model_name = MODELS["xtts"]["model_name"]
    print(f"  Loading {model_name} (one-time download)...")
    tts = TTS(model_name)
    print(f"  done")


def download_sherpa_telugu(root: Path) -> None:
    print("[sherpa-onnx Telugu VITS] (optional)")
    try:
        import sherpa_onnx
    except ImportError:
        print("  sherpa-onnx not installed; skipping")
        return
    # Use sherpa-onnx's own model downloader if available, else direct URLs.
    # The Telugu VITS model from AI4Bharat IndicTTS:
    base = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
    files = [
        ("vits-ai4bharat-te.tar.bz2",
         f"{base}/vits-ai4bharat-te.tar.bz2"),
    ]
    import tarfile, tempfile
    tmp_dir = root / "models" / "sherpa-onnx" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest_dir = root / "models" / "sherpa-onnx" / "telugu-vits"
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"  (already exists at {dest_dir})")
        return
    for fname, url in files:
        download_url(url, tmp_dir / fname)
        # Extract
        with tarfile.open(tmp_dir / fname, "r:bz2") as tar:
            tar.extractall(dest_dir.parent)
        (tmp_dir / fname).unlink()
    print(f"  done -> {dest_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-sherpa", action="store_true",
                        help="Also download sherpa-onnx Telugu VITS model")
    parser.add_argument("--skip-xtts", action="store_true",
                        help="Skip Coqui XTTS download (large)")
    parser.add_argument("--skip-stt", action="store_true",
                        help="Skip faster-whisper download")
    parser.add_argument("--skip-vad", action="store_true",
                        help="Skip Silero VAD download")
    args = parser.parse_args()

    # Resolve root: data dir two levels up from this script (./data)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_root = project_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"Project root: {project_root}")
    print(f"Data root:    {data_root}")

    if not args.skip_vad:
        download_silero(data_root)
    if not args.skip_stt:
        download_faster_whisper(data_root)
    if not args.skip_xtts:
        download_xtts(data_root)
    if args.include_sherpa:
        download_sherpa_telugu(data_root)

    print("\nAll requested models downloaded.")


if __name__ == "__main__":
    main()
