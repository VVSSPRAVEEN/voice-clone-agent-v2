# Evaluation Methodology

This document describes the three evaluation scripts in `evaluation/` and
how to interpret their outputs.

## Test assets

Place your test assets under `evaluation/`:

```
evaluation/
├── test_clips/         # Audio files (gitignored)
│   ├── telugu_10min.wav
│   ├── speaker_ref.wav
│   ├── q1.wav
│   ├── q2.wav
│   └── ...
├── ground_truth/       # Reference transcripts (gitignored)
│   ├── telugu_10min.txt
│   ├── tts_test_texts.txt
│   └── e2e_conversation.json
└── results/            # Output (gitignored, generated)
    ├── stt_<ts>.json
    ├── tts/tts_report_<ts>.json
    ├── tts/mos_template_<ts>.csv
    └── e2e/e2e_report_<ts>.json
```

## 1. STT evaluation (`evaluate_stt.py`)

Measures faster-whisper accuracy and latency on a long audio file.

### Usage

```bash
python evaluation/evaluate_stt.py \
    --audio evaluation/test_clips/telugu_10min.wav \
    --ground-truth evaluation/ground_truth/telugu_10min.txt \
    --language te \
    --chunk-seconds 30
```

### Metrics

| Metric              | Description                                              |
|---------------------|----------------------------------------------------------|
| `wer`               | Word error rate (Levenshtein on word level)              |
| `cer`               | Character error rate (Levenshtein on char level)         |
| `num_segments`      | Number of 30-s chunks transcribed                       |
| `total_audio_s`     | Total audio duration                                     |
| `total_processing_s`| Wall-clock time to transcribe the whole file             |
| `rtf`               | Real-time factor = `total_audio_s / total_processing_s`  |
| `latency_ms.min/mean/p95/max` | Per-chunk transcription latency               |

### Output

JSON report at `evaluation/results/stt_<timestamp>.json` containing the
above metrics plus per-segment details and the full hypothesis text.

### Interpreting WER for Telugu

Telugu WER on `faster-whisper medium` for clean conversational speech
should be in the 4-8% range. For noisy recordings or code-switched
(Telugu+English) speech, expect 10-20%. If WER is above 25%, check:

- Audio sample rate (must be 16 kHz or auto-resampled)
- Background noise level
- Speaker accent vs. training data

## 2. TTS evaluation (`evaluate_tts.py`)

Measures voice cloning quality: speaker similarity, intelligibility,
and latency.

### Usage

```bash
python evaluation/evaluate_tts.py \
    --reference evaluation/test_clips/speaker_ref.wav \
    --texts evaluation/ground_truth/tts_test_texts.txt \
    --language te \
    --output-dir evaluation/results/tts
```

The `tts_test_texts.txt` file should contain one test utterance per line:

```
నమస్తే, ఎలా ఉన్నారు?
This is a test of the voice cloning system.
నేను రమేష్ మాట్లాడుతున్నాను.
...
```

### Metrics

| Metric                     | Description                                              |
|----------------------------|----------------------------------------------------------|
| `mean_speaker_similarity`  | Cosine similarity between MFCC features of reference and generated audio. Range 0-1; higher is better. **Caveat:** MFCC-based similarity is a rough proxy; for true speaker similarity use ECAPA-TDNN embeddings (see below). |
| `mean_asr_wer`             | WER between input text and ASR output on the generated audio. Lower = more intelligible. |
| `mean_latency_ms`          | TTS synthesis latency per utterance                     |
| `mean_rtf`                 | Real-time factor = audio_seconds / latency_seconds      |

### MOS (Mean Opinion Score) template

A CSV template is generated at `evaluation/results/tts/mos_template_<ts>.csv`
with one row per utterance. After listening to each generated audio file,
fill in scores 1-5 on:

- `mos_naturalness` — how natural the speech sounds
- `mos_similarity` — how similar to the reference speaker
- `mos_intelligibility` — how easy to understand

Then compute the mean per column for the aggregate MOS.

### Improving speaker similarity measurement

The default MFCC-based similarity is fast but rough. For research-grade
evaluation, install `speechbrain` and replace the `_mfcc_features` function
in `evaluate_tts.py` with ECAPA-TDNN embeddings:

```python
from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def _ecapa_embed(pcm_int16, sr):
    import torch
    f32 = torch.tensor(pcm_int16.astype(np.float32) / 32768.0).unsqueeze(0)
    if sr != 16000:
        # resample
        ...
    return classifier.encode_batch(f32).squeeze().numpy()
```

ECAPA-TDNN cosine similarity above 0.7 is generally considered a good
voice clone.

## 3. End-to-end evaluation (`evaluate_e2e.py`)

Sends pre-recorded user utterances through the full pipeline (VAD → STT →
LLM → TTS) and measures end-to-end latency.

### Usage

```bash
python evaluation/evaluate_e2e.py \
    --speaker-id ramesh_telugu \
    --conversation evaluation/ground_truth/e2e_conversation.json
```

Conversation JSON format:

```json
[
  {"user_audio": "evaluation/test_clips/q1.wav", "expected_language": "te"},
  {"user_audio": "evaluation/test_clips/q2.wav", "expected_language": "en"},
  {"user_audio": "evaluation/test_clips/q3.wav", "expected_language": "te"}
]
```

### Metrics

| Metric                  | Description                                              |
|-------------------------|----------------------------------------------------------|
| `mean_turn_latency_ms`  | End of user audio → start of bot audio                   |
| `mean_stt_latency_ms`   | Per-utterance STT time                                   |
| `mean_llm_latency_ms`   | Per-utterance LLM time (or 50ms if rule-based fallback)  |
| `mean_tts_latency_ms`   | Per-utterance TTS time                                   |
| `language_match_rate`   | Fraction of turns where STT detected the expected language |
| `num_success/failed`    | Pipeline success rate                                    |

### Hallucination rate

The script does **not** automatically detect hallucinated LLM responses.
After running, open the report JSON and review each `llm_reply` field
manually. Mark each as:

- `0` — hallucinated (irrelevant, wrong language, nonsensical)
- `1` — sane (relevant response to the user's utterance)

Then compute `hallucination_rate = sum(0s) / total_replies`.

### Conversation coherence

Subjective. Listen to the full conversation (concatenate user_audio +
bot_reply audio for each turn) and rate 1-5 on:

- Context retention (does the bot reference earlier turns?)
- Topic coherence (does the reply make sense for the question?)
- Language consistency (does the bot stay in the user's language?)

## Benchmark script (`benchmark.py`)

Synthetic micro-benchmark for quick regression checks (no real audio
required). Measures raw model invocation cost on synthetic input.

```bash
python scripts/benchmark.py
# or with a real speaker reference for TTS benchmark:
python scripts/benchmark.py --speaker-ref data/speakers/ramesh_telugu/ref.wav
```

Use this to verify performance hasn't regressed after dependency updates.

## Acceptance criteria

For the system to be considered production-ready on a 3060 6GB:

| Metric                        | Target       |
|-------------------------------|--------------|
| STT WER (clean Telugu)        | < 8%         |
| STT WER (clean English)       | < 5%         |
| STT RTF (medium INT8)         | < 0.3        |
| TTS speaker similarity (MFCC) | > 0.85       |
| TTS RTF (XTTS v2)             | < 0.5        |
| TTS MOS (naturalness)         | > 3.5 / 5    |
| E2E turn latency              | < 1500 ms    |
| E2E success rate              | > 95%        |
| Hallucination rate            | < 5%         |
