# Transcription API

The voice boundary is intentionally separate from `/v1/answers`:

```text
POST /v1/transcriptions  (multipart/form-data, field: file)
        ↓
validated WAV
        ↓
ASRProvider.transcribe
        ↓
TranscriptResult
        ↓
user correction in the UI
        ↓
POST /v1/answers
```

The first implementation accepts non-empty PCM WAV files up to 10 MiB and 30
seconds. It reports `415` for malformed, silent or unsupported audio, `429` when
the single inference slot is busy, `503` when the Lmaana runtime is unavailable,
and `504` for a transcription timeout.

Successful responses contain:

```json
{
  "id": "transcription-id",
  "text": "...",
  "model": "Lmaana/lmaana-2.4",
  "revision": "pinned-commit",
  "audio_duration_ms": 4200,
  "confidence": null,
  "segments": [],
  "warnings": ["Greedy CTC decoding; confidence is not calibrated."],
  "timings_ms": {"total": 1234.5, "load": 900.0, "inference": 334.5}
}
```

Transcript text and model/revision identifiers must be nonblank. Segment timestamps
must satisfy `0 <= start_ms <= end_ms <= audio_duration_ms`, and timing values must
be finite and nonnegative. A malformed HTTP worker transcript is treated as a
dependency failure (`503`); it is never submitted to retrieval as a valid question.
Returned transcript text is preserved, including meaningful spacing and script.

The API does not persist uploaded audio. The adapter writes one temporary WAV for
the native runtime and removes it after the inference attempt. The raw transcript
must remain editable before it is submitted to the retrieval pipeline. The first
request may include a several-minute model load; the configured timeout applies
to audio inference after the model is ready, while load and inference timings are
reported separately.
