"""Bounded WAV validation before an ASR runtime sees uploaded bytes."""

import wave
from dataclasses import dataclass
from io import BytesIO

from lmaana_assistant.errors import UnsupportedAudio

MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 30.0


@dataclass(frozen=True)
class AudioSample:
    data: bytes
    filename: str
    media_type: str | None
    sample_rate: int
    channels: int
    sample_width: int
    duration_seconds: float


def validate_wav(data: bytes, filename: str, media_type: str | None = None) -> AudioSample:
    if not data:
        raise UnsupportedAudio("The uploaded audio file is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise UnsupportedAudio("The uploaded audio file exceeds the 10 MiB limit.")
    try:
        with wave.open(BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
            duration = frames / sample_rate if sample_rate else 0.0
            if channels not in {1, 2} or sample_rate <= 0 or sample_width not in {1, 2, 3, 4}:
                raise UnsupportedAudio("The WAV format is not supported.")
            if duration <= 0 or duration > MAX_AUDIO_SECONDS:
                raise UnsupportedAudio("Audio duration must be between 0 and 30 seconds.")
            payload = wav.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise UnsupportedAudio("Only a valid PCM WAV file is accepted for this milestone.") from exc
    if not any(payload):
        raise UnsupportedAudio("The uploaded audio is silent.")
    return AudioSample(
        data=data,
        filename=filename or "audio.wav",
        media_type=media_type,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        duration_seconds=duration,
    )
