import io
import wave

import pytest

from lmaana_assistant.audio import validate_wav
from lmaana_assistant.errors import UnsupportedAudio


def make_wav(seconds=1, *, silent=False):
    stream = io.BytesIO()
    frames = (b"\x00\x00" if silent else b"\x01\x00") * int(seconds * 16000)
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(frames)
    return stream.getvalue()


def test_validate_wav_returns_audio_metadata():
    sample = validate_wav(make_wav(), "question.wav", "audio/wav")

    assert sample.sample_rate == 16000
    assert sample.channels == 1
    assert sample.duration_seconds == 1


@pytest.mark.parametrize("payload", [b"not-a-wave", make_wav(silent=True), make_wav(31)])
def test_validate_wav_rejects_invalid_or_unsafe_audio(payload):
    with pytest.raises(UnsupportedAudio):
        validate_wav(payload, "question.wav", "audio/wav")
