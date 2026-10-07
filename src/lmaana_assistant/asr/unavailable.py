"""Explicit provider used until a native ASR runtime is installed."""

from lmaana_assistant.asr.base import TranscriptResult
from lmaana_assistant.audio import AudioSample
from lmaana_assistant.config import Settings
from lmaana_assistant.errors import DependencyUnavailable


class UnavailableASR:
    name = "unavailable"

    def __init__(self, settings: Settings):
        self.model = settings.asr_model
        self.revision = settings.asr_revision

    def ready(self) -> bool:
        return False

    def transcribe(self, audio: AudioSample) -> TranscriptResult:
        raise DependencyUnavailable(
            "The native Lmaana ASR runtime is not installed; transcription is unavailable."
        )

    def close(self) -> None:
        pass
