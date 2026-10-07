"""ASR contracts shared by Lmaana, Whisper and the HTTP boundary."""

from datetime import datetime
from typing import Protocol
from uuid import uuid4

from pydantic import Field

from lmaana_assistant.audio import AudioSample
from lmaana_assistant.contracts import StrictModel


class TranscriptSegment(StrictModel):
    text: str = Field(min_length=1, max_length=3000)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class TranscriptResult(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(min_length=1, max_length=6000)
    model: str
    revision: str
    audio_duration_ms: int = Field(ge=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    segments: list[TranscriptSegment] = Field(default_factory=list, max_length=256)
    warnings: list[str] = Field(default_factory=list, max_length=16)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    created_at: datetime | None = None


class ASRProvider(Protocol):
    name: str
    model: str
    revision: str

    def ready(self) -> bool: ...

    def transcribe(self, audio: AudioSample) -> TranscriptResult: ...

    def close(self) -> None: ...
