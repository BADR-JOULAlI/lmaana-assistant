"""ASR contracts shared by Lmaana, Whisper and the HTTP boundary."""

from datetime import datetime
from typing import Annotated, Protocol
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from lmaana_assistant.audio import AudioSample
from lmaana_assistant.contracts import StrictModel


class TranscriptSegment(StrictModel):
    text: str = Field(min_length=1, max_length=3000)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Transcript text must be nonblank.")
        return value

    @model_validator(mode="after")
    def ordered_timestamps(self):
        if self.end_ms < self.start_ms:
            raise ValueError("A transcript segment must end at or after its start.")
        return self


class TranscriptResult(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(min_length=1, max_length=6000)
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    audio_duration_ms: int = Field(ge=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    segments: list[TranscriptSegment] = Field(default_factory=list, max_length=256)
    warnings: list[str] = Field(default_factory=list, max_length=16)
    timings_ms: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        default_factory=dict
    )
    created_at: datetime | None = None

    @field_validator("text", "model", "revision")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Transcript text, model and revision must be nonblank.")
        return value

    @model_validator(mode="after")
    def segments_fit_audio(self):
        if any(segment.end_ms > self.audio_duration_ms for segment in self.segments):
            raise ValueError("Transcript segments must fit inside the audio duration.")
        return self


class ASRProvider(Protocol):
    name: str
    model: str
    revision: str

    def ready(self) -> bool: ...

    def transcribe(self, audio: AudioSample) -> TranscriptResult: ...

    def close(self) -> None: ...
