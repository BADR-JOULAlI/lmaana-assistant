"""Explicit loopback bridge from the text API to a native Lmaana WSL worker."""

import httpx

from lmaana_assistant.asr.base import TranscriptResult
from lmaana_assistant.audio import AudioSample
from lmaana_assistant.config import Settings
from lmaana_assistant.errors import DependencyUnavailable, InferenceBusy, InferenceTimeout


class LocalLmaanaASR:
    name = "lmaana-2.4-local-worker"

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.model = settings.asr_model
        self.revision = settings.asr_revision
        self.client = client or httpx.Client(
            base_url=settings.asr_worker_url.rstrip("/"),
            timeout=settings.timeout_seconds + 10,
            trust_env=False,
            follow_redirects=False,
        )

    @property
    def reason(self) -> str | None:
        try:
            response = self.client.get("/health/ready", timeout=2)
            state = response.json()
            if state.get("model") != self.model or state.get("revision") != self.revision:
                return "The ASR worker model or revision does not match the configured Lmaana."
            if response.status_code != 200 or state.get("ready") is not True:
                return str(state.get("reason") or "The local Lmaana worker is not ready.")
            return None
        except (httpx.HTTPError, ValueError, AttributeError):
            return "The local Lmaana WSL worker is unreachable."

    def ready(self) -> bool:
        return self.reason is None

    def transcribe(self, audio: AudioSample) -> TranscriptResult:
        try:
            response = self.client.post(
                "/v1/transcriptions",
                files={"file": (audio.filename, audio.data, "audio/wav")},
            )
            if response.status_code == 429:
                raise InferenceBusy("The local Lmaana worker is busy.")
            if response.status_code == 504:
                raise InferenceTimeout("The local Lmaana worker timed out.")
            response.raise_for_status()
            result = TranscriptResult.model_validate(response.json())
            if result.model != self.model or result.revision != self.revision:
                raise DependencyUnavailable(
                    "The ASR worker returned a different model or revision."
                )
            return result
        except httpx.TimeoutException as exc:
            raise InferenceTimeout("The local Lmaana worker timed out.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise DependencyUnavailable(
                "The local Lmaana worker returned an invalid response."
            ) from exc

    def close(self) -> None:
        self.client.close()
