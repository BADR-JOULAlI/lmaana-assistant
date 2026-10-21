import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

from lmaana_assistant.asr.base import TranscriptResult
from lmaana_assistant.asr.http import LocalLmaanaASR
from lmaana_assistant.asr.worker import create_worker
from lmaana_assistant.audio import validate_wav
from lmaana_assistant.errors import DependencyUnavailable, InferenceBusy, InferenceTimeout
from tests.test_api import wav_bytes


class FakeNative:
    model = "Lmaana/lmaana-2.4"
    revision = "537c5e5a0e2b1015b5ad10798be54f66bc3a2d7a"

    def warmup(self):
        pass

    def transcribe(self, audio):
        return TranscriptResult(
            text="شنو الوثائق؟",
            model=self.model,
            revision=self.revision,
            audio_duration_ms=round(audio.duration_seconds * 1000),
        )

    def close(self):
        pass


def test_worker_is_live_but_rejects_audio_until_model_is_loaded(settings):
    entered, release = threading.Event(), threading.Event()

    class LoadingNative(FakeNative):
        def warmup(self):
            entered.set()
            assert release.wait(5)

    app = create_worker(settings, provider=LoadingNative())
    with TestClient(app) as client:
        try:
            assert entered.wait(2)
            assert client.get("/health/live").status_code == 200
            assert client.get("/health/ready").status_code == 503
            assert (
                client.post(
                    "/v1/transcriptions", files={"file": ("q.wav", wav_bytes(), "audio/wav")}
                ).status_code
                == 503
            )
        finally:
            release.set()
        app.state.load.result(timeout=2)
        assert client.get("/health/ready").status_code == 200
        response = client.post(
            "/v1/transcriptions", files={"file": ("q.wav", wav_bytes(), "audio/wav")}
        )
        assert response.status_code == 200
        assert response.json()["text"] == "شنو الوثائق؟"


def test_worker_keeps_health_live_and_rejects_concurrent_inference(settings):
    entered, release = threading.Event(), threading.Event()

    class BlockingNative(FakeNative):
        def transcribe(self, audio):
            entered.set()
            assert release.wait(5)
            return super().transcribe(audio)

    app = create_worker(settings, provider=BlockingNative())
    with TestClient(app) as client, ThreadPoolExecutor() as executor:
        app.state.load.result(timeout=2)
        first = executor.submit(
            client.post,
            "/v1/transcriptions",
            files={"file": ("q.wav", wav_bytes(), "audio/wav")},
        )
        try:
            assert entered.wait(2)
            assert client.get("/health/live").status_code == 200
            busy = client.post(
                "/v1/transcriptions", files={"file": ("q.wav", wav_bytes(), "audio/wav")}
            )
            assert busy.status_code == 429
        finally:
            release.set()
        assert first.result(timeout=2).status_code == 200


def test_http_bridge_preserves_model_identity_and_transcript(settings):
    native = FakeNative()
    audio = validate_wav(wav_bytes(), "q.wav", "audio/wav")

    def handle(request):
        if request.url.path == "/health/ready":
            return httpx.Response(
                200, json={"ready": True, "model": native.model, "revision": native.revision}
            )
        assert b"RIFF" in request.read()
        return httpx.Response(200, json=native.transcribe(audio).model_dump(mode="json"))

    with httpx.Client(base_url=settings.asr_worker_url, transport=httpx.MockTransport(handle)) as c:
        provider = LocalLmaanaASR(settings, c)
        assert provider.ready()
        assert provider.transcribe(audio).text == "شنو الوثائق؟"


@pytest.mark.parametrize("status,error", [(429, InferenceBusy), (504, InferenceTimeout)])
def test_http_bridge_preserves_operational_errors(settings, status, error):
    with httpx.Client(
        base_url=settings.asr_worker_url,
        transport=httpx.MockTransport(lambda request: httpx.Response(status)),
    ) as c:
        with pytest.raises(error):
            LocalLmaanaASR(settings, c).transcribe(validate_wav(wav_bytes(), "q.wav"))


def test_http_bridge_rejects_another_model(settings):
    with httpx.Client(
        base_url=settings.asr_worker_url,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"ready": True, "model": "other"})
        ),
    ) as c:
        provider = LocalLmaanaASR(settings, c)
        assert not provider.ready()
        with pytest.raises(DependencyUnavailable):
            provider.transcribe(validate_wav(wav_bytes(), "q.wav"))
