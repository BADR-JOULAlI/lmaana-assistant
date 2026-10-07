import io
import threading
import wave
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from lmaana_assistant.api.app import create_app
from lmaana_assistant.errors import InferenceTimeout
from lmaana_assistant.generation.adapters import ExcerptGenerator


def wav_bytes(*, seconds=1, sample_rate=16000, silent=False):
    frames = (b"\x00\x00" if silent else b"\x01\x00") * int(seconds * sample_rate)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)
    return stream.getvalue()


def test_empty_corpus_is_unready_but_process_is_live(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/v1/answers", json={"question": "hello"}).status_code == 503


def test_api_returns_citations_and_validates_input(settings, indexed):
    with TestClient(create_app(settings)) as client:
        health = client.get("/health/ready").json()
        assert health["ready"] is True
        assert health["sources"][0]["title"] == "Lmaana test guide"
        assert health["sources"][0]["verification_status"] == "verified"
        assert health["verified_source_count"] == 1
        response = client.post("/v1/answers", json={"question": "inscrire projet documentation"})
        assert response.status_code == 200
        assert response.json()["citations"]
        for question in ("", "   ", "x" * 1501):
            assert client.post("/v1/answers", json={"question": question}).status_code == 422
        assert client.post("/v1/answers", json={"question": "ok", "unknown": 1}).status_code == 422


def test_parallel_inference_is_rejected_and_slot_is_released(settings, indexed):
    entered, release = threading.Event(), threading.Event()

    class BlockingGenerator(ExcerptGenerator):
        def generate(self, query, evidence):
            entered.set()
            assert release.wait(5)
            return super().generate(query, evidence)

    with TestClient(create_app(settings, generator=BlockingGenerator())) as client:
        with ThreadPoolExecutor() as pool:
            first = pool.submit(
                client.post, "/v1/answers", json={"question": "inscrire projet documentation"}
            )
            try:
                assert entered.wait(5)
                second = client.post("/v1/answers", json={"question": "inscrire projet"})
                assert second.status_code == 429
                assert second.headers["Retry-After"] == "2"
            finally:
                release.set()
            assert first.result().status_code == 200
        assert client.post("/v1/answers", json={"question": "inscrire projet"}).status_code == 200


def test_timeout_is_an_outage_not_an_abstention(settings, indexed):
    class TimeoutGenerator(ExcerptGenerator):
        def generate(self, query, evidence):
            raise InferenceTimeout("timed out")

    with TestClient(create_app(settings, generator=TimeoutGenerator())) as client:
        response = client.post("/v1/answers", json={"question": "inscrire projet"})
        assert response.status_code == 504
        assert client.get("/health/ready").status_code == 503
        assert client.post("/v1/answers", json={"question": "inscrire projet"}).status_code == 503


def test_transcription_validates_audio_before_reporting_unavailable(settings, indexed):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/transcriptions", files={"file": ("question.wav", wav_bytes(), "audio/wav")}
        )
        assert response.status_code == 503
        assert "runtime is unavailable" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload, status",
    [(b"not-wav", 415), (wav_bytes(silent=True), 415), (wav_bytes(seconds=31), 415)],
)
def test_transcription_rejects_invalid_audio(settings, indexed, payload, status):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/transcriptions", files={"file": ("question.wav", payload, "audio/wav")}
        )
        assert response.status_code == status
