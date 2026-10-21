"""Native WSL ASR service with background preloading and observable readiness.

Run with uvicorn lmaana_assistant.asr.worker:create_worker --factory
--host 127.0.0.1 --port 8001 --workers 1 in the native runtime environment.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from lmaana_assistant.asr.base import TranscriptResult
from lmaana_assistant.asr.lmaana import LmaanaASR
from lmaana_assistant.audio import MAX_AUDIO_BYTES, validate_wav
from lmaana_assistant.config import Settings
from lmaana_assistant.errors import DependencyUnavailable, InferenceTimeout, UnsupportedAudio

logger = logging.getLogger(__name__)


def create_worker(settings: Settings | None = None, *, provider=None) -> FastAPI:
    config = settings or Settings()
    native = provider or LmaanaASR(config)
    inference_lock = threading.Lock()

    def load_model():
        try:
            native.warmup()
        except Exception:
            logger.exception("Lmaana worker failed to preload the checkpoint")
            raise

    @asynccontextmanager
    async def lifespan(app):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lmaana-load")
        app.state.load = executor.submit(load_model)
        try:
            yield
        finally:
            executor.shutdown(wait=True)
            native.close()

    app = FastAPI(title="Local Lmaana ASR worker", lifespan=lifespan)

    def status():
        load = app.state.load
        reason = "Lmaana is loading; the first startup may take several minutes."
        if load.done():
            if load.exception() is not None:
                reason = "Lmaana could not load. Inspect the native worker log."
            else:
                reason = None
        return {
            "ready": reason is None,
            "reason": reason,
            "model": native.model,
            "revision": native.revision,
        }

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        state = status()
        return JSONResponse(state, status_code=200 if state["ready"] else 503)

    @app.post("/v1/transcriptions", response_model=TranscriptResult)
    def transcribe(file: UploadFile = File(...)):  # noqa: B008
        try:
            audio = validate_wav(
                file.file.read(MAX_AUDIO_BYTES + 1), file.filename or "audio.wav", file.content_type
            )
        except UnsupportedAudio as exc:
            raise HTTPException(415, str(exc)) from exc
        if not status()["ready"]:
            raise HTTPException(503, status()["reason"])
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(429, "Inference is busy.", headers={"Retry-After": "2"})
        try:
            return native.transcribe(audio)
        except InferenceTimeout as exc:
            raise HTTPException(504, str(exc)) from exc
        except DependencyUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            inference_lock.release()

    return app
