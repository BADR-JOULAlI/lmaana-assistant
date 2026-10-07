"""Single-worker FastAPI application with explicit dependency readiness."""

import json
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from lmaana_assistant import __version__
from lmaana_assistant.asr.factory import make_asr
from lmaana_assistant.audio import MAX_AUDIO_BYTES, validate_wav
from lmaana_assistant.config import Settings
from lmaana_assistant.contracts import (
    LANGUAGE_POLICY_VERSION,
    SOURCE_POLICY_VERSION,
    AnswerRequest,
    AnswerResponse,
    Source,
    TranscriptionResponse,
)
from lmaana_assistant.errors import DependencyUnavailable, InferenceTimeout, UnsupportedAudio
from lmaana_assistant.generation.adapters import make_generator
from lmaana_assistant.pipeline import AnswerPipeline
from lmaana_assistant.retrieval.embeddings import make_embedder
from lmaana_assistant.source_policy import utc_today, verification_status

logger = logging.getLogger("lmaana.requests")


def create_app(settings: Settings | None = None, *, embedder=None, generator=None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = None
        app.state.startup_error = None
        app.state.inference_lock = threading.Lock()
        app.state.healthy = True
        app.state.asr = make_asr(config)
        app.state.generator = generator or make_generator(config)
        try:
            app.state.pipeline = AnswerPipeline(
                config, embedder or make_embedder(config), app.state.generator
            )
        except (DependencyUnavailable, OSError, ValueError) as exc:
            app.state.startup_error = str(exc)
            logger.warning("Application not ready: %s", exc)
        try:
            yield
        finally:
            if app.state.pipeline:
                app.state.pipeline.close()
            app.state.generator.close()
            app.state.asr.close()

    app = FastAPI(title=config.app_name, version=__version__, lifespan=lifespan)

    def readiness() -> dict:
        pipeline = app.state.pipeline
        reason = app.state.startup_error
        if not app.state.healthy:
            reason = "Inference timed out; restart the API after checking the model service."
        elif not reason and not app.state.generator.ready():
            reason = (
                "Configured generator/model/context is unavailable. Check the local model server."
            )
        sources = []
        if pipeline:
            for record in pipeline.manifest.get("sources", []):
                source = Source.model_validate(record)
                status = verification_status(
                    source, source.verification.content_sha256 or "", utc_today()
                )
                sources.append(
                    {
                        "title": source.title,
                        "publisher": source.publisher,
                        "verification_status": status,
                        "checked_on": str(source.verification.checked_on)
                        if source.verification.checked_on
                        else None,
                        "verification_note": source.verification.note,
                    }
                )
        return {
            "ready": reason is None and pipeline is not None,
            "reason": reason,
            "embedding_backend": config.embedding_backend,
            "generator_backend": config.generator_backend,
            "corpus_release": pipeline.release if pipeline else None,
            "source_count": pipeline.manifest["source_count"] if pipeline else 0,
            "chunk_count": pipeline.manifest["chunk_count"] if pipeline else 0,
            "source_policy": SOURCE_POLICY_VERSION,
            "language_policy": LANGUAGE_POLICY_VERSION,
            "generator": app.state.generator.name,
            "asr_backend": config.asr_backend,
            "asr_model": app.state.asr.model,
            "asr_ready": app.state.asr.ready(),
            "sources": sources,
            "verified_source_count": sum(s["verification_status"] == "verified" for s in sources),
        }

    @app.get("/health/live")
    def live():
        return {"status": "alive", "app": config.app_name, "version": __version__}

    @app.get("/health/ready")
    def ready():
        state = readiness()
        return JSONResponse(state, status_code=200 if state["ready"] else 503)

    @app.post("/v1/transcriptions", response_model=TranscriptionResponse)
    async def transcribe(file: UploadFile = File(...)):  # noqa: B008
        data = await file.read(MAX_AUDIO_BYTES + 1)
        try:
            audio = validate_wav(data, file.filename or "audio.wav", file.content_type)
        except UnsupportedAudio as exc:
            raise HTTPException(415, str(exc)) from exc
        if not app.state.asr.ready():
            raise HTTPException(
                503,
                "The configured Lmaana ASR runtime is unavailable; "
                "audio was validated but not transcribed.",
            )
        if not app.state.inference_lock.acquire(blocking=False):
            raise HTTPException(
                429, "Inference is busy. Try again shortly.", headers={"Retry-After": "2"}
            )
        try:
            return app.state.asr.transcribe(audio)
        except InferenceTimeout as exc:
            app.state.healthy = False
            raise HTTPException(504, str(exc)) from exc
        except DependencyUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            app.state.inference_lock.release()

    @app.post("/v1/answers", response_model=AnswerResponse)
    def answer(request: AnswerRequest):
        # Sync route runs in Starlette's thread pool. The slot is held until work ends,
        # including when the caller disconnects; it is never released by a timer alone.
        if not app.state.inference_lock.acquire(blocking=False):
            raise HTTPException(
                429, "Inference is busy. Try again shortly.", headers={"Retry-After": "2"}
            )
        try:
            state = readiness()
            if not state["ready"]:
                raise HTTPException(503, state["reason"] or "Corpus unavailable.")
            result = app.state.pipeline.answer(request.question, request.response_language)
            logger.info(
                json.dumps(
                    {
                        "request_id": result.request_id,
                        "outcome": result.outcome,
                        "corpus_release": result.corpus_release,
                        "embedding": result.embedding,
                        "generator": result.generator,
                        "timings_ms": result.timings_ms,
                    }
                )
            )
            return result
        except InferenceTimeout as exc:
            app.state.healthy = False
            raise HTTPException(504, str(exc)) from exc
        except DependencyUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            app.state.inference_lock.release()

    return app
