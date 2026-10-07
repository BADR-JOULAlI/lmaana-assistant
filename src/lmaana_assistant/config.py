"""Explicit, environment-driven profiles; no silent model fallback."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LMAANA_", env_file=".env", extra="ignore")

    app_name: str = "Lmaana Assistant"
    data_dir: Path = Path("data")
    # Reserved for the next voice milestone. The current application is text-only.
    asr_backend: Literal["unavailable", "lmaana"] = "unavailable"
    asr_model: Literal["Lmaana/lmaana-2.4"] = "Lmaana/lmaana-2.4"
    asr_revision: str = "537c5e5a0e2b1015b5ad10798be54f66bc3a2d7a"
    # Local path is intentionally opt-in and machine-specific; it is never committed.
    asr_model_dir: Path | None = None
    embedding_backend: Literal["lexical", "qwen"] = "lexical"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_revision: str = ""
    embedding_device: str = "cpu"
    generator_backend: Literal["excerpt", "llamacpp", "ollama"] = "excerpt"
    llm_url: str = "http://127.0.0.1:8080"
    llm_model: str = "lmaana-qwen3"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = Field(default="qwen3:4b", min_length=1)
    ollama_model_digest: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    context_tokens: int = Field(default=4096, ge=1024, le=32768)
    max_output_tokens: int = Field(default=512, ge=64, le=2048)
    timeout_seconds: float = Field(default=60, gt=0, le=300)
    retrieval_k: int = Field(default=10, ge=1, le=30)
    context_chunks: int = Field(default=4, ge=1, le=8)
    min_score: float = Field(default=0.18, ge=0, le=1)

    @model_validator(mode="after")
    def validate_profile(self):
        if self.max_output_tokens >= self.context_tokens:
            raise ValueError("Output budget must be smaller than the context window.")
        if self.context_chunks > self.retrieval_k:
            raise ValueError("context_chunks must not exceed retrieval_k.")
        if self.generator_backend == "ollama":
            url = urlsplit(self.ollama_url)
            if (
                url.scheme != "http"
                or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.username is not None
                or url.password is not None
                or url.path not in {"", "/"}
                or url.query
                or url.fragment
            ):
                raise ValueError("Ollama must use an HTTP loopback URL without credentials.")
            if "cloud" in self.ollama_model.lower():
                raise ValueError("This profile only supports local Ollama models.")
        return self

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpora"

    @property
    def active_path(self) -> Path:
        return self.corpus_dir / "active.json"
