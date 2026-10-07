"""Local Qwen3 via Ollama's native API; no downloads or cloud fallback at request time."""

import json

import httpx

from lmaana_assistant.config import Settings
from lmaana_assistant.contracts import (
    GENERATION_POLICY_VERSION,
    Evidence,
    GeneratedAnswer,
    ProcessedQuery,
)
from lmaana_assistant.errors import DependencyUnavailable, InferenceTimeout, InvalidGeneration
from lmaana_assistant.generation.adapters import (
    generation_messages,
    generation_schema,
    parse_generation,
)


class OllamaGenerator:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        digest = f"@{settings.ollama_model_digest}" if settings.ollama_model_digest else ""
        self.name = f"ollama:{settings.ollama_model}{digest}#{GENERATION_POLICY_VERSION}"
        self.client = client or httpx.Client(
            base_url=settings.ollama_url.rstrip("/"),
            timeout=settings.timeout_seconds,
            trust_env=False,
            follow_redirects=False,
        )

    def ready(self) -> bool:
        """Check presence and metadata without loading weights or pulling a model."""
        try:
            response = self.client.get("/api/tags", timeout=3)
            response.raise_for_status()
            tag = self.settings.ollama_model
            canonical = tag if ":" in tag.rsplit("/", 1)[-1] else f"{tag}:latest"
            models = response.json()["models"]
            model = next(item for item in models if item["name"] == canonical)
            if model.get("remote_host") or model.get("remote_model"):
                return False
            if (
                self.settings.ollama_model_digest
                and model.get("digest") != self.settings.ollama_model_digest
            ):
                return False
            response = self.client.post("/api/show", json={"model": tag}, timeout=3)
            response.raise_for_status()
            info = response.json()
            return (
                not info.get("remote_host")
                and not info.get("remote_model")
                and not info.get("messages")
                and info["details"]["family"] == "qwen3"
                and "completion" in info["capabilities"]
                and info["model_info"]["qwen3.context_length"] >= self.settings.context_tokens
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError, StopIteration):
            return False

    def generate(self, query: ProcessedQuery, evidence: list[Evidence]) -> GeneratedAnswer:
        # Qwen3 uses byte-level BPE. UTF-8 bytes are a deliberately conservative
        # content budget, not an exact token count. Reserve template and output space.
        # Ollama's truncate/shift=false also prevents silent loss of evidence.
        selected = list(evidence)
        while selected:
            messages = generation_messages(query, selected)
            content_bound = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
            if (
                content_bound + 512 + self.settings.max_output_tokens
                <= self.settings.context_tokens
            ):
                break
            selected.pop()
        if not selected:
            raise InvalidGeneration("The question and evidence exceed the configured context.")
        try:
            response = self.client.post(
                "/api/chat",
                json={
                    "model": self.settings.ollama_model,
                    "messages": messages,
                    "format": generation_schema(selected),
                    "stream": False,
                    "think": False,
                    "truncate": False,
                    "shift": False,
                    "keep_alive": "5m",
                    "options": {
                        "num_ctx": self.settings.context_tokens,
                        "num_predict": self.settings.max_output_tokens,
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 20,
                        "min_p": 0,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object.")
        except httpx.TimeoutException as exc:
            raise InferenceTimeout("The local Ollama generation service timed out.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise DependencyUnavailable(
                "The local Ollama generation service is unavailable."
            ) from exc
        if result.get("remote_host") or result.get("remote_model"):
            raise DependencyUnavailable("Remote Ollama inference is not permitted.")
        if result.get("done") is not True or result.get("done_reason") != "stop":
            raise InvalidGeneration("The generated answer was incomplete or truncated.")
        message = result.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise InvalidGeneration("Ollama returned an invalid assistant message.")
        if message.get("tool_calls") or message.get("thinking"):
            raise InvalidGeneration("Unexpected tools or reasoning in the model response.")
        return parse_generation(message.get("content"), query, selected)

    def close(self) -> None:
        self.client.close()
