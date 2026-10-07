"""An explicit lexical development baseline and the planned Qwen adapter."""

import hashlib
import math
import re
from collections import Counter

from lmaana_assistant.config import Settings
from lmaana_assistant.errors import DependencyUnavailable
from lmaana_assistant.normalization import lexical_tokens


class LexicalEmbedder:
    """Deterministic hashed term vectors, not a semantic or multilingual model."""

    dimension = 4096
    fingerprint = "lexical-sha256-4096-v3"

    def encode_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for term, count in Counter(lexical_tokens(text)).items():
            index = int.from_bytes(hashlib.sha256(term.encode()).digest()[:4], "big")
            vector[index % self.dimension] += 1 + math.log(count)
        length = math.sqrt(sum(value * value for value in vector)) or 1
        return [value / length for value in vector]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.encode_query(text) for text in texts]

    def split(self, text: str, size: int, overlap: int) -> list[str]:
        spans = list(re.finditer(r"\S+", text))
        result = []
        for start in range(0, len(spans), size - overlap):
            end = min(start + size, len(spans))
            result.append(text[spans[start].start() : spans[end - 1].end()])
            if end == len(spans):
                break
        return result


class QwenEmbedder:
    dimension = 1024
    instruction = "Given a Moroccan public-service question, retrieve passages that answer it."

    def __init__(self, settings: Settings):
        if not re.fullmatch(r"[0-9a-f]{40}", settings.embedding_revision):
            raise DependencyUnavailable("Qwen requires a full EMBEDDING_REVISION commit hash.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DependencyUnavailable("Install the models extra to use Qwen embeddings.") from exc
        self.model = SentenceTransformer(
            settings.embedding_model,
            revision=settings.embedding_revision,
            device=settings.embedding_device,
            trust_remote_code=False,
        )
        self.fingerprint = (
            f"{settings.embedding_model}@{settings.embedding_revision}:1024:cosine:instruction-v1"
        )

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=8, normalize_embeddings=True).tolist()

    def encode_query(self, text: str) -> list[float]:
        query = f"Instruct: {self.instruction}\nQuery: {text}"
        return self.model.encode([query], normalize_embeddings=True)[0].tolist()

    def split(self, text: str, size: int, overlap: int) -> list[str]:
        offsets = self.model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        result = []
        for start in range(0, len(offsets), size - overlap):
            end = min(start + size, len(offsets))
            result.append(text[offsets[start][0] : offsets[end - 1][1]])
            if end == len(offsets):
                break
        return result


def make_embedder(settings: Settings):
    return LexicalEmbedder() if settings.embedding_backend == "lexical" else QwenEmbedder(settings)
