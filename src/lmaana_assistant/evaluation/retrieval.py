"""Isolated passage-ranking experiments; never open the application's index."""

import hashlib
import json
import math
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal

from pydantic import Field, field_validator, model_validator

from lmaana_assistant.contracts import QUERY_RULES_VERSION, Embedder, ReplyLanguage, StrictModel
from lmaana_assistant.evaluation.dataset import EvaluationDataset, Identifier, RetrievalCase
from lmaana_assistant.evaluation.metrics import latency_summary
from lmaana_assistant.evaluation.runner import evaluate
from lmaana_assistant.normalization import process_query
from lmaana_assistant.retrieval.vectors import unit_vector


class BenchmarkPassage(StrictModel):
    id: Identifier
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("id", "text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Passage IDs and text must be nonblank.")
        return value


class BenchmarkQuestion(StrictModel):
    id: Identifier
    question: str = Field(min_length=1, max_length=1500)
    language: ReplyLanguage
    relevant_ids: list[Identifier]

    @field_validator("id", "question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question IDs and text must be nonblank.")
        return value

    @field_validator("relevant_ids")
    @classmethod
    def unique_references(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(not value.strip() for value in values):
            raise ValueError("Relevant passage IDs must be unique and nonblank.")
        return values


class RetrievalBenchmark(StrictModel):
    schema_version: Literal[1] = 1
    dataset_id: Identifier
    notes: str = Field(min_length=1)
    passages: list[BenchmarkPassage] = Field(min_length=1)
    questions: list[BenchmarkQuestion] = Field(min_length=1)
    retrieval_k: list[int] = Field(default_factory=lambda: [1, 3, 5])

    @field_validator("retrieval_k")
    @classmethod
    def cutoffs(cls, values: list[int]) -> list[int]:
        if not values or any(value < 1 for value in values) or len(values) != len(set(values)):
            raise ValueError("Retrieval cutoffs must be positive, unique and nonempty.")
        return sorted(values)

    @model_validator(mode="after")
    def consistent_dataset(self):
        if not self.dataset_id.strip() or not self.notes.strip():
            raise ValueError("Dataset identity and experiment notes must be nonblank.")
        for records in (self.passages, self.questions):
            ids = [record.id for record in records]
            if len(ids) != len(set(ids)):
                raise ValueError("Passage and question IDs must be unique within each list.")
        available = {passage.id for passage in self.passages}
        for question in self.questions:
            if not set(question.relevant_ids) <= available:
                raise ValueError(f"Unknown relevant passage in question {question.id!r}.")
        return self


def benchmark_retrieval(
    dataset: RetrievalBenchmark,
    embedder: Embedder,
    *,
    min_score: float = 0.18,
    query_mode: Literal["raw", "expanded"] = "expanded",
) -> dict:
    """Rank pre-segmented passages by cosine, without source policies or an LLM.

    The report keeps both unfiltered and thresholded rankings. Neither is the
    application answerability/currentness guard, and thresholds are not confidence.
    """
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("The cosine threshold must be finite and between zero and one.")
    if query_mode not in {"raw", "expanded"}:
        raise ValueError("Query mode must be raw or expanded.")
    if embedder.dimension < 1:
        raise ValueError("The embedding dimension must be positive.")
    started = perf_counter()
    vectors = embedder.encode_documents([passage.text for passage in dataset.passages])
    if len(vectors) != len(dataset.passages):
        raise ValueError("Embedding count does not match the number of passages.")
    documents = [unit_vector(vector, embedder.dimension) for vector in vectors]
    document_ms = (perf_counter() - started) * 1000
    rows, filtered, unfiltered = [], [], []
    for question in dataset.questions:
        started = perf_counter()
        query = process_query(question.question)
        text = query.search_text if query_mode == "expanded" else question.question
        vector = unit_vector(embedder.encode_query(text), embedder.dimension)
        ranked = []
        if vector is not None:
            for passage, document in zip(dataset.passages, documents, strict=True):
                if document is not None:
                    score = math.fsum(x * y for x, y in zip(vector, document, strict=True))
                    ranked.append({"id": passage.id, "score": max(-1.0, min(1.0, score))})
        ranked.sort(key=lambda item: (-item["score"], item["id"]))
        ranked = ranked[: max(dataset.retrieval_k)]
        accepted = [item for item in ranked if item["score"] >= min_score]
        filtered.append(
            RetrievalCase(
                id=question.id,
                relevant_ids=question.relevant_ids,
                ranked_ids=[item["id"] for item in accepted],
            )
        )
        unfiltered.append(
            RetrievalCase(
                id=question.id,
                relevant_ids=question.relevant_ids,
                ranked_ids=[item["id"] for item in ranked],
            )
        )
        rows.append(
            {
                "id": question.id,
                "question": question.question,
                "language": question.language,
                "detected_language": query.language.detected,
                "intent": query.intent,
                "query_text": text,
                "relevant_ids": question.relevant_ids,
                "ranked": ranked,
                "accepted": accepted,
                "latency_ms": (perf_counter() - started) * 1000,
            }
        )

    def metrics(cases: list[RetrievalCase]) -> dict:
        return evaluate(
            EvaluationDataset(
                dataset_id=dataset.dataset_id, retrieval=cases, retrieval_k=dataset.retrieval_k
            )
        )["retrieval"]

    canonical = json.dumps(dataset.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return {
        "schema_version": 1,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "notes": dataset.notes,
        "scope": (
            "Isolated embedding retrieval on supplied passages; no active index, topic/currentness "
            "filters, answerability guards, ASR or generation. Not an end-to-end quality score."
        ),
        "embedding": embedder.fingerprint,
        "embedding_instruction": getattr(embedder, "instruction", None),
        "dimension": embedder.dimension,
        "query_rules": QUERY_RULES_VERSION,
        "query_mode": query_mode,
        "min_score": min_score,
        "retrieval_k": dataset.retrieval_k,
        "passage_count": len(dataset.passages),
        "zero_document_vector_count": sum(vector is None for vector in documents),
        "ranking": "cosine descending; exact ties ordered by passage ID",
        "document_encoding_ms": document_ms,
        "query_latency": latency_summary([row["latency_ms"] for row in rows]),
        "timing_scope": "Single sequential run; excludes model loading and document encoding.",
        "retrieval": metrics(filtered),
        "unfiltered_retrieval": metrics(unfiltered),
        "by_language": {
            language: {
                "retrieval": metrics(
                    [
                        case
                        for case, row in zip(filtered, rows, strict=True)
                        if row["language"] == language
                    ]
                ),
                "unfiltered_retrieval": metrics(
                    [
                        case
                        for case, row in zip(unfiltered, rows, strict=True)
                        if row["language"] == language
                    ]
                ),
            }
            for language in sorted({question.language for question in dataset.questions})
        },
        "cases": rows,
    }
