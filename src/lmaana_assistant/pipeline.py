"""The request pipeline shared by the application and offline tests."""

import json
from time import perf_counter
from uuid import uuid4

from lmaana_assistant.config import Settings
from lmaana_assistant.contracts import (
    CORPUS_SCHEMA_VERSION,
    QUERY_RULES_VERSION,
    SOURCE_POLICY_VERSION,
    AnswerResponse,
    Embedder,
    GeneratedAnswer,
    Generator,
    LanguagePreference,
    Statement,
)
from lmaana_assistant.errors import DependencyUnavailable, InvalidGeneration
from lmaana_assistant.generation.adapters import validate_citations
from lmaana_assistant.generation.excerpts import ABSTENTION, focused_excerpts
from lmaana_assistant.ingestion.sources import CHUNKER_VERSION
from lmaana_assistant.language import LANGUAGE_CLARIFICATION, message
from lmaana_assistant.normalization import lexical_tokens, process_query, topic_matches
from lmaana_assistant.retrieval.store import VectorStore
from lmaana_assistant.source_policy import (
    VERIFICATION_REQUIRED,
    source_reference,
    utc_today,
    verification_status,
)


class AnswerPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, generator: Generator):
        self.settings, self.embedder, self.generator = settings, embedder, generator
        if not settings.active_path.exists():
            raise DependencyUnavailable("No corpus is active. Run lmaana ingest first.")
        try:
            self.manifest = json.loads(settings.active_path.read_text(encoding="utf-8"))
            compatible = (
                self.manifest["schema_version"] == CORPUS_SCHEMA_VERSION
                and self.manifest.get("source_policy") == SOURCE_POLICY_VERSION
                and self.manifest["embedding"] == embedder.fingerprint
                and self.manifest["dimension"] == embedder.dimension
                and self.manifest["distance"] == "cosine"
                and self.manifest["query_rules"] == QUERY_RULES_VERSION
                and self.manifest.get("chunker") == CHUNKER_VERSION
            )
            self.release = self.manifest["release"]
        except (ValueError, KeyError, TypeError) as exc:
            raise DependencyUnavailable("The active corpus manifest is invalid.") from exc
        if not compatible:
            raise DependencyUnavailable(
                "Corpus and embedding configuration differ. Re-ingest first."
            )
        self.store = VectorStore(settings)
        if not self.store.exists(self.release):
            self.store.close()
            raise DependencyUnavailable("The active corpus collection is missing.")

    def answer(
        self, question: str, response_language: LanguagePreference = "auto"
    ) -> AnswerResponse:
        started = perf_counter()
        query = process_query(question, response_language)
        normalized_at = perf_counter()
        vector = self.embedder.encode_query(query.search_text)
        embedded_at = perf_counter()
        results = self.store.search(self.release, vector, self.settings.retrieval_k)
        candidates, seen = [], set()
        for item in results:
            if not topic_matches(query.normalized, item.chunk.source.topic):
                continue
            identity = (item.chunk.source.id, item.chunk.text)
            if item.score < self.settings.min_score or identity in seen:
                continue
            # Hash collisions alone must never make an unrelated lexical result relevant.
            if self.settings.embedding_backend == "lexical":
                if not set(lexical_tokens(query.search_text)) & set(
                    lexical_tokens(item.chunk.text)
                ):
                    continue
            seen.add(identity)
            candidates.append(item)
        today = utc_today()
        evidence = [
            item
            for item in candidates
            if verification_status(item.chunk.source, item.chunk.content_hash, today) == "verified"
        ][: self.settings.context_chunks]
        withheld = [
            item
            for item in candidates
            if verification_status(item.chunk.source, item.chunk.content_hash, today) != "verified"
        ]
        # Show a withheld reference only when it contains a relevant extract;
        # never turn a matching topic word into an alleged answer to fees/deadlines.
        related, related_ids = [], set()
        for item in withheld:
            if item.chunk.source.id in related_ids:
                continue
            if focused_excerpts(query, [item]).outcome == "source_excerpts":
                related.append(source_reference(item.chunk, today))
                related_ids.add(item.chunk.source.id)
            if len(related) == 3:
                break
        retrieved_at = perf_counter()
        warnings = []
        if self.settings.embedding_backend == "lexical":
            warnings.append(
                "Lexical development retrieval; multilingual semantic quality is untested."
            )
        if self.settings.generator_backend == "excerpt":
            warnings.append(
                "Excerpt mode returns original passages, not a synthesized Darija answer."
            )
        answer = GeneratedAnswer(
            outcome="insufficient_evidence", statements=[Statement(text=ABSTENTION)]
        )
        generation_failed = False
        if self.settings.generator_backend != "excerpt" and query.intent != "unknown":
            # A known topic alone does not establish that the requested detail is
            # present. Apply the same conservative answerability cues before an LLM.
            if focused_excerpts(query, evidence).outcome != "source_excerpts":
                evidence = []
        if evidence and query.language.reply != "und":
            for attempt in range(2):
                try:
                    answer = self.generator.generate(query, evidence)
                    validate_citations(answer, evidence)
                    break
                except InvalidGeneration:
                    answer = GeneratedAnswer(
                        outcome="insufficient_evidence", statements=[Statement(text=ABSTENTION)]
                    )
                    if attempt == 1:
                        generation_failed = True
                        warnings.append(
                            "The generated answer failed validation twice and was withheld."
                        )
        if related and not evidence and answer.outcome == "insufficient_evidence":
            answer = GeneratedAnswer(
                outcome="verification_required",
                statements=[Statement(text=VERIFICATION_REQUIRED)],
            )
        if related:
            warnings.append("Unverified references were excluded from the answer context.")
        if query.language.reply == "und":
            answer = GeneratedAnswer(
                outcome="clarification_required",
                statements=[Statement(text=LANGUAGE_CLARIFICATION)],
            )
            related = []
        elif answer.outcome in {"insufficient_evidence", "verification_required"}:
            answer = GeneratedAnswer(
                outcome=answer.outcome,
                statements=[
                    Statement(
                        text=message(
                            "generation_failed" if generation_failed else answer.outcome,
                            query.language.reply,
                        )
                    )
                ],
            )
        if query.language.confidence == "low":
            warnings.append(
                "Language detection is uncertain; select a reply language to override it."
            )
        completed = perf_counter()
        by_id = {item.chunk.id: item.chunk for item in evidence}
        cited_ids = list(
            dict.fromkeys(
                identifier
                for statement in answer.statements
                for identifier in statement.citation_ids
            )
        )
        citations = [source_reference(by_id[identifier], today) for identifier in cited_ids]
        labels = {identifier: number + 1 for number, identifier in enumerate(cited_ids)}
        rendered = []
        for statement in answer.statements:
            suffix = " ".join(f"[{labels[key]}]" for key in statement.citation_ids)
            rendered.append(f"{statement.text} {suffix}".strip())
        return AnswerResponse(
            request_id=str(uuid4()),
            outcome=answer.outcome,
            answer="\n\n".join(rendered),
            statements=answer.statements,
            citations=citations,
            related_sources=related,
            query=query,
            answer_language="source"
            if answer.outcome == "source_excerpts"
            else query.language.reply,
            generation_failed=generation_failed,
            corpus_release=self.release,
            embedding=self.embedder.fingerprint,
            generator=self.generator.name,
            warnings=warnings,
            timings_ms={
                "normalization": round((normalized_at - started) * 1000, 2),
                "embedding": round((embedded_at - normalized_at) * 1000, 2),
                "retrieval": round((retrieved_at - embedded_at) * 1000, 2),
                "generation": round((completed - retrieved_at) * 1000, 2),
                "total": round((perf_counter() - started) * 1000, 2),
            },
        )

    def close(self) -> None:
        self.store.close()
