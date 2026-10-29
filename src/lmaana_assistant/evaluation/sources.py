"""Build local retrieval probes from a pinned source snapshot and authored labels."""

import hashlib
import json
import re
from typing import Literal

from pydantic import Field, model_validator

from lmaana_assistant.contracts import SourceManifest, StrictModel
from lmaana_assistant.evaluation.dataset import Identifier
from lmaana_assistant.evaluation.retrieval import (
    BenchmarkPassage,
    BenchmarkQuestion,
    RetrievalBenchmark,
    dataset_sha256,
)
from lmaana_assistant.ingestion.sources import MAX_BYTES, extract_sections


class SourcePassage(StrictModel):
    id: Identifier
    section_index: int = Field(default=0, ge=0)
    start_marker: str = Field(min_length=1)
    end_marker: str | None = Field(default=None, min_length=1)


class SourceQuestion(BenchmarkQuestion):
    family: Identifier


class SourceBenchmarkSpec(StrictModel):
    schema_version: Literal[1] = 1
    dataset_id: Identifier
    notes: str = Field(min_length=1)
    annotation_status: Literal["draft"]
    source_id: Identifier
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passages: list[SourcePassage] = Field(min_length=1)
    questions: list[SourceQuestion] = Field(min_length=1)
    retrieval_k: list[int] = Field(default_factory=lambda: [1, 3, 5])

    @model_validator(mode="after")
    def consistent_annotations(self):
        # Reuse the dataset's structural rules before any source extraction.
        RetrievalBenchmark(
            dataset_id=self.dataset_id,
            notes=self.notes,
            passages=[BenchmarkPassage(id=p.id, text=p.start_marker) for p in self.passages],
            questions=[
                BenchmarkQuestion(**q.model_dump(exclude={"family"})) for q in self.questions
            ],
            retrieval_k=self.retrieval_k,
        )
        families = {}
        for question in self.questions:
            if not question.family.strip():
                raise ValueError("Question families must be nonblank.")
            relevant = set(question.relevant_ids)
            if question.family in families and families[question.family] != relevant:
                raise ValueError("Translated questions in one family must share relevance labels.")
            families[question.family] = relevant
        if any(
            not p.start_marker.strip() or (p.end_marker is not None and not p.end_marker.strip())
            for p in self.passages
        ):
            raise ValueError("Source markers must be nonblank.")
        return self


def _unique_marker(text: str, marker: str) -> re.Match:
    pattern = r"\s+".join(re.escape(word) for word in marker.split())
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError("A benchmark source marker is missing or ambiguous.")
    return matches[0]


def prepare_source_benchmark(
    spec: SourceBenchmarkSpec, manifest: SourceManifest, raw: bytes
) -> tuple[RetrievalBenchmark, dict]:
    if len(raw) > MAX_BYTES:
        raise ValueError("Source exceeds the 10 MiB preparation limit.")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec.source_sha256:
        raise ValueError("The source snapshot hash does not match the benchmark specification.")
    matching = [source for source in manifest.sources if source.id == spec.source_id]
    if len(matching) != 1:
        raise ValueError("The benchmark must select exactly one source in the source manifest.")
    source = matching[0]
    recorded = source.verification.content_sha256
    if recorded is not None and recorded != digest:
        raise ValueError("The source manifest and benchmark snapshot hashes differ.")
    sections = extract_sections(source, raw)
    passages, provenance, spans = [], [], {}
    for selection in spec.passages:
        if selection.section_index >= len(sections):
            raise ValueError("A benchmark passage references a missing source section.")
        location, text = sections[selection.section_index]
        start = _unique_marker(text, selection.start_marker).start()
        end = (
            _unique_marker(text, selection.end_marker).start()
            if selection.end_marker is not None
            else len(text)
        )
        if end <= start:
            raise ValueError("Benchmark passage markers are out of order.")
        selected = text[start:end]
        start += len(selected) - len(selected.lstrip())
        end -= len(selected) - len(selected.rstrip())
        for old_start, old_end in spans.get(selection.section_index, []):
            if start < old_end and old_start < end:
                raise ValueError("Benchmark passage selections must not overlap.")
        spans.setdefault(selection.section_index, []).append((start, end))
        passage = BenchmarkPassage(id=selection.id, text=text[start:end])
        passages.append(passage)
        provenance.append(
            {
                "passage_id": passage.id,
                "source_id": source.id,
                "source_url": str(source.url),
                "source_sha256": digest,
                "location": location,
                "section_index": selection.section_index,
                "start_character": start,
                "end_character": end,
                "text_sha256": hashlib.sha256(passage.text.encode()).hexdigest(),
            }
        )
    spec_hash = hashlib.sha256(
        json.dumps(spec.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    notes = (
        f"{spec.notes}\nAnnotation status: draft; source-snapshot development probes. "
        f"Source: {source.id}; URL: {source.url}; SHA-256: {digest}; spec SHA-256: {spec_hash}. "
        "No current administrative applicability or held-out performance is established."
    )
    dataset = RetrievalBenchmark(
        dataset_id=spec.dataset_id,
        notes=notes,
        passages=passages,
        questions=[BenchmarkQuestion(**q.model_dump(exclude={"family"})) for q in spec.questions],
        retrieval_k=spec.retrieval_k,
    )
    return dataset, {
        "schema_version": 1,
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": dataset_sha256(dataset),
        "spec_sha256": spec_hash,
        "annotation_status": spec.annotation_status,
        "split": "development",
        "source_snapshot_sha256": digest,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(manifest.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest(),
        "passages": provenance,
        "question_families": {q.id: q.family for q in spec.questions},
        "split_policy": "Keep each source snapshot and all translations of a family in one split.",
        "scope": "Snapshot relevance only; no service calls, source-currentness or answer grading.",
    }
