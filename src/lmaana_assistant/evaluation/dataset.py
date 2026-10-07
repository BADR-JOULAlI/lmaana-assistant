"""Explicit offline annotations; no model downloads or invented claim labels."""

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from lmaana_assistant.contracts import AnswerOutcome, ReplyLanguage, StrictModel

Identifier = Annotated[str, Field(min_length=1, max_length=200)]


class AsrCase(StrictModel):
    id: Identifier
    reference: str
    hypothesis: str


class RetrievalCase(StrictModel):
    id: Identifier
    relevant_ids: list[Identifier]
    ranked_ids: list[Identifier]

    @field_validator("relevant_ids", "ranked_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(not value.strip() for value in values):
            raise ValueError("Passage IDs must be unique and nonblank.")
        return values


class ResponseCase(StrictModel):
    id: Identifier
    expected_outcome: AnswerOutcome
    actual_outcome: AnswerOutcome
    expected_reply_language: ReplyLanguage | Literal["source"]
    actual_reply_language: ReplyLanguage | Literal["source"]
    generation_failed: bool = False
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    # Optional HUMAN labels. Absence must remain unevaluated, never perfect.
    claim_support: list[Literal["supported", "partial", "unsupported", "contradicted"]] | None = (
        None
    )

    @model_validator(mode="after")
    def consistent_answer(self):
        if self.generation_failed and self.actual_outcome in {"answered", "source_excerpts"}:
            raise ValueError("A failed generation cannot count as a displayed answer.")
        if self.claim_support is not None and self.actual_outcome != "answered":
            raise ValueError("Claim-support annotations require an actual generated answer.")
        return self


class EvaluationDataset(StrictModel):
    schema_version: Literal[1] = 1
    dataset_id: Identifier
    notes: str = ""
    # Provenance of outputs is operator-supplied, not inferred by metric code.
    model_profile: str | None = None
    corpus_release: str | None = None
    asr: list[AsrCase] = Field(default_factory=list)
    retrieval: list[RetrievalCase] = Field(default_factory=list)
    responses: list[ResponseCase] = Field(default_factory=list)
    retrieval_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    cer_remove_whitespace: bool = False

    @field_validator("retrieval_k")
    @classmethod
    def positive_unique_k(cls, values: list[int]) -> list[int]:
        if not values or any(value < 1 for value in values) or len(values) != len(set(values)):
            raise ValueError("Retrieval cutoffs must be positive, unique and nonempty.")
        return sorted(values)

    @model_validator(mode="after")
    def identifiable_cases(self):
        if not any((self.asr, self.retrieval, self.responses)):
            raise ValueError("An evaluation needs at least one annotated case.")
        for cases in (self.asr, self.retrieval, self.responses):
            ids = [case.id for case in cases]
            if len(ids) != len(set(ids)) or any(not value.strip() for value in ids):
                raise ValueError("Case IDs must be unique and nonblank within each task.")
        if not self.dataset_id.strip():
            raise ValueError("The dataset ID cannot be blank.")
        return self
